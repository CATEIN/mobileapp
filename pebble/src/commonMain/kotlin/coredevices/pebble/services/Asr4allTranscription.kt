package coredevices.pebble.services

import co.touchlab.kermit.Logger
import coredevices.pebble.services.asr4all.Asr4allModelStore
import io.rebble.libpebblecommon.voice.TranscriptionProvider
import io.rebble.libpebblecommon.voice.TranscriptionResult
import io.rebble.libpebblecommon.voice.TranscriptionWord
import io.rebble.libpebblecommon.voice.VoiceEncoderInfo
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.filter
import kotlinx.coroutines.launch
import kotlinx.coroutines.withTimeoutOrNull
import kotlin.math.abs
import kotlin.math.log10
import kotlin.math.roundToInt
import kotlin.math.sqrt
import kotlin.time.TimeSource

/**
 * On-device streaming transcription with FUTO's ASR4ALL. Audio is decoded while the watch is still
 * recording, so only the last window and the punctuation pass remain once it stops.
 */
class Asr4allTranscription(
    private val store: Asr4allModelStore,
) : TranscriptionProvider {
    companion object {
        private val logger = Logger.withTag("Asr4allTranscription")
    }

    override suspend fun canServeSession(): Boolean = store.isReady()

    @OptIn(ExperimentalUnsignedTypes::class)
    override suspend fun transcribe(
        encoderInfo: VoiceEncoderInfo,
        audioFrames: Flow<UByteArray>,
        isNotificationReply: Boolean,
        onPartialResult: suspend (String) -> Unit,
    ): TranscriptionResult {
        require(encoderInfo is VoiceEncoderInfo.Speex) {
            "ASR4ALL transcription only supports Speex encoding, got ${encoderInfo::class.simpleName}"
        }
        if (encoderInfo.sampleRate != 16_000L) {
            return TranscriptionResult.Error("Unsupported sample rate ${encoderInfo.sampleRate}")
        }
        return try {
            coroutineScope {
                val pcm = Channel<FloatArray>(Channel.UNLIMITED)
                val level = AudioLevel()
                // Conflated so a slow watch link only ever gets the latest text.
                val partial = MutableStateFlow("")
                val partials = launch {
                    partial.filter { it.isNotEmpty() }.collect { onPartialResult(it) }
                }
                // Collected apart from the model so slow inference never backs up the watch link.
                val audio = launch {
                    try {
                        decodeSpeex(encoderInfo, audioFrames) {
                            level.add(it)
                            pcm.send(it)
                        }
                    } finally {
                        pcm.close()
                    }
                }
                val text = async(Dispatchers.Default) {
                    store.openSession().use { session ->
                        for (chunk in pcm) {
                            if (session.decoder.accept(chunk)) {
                                partial.value = session.decoder.text()
                            }
                        }
                        session.decoder.finish()
                        session.decoder.text()
                    }
                }
                audio.join()
                val audioEnded = TimeSource.Monotonic.markNow()
                logger.i { "Audio ended: $level" }
                val result = withTimeoutOrNull(PEBBLE_TRANSCRIPTION_BUDGET) { text.await() }
                partials.cancelAndJoin()
                if (result == null) {
                    text.cancel()
                    TranscriptionResult.ConnectionError("Transcription timed out")
                } else {
                    logger.i { "Transcribed in ${audioEnded.elapsedNow()} after audio ended" }
                    TranscriptionResult.Success(
                        words = result.split(" ").filter { it.isNotEmpty() }.map {
                            TranscriptionWord(word = it, confidence = 0.9f)
                        }
                    )
                }
            }
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            logger.e(e) { "ASR4ALL transcription failed" }
            TranscriptionResult.Error("Transcription failed: ${e.message}")
        }
    }

    @OptIn(ExperimentalUnsignedTypes::class)
    private suspend fun decodeSpeex(
        encoderInfo: VoiceEncoderInfo.Speex,
        frames: Flow<UByteArray>,
        emit: suspend (FloatArray) -> Unit,
    ) {
        WatchSpeexDecoder.create(encoderInfo).use { speex ->
            frames.collect { frame ->
                emit(pcm16ToFloat(speex.decode(frame.asByteArray())))
            }
        }
    }
}

/** Loudness summary of a session, to tell a quiet recording apart from a recognition miss. */
internal class AudioLevel {
    private var samples = 0L
    private var sumSquares = 0.0
    private var peak = 0f

    fun add(chunk: FloatArray) {
        for (s in chunk) {
            sumSquares += s * s
            peak = maxOf(peak, abs(s))
        }
        samples += chunk.size
    }

    private fun dbfs(x: Double) = if (x > 0) (20 * log10(x)).roundToInt() else -100

    override fun toString() = "${samples / 16} ms, rms ${
        dbfs(if (samples > 0) sqrt(sumSquares / samples) else 0.0)
    } dBFS, peak ${dbfs(peak.toDouble())} dBFS"
}

internal fun pcm16ToFloat(pcm: ByteArray): FloatArray = FloatArray(pcm.size / 2) { i ->
    val sample = (pcm[2 * i].toInt() and 0xFF) or (pcm[2 * i + 1].toInt() shl 8)
    sample.toShort() / 32768f
}
