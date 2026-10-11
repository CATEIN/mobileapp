package coredevices.coreapp.transcription

import android.os.Process
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import coredevices.pebble.services.Asr4allTranscription
import coredevices.pebble.services.asr4all.AndroidAsr4allPlatform
import coredevices.pebble.services.asr4all.Asr4allModelSpec
import coredevices.pebble.services.asr4all.Asr4allModelState
import coredevices.pebble.services.asr4all.Asr4allModelStore
import coredevices.util.CoreConfig
import coredevices.util.CoreConfigFlow
import coredevices.util.STTConfig
import io.ktor.client.HttpClient
import io.ktor.client.engine.okhttp.OkHttp
import io.rebble.libpebblecommon.connection.AppContext
import io.rebble.libpebblecommon.voice.TranscriptionResult
import io.rebble.libpebblecommon.voice.VoiceEncoderInfo
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import kotlinx.serialization.json.Json
import org.junit.Assume
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.Parameterized
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.minutes
import kotlin.time.TimeSource

/**
 * Runs the real ASR4ALL models. Downloads each on first run. Expected transcripts come from
 * FUTO's reference `asr4all_executorch.py` on the same audio.
 */
@RunWith(Parameterized::class)
class Asr4allTranscriptionTest(private val spec: Asr4allModelSpec) {
    companion object {
        private const val TAG = "Asr4allTest"

        @JvmStatic
        @Parameterized.Parameters(name = "{index}")
        fun models() = Asr4allModelSpec.all

        private const val HELLO =
            "Hey, can you pick up some milk and eggs on the way home? I will be back around seven thirty."

        // file to (reference transcript, word edits allowed). The int8 kernels differ slightly per
        // CPU, which flips the ambiguous words of the Eric clip between ISAs.
        private val recorded = mapOf(
            Asr4allModelSpec.Small to mapOf(
                "eval_reminder_30min.raw" to ("Remind me thirty minutes to send out more ring inites." to 0),
                "eval_set_alarm_750am.raw" to ("Set an alarm for seven fifty AM." to 0),
                "eval_shopping_list_shrimp.raw" to ("Add shrimp and corn starch to my shopping list." to 0),
                "eval_text_eric_shrimp.raw" to ("Text, Eric, the kids have breaded shrimp. Period. We need rim. Maybe the Argentinian one would be good." to 2),
                "eval_note_jared_size10.raw" to ("Jared as a size ten ring." to 0),
            ),
            Asr4allModelSpec.Medium to mapOf(
                "eval_reminder_30min.raw" to ("Remind me in thirty minutes to send out more ring invites." to 0),
                "eval_set_alarm_750am.raw" to ("Set an alarm for seven fifty AM." to 0),
                "eval_shopping_list_shrimp.raw" to ("Add shrimp and corn starts to my shopping list." to 0),
                "eval_text_eric_shrimp.raw" to ("Text, Eric, the kids have breaded shrimp. Period. We need rim. Maybe the Argentinian one would be good." to 3),
                "eval_note_jared_size10.raw" to ("Jared is a size ten ring." to 0),
            ),
            Asr4allModelSpec.Large to mapOf(
                "eval_reminder_30min.raw" to ("Remind me in thirty minutes to send out more ring invites." to 0),
                "eval_set_alarm_750am.raw" to ("Said an alarm for seven fifty Am." to 2),
                "eval_shopping_list_shrimp.raw" to ("Add shrimp and cornstarch to my shopping list." to 0),
                "eval_text_eric_shrimp.raw" to ("Text Eric, the kids have breaded shrimp period. We need raw shrimp. Maybe the Argentinian one would be good." to 3),
                "eval_note_jared_size10.raw" to ("Jared is a size ten ring." to 0),
            ),
        )
    }

    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val store = Asr4allModelStore(
        platform = AndroidAsr4allPlatform(AppContext(instrumentation.targetContext)),
        httpClient = HttpClient(OkHttp),
        json = Json { ignoreUnknownKeys = true },
        coreConfigFlow = CoreConfigFlow(
            MutableStateFlow(CoreConfig(sttConfig = STTConfig(asr4allModel = spec.id)))
        ),
    )

    private fun asset(name: String) = instrumentation.context.assets.open(name).use { it.readBytes() }

    @Before
    fun ensureModel() = runBlocking {
        Assume.assumeTrue("ExecuTorch needs a 64-bit process", Process.is64Bit())
        if (!store.isReady()) {
            store.download()
            withTimeout(5.minutes) {
                store.state.first { it is Asr4allModelState.Ready || it is Asr4allModelState.Failed }
            }
        }
        assertEquals(Asr4allModelState.Ready, store.state.value)
    }

    /** Speex frames as the watch sends them (1-byte header + payload), paced in real time. */
    @OptIn(ExperimentalUnsignedTypes::class)
    @Test
    fun transcribesWatchSpeexStream() = runBlocking {
        val data = asset("asr4all_hello.speexframes")
        val frames = buildList {
            var i = 0
            while (i < data.size) {
                val length = data[i].toInt() and 0xFF
                add(data.copyOfRange(i, i + 1 + length).asUByteArray())
                i += 1 + length
            }
        }
        val audio = flow {
            for (frame in frames) {
                emit(frame)
                delay(20)
            }
        }
        val partials = mutableListOf<String>()
        val start = TimeSource.Monotonic.markNow()
        val result = Asr4allTranscription(store).transcribe(
            encoderInfo = VoiceEncoderInfo.Speex(
                sampleRate = 16_000,
                version = "1.2rc2",
                bitRate = 9_800,
                bitstreamVersion = 4,
                frameSize = 320,
            ),
            audioFrames = audio,
            isNotificationReply = false,
            onPartialResult = { partials.add(it) },
        )
        val elapsed = start.elapsedNow()
        assertIs<TranscriptionResult.Success>(result)
        val text = result.words.joinToString(" ") { it.word }
        Log.i(TAG, "${spec.id}: ${frames.size} frames (${frames.size * 20} ms audio) -> \"$text\" in $elapsed")
        partials.forEach { Log.i(TAG, "${spec.id} partial: $it") }
        assertEquals(HELLO, text)
        assertTrue(partials.size >= 3, "expected live partial results, got $partials")
    }

    @Test
    fun transcribesRecordedSpeech() = runBlocking {
        for ((file, reference) in recorded.getValue(spec)) {
            val (transcript, allowedEdits) = reference
            val pcm = ByteBuffer.wrap(asset(file)).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
            val samples = FloatArray(pcm.remaining()) { pcm.get(it) / 32768f }
            val start = TimeSource.Monotonic.markNow()
            val text = store.openSession().use { session ->
                // 0.5 s chunks, like a capture loop.
                samples.asList().chunked(8_000).forEach { session.decoder.accept(it.toFloatArray()) }
                session.decoder.finish()
                session.decoder.text()
            }
            val elapsed = start.elapsedNow()
            Log.i(TAG, "${spec.id} $file: ${samples.size / 16} ms audio decoded in $elapsed -> \"$text\"")
            assertTrue(wordEdits(transcript, text) <= allowedEdits, "$file: expected <$transcript>, got <$text>")
        }
    }

    private fun wordEdits(a: String, b: String): Int {
        val x = a.split(" ")
        val y = b.split(" ")
        var prev = IntArray(y.size + 1) { it }
        for (i in 1..x.size) {
            val cur = IntArray(y.size + 1)
            cur[0] = i
            for (j in 1..y.size) {
                cur[j] = minOf(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + if (x[i - 1] == y[j - 1]) 0 else 1)
            }
            prev = cur
        }
        return prev[y.size]
    }
}
