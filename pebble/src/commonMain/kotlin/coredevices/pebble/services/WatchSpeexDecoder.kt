package coredevices.pebble.services

import co.touchlab.kermit.Logger
import coredevices.speex.SpeexCodec
import coredevices.speex.SpeexDecodeResult
import io.rebble.libpebblecommon.voice.VoiceEncoderInfo

/**
 * Decodes the Speex frames a watch streams for dictation. A frame that fails to decode (e.g.
 * damaged over Bluetooth) becomes a frame of silence, so it costs 20 ms of audio instead of the
 * whole dictation.
 */
internal class WatchSpeexDecoder(
    frameSize: Int,
    private val decodeFrame: (frame: ByteArray, pcm: ByteArray) -> SpeexDecodeResult,
    private val onClose: () -> Unit = {},
) : AutoCloseable {
    companion object {
        private val logger = Logger.withTag("WatchSpeexDecoder")

        fun create(encoderInfo: VoiceEncoderInfo.Speex): WatchSpeexDecoder {
            val codec = SpeexCodec(
                sampleRate = encoderInfo.sampleRate,
                bitRate = encoderInfo.bitRate,
                frameSize = encoderInfo.frameSize,
            )
            return WatchSpeexDecoder(
                frameSize = encoderInfo.frameSize,
                decodeFrame = { frame, pcm -> codec.decodeFrame(frame, pcm, hasHeaderByte = true) },
                onClose = codec::close,
            )
        }
    }

    private val pcm = ByteArray(frameSize * Short.SIZE_BYTES)
    private var frames = 0
    private var badFrames = 0

    /** 16-bit little-endian PCM for [frame]. The returned buffer is reused by the next call. */
    fun decode(frame: ByteArray): ByteArray {
        frames++
        val result = decodeFrame(frame, pcm)
        if (result != SpeexDecodeResult.Success) {
            badFrames++
            pcm.fill(0)
        }
        return pcm
    }

    override fun close() {
        onClose()
        if (badFrames > 0) {
            logger.w { "Replaced $badFrames of $frames Speex frames with silence" }
        }
    }
}
