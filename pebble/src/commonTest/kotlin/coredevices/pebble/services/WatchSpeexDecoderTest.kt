package coredevices.pebble.services

import coredevices.speex.SpeexDecodeResult
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertTrue

class WatchSpeexDecoderTest {
    private fun decoder(results: List<SpeexDecodeResult>): WatchSpeexDecoder {
        val queue = results.toMutableList()
        return WatchSpeexDecoder(frameSize = 2, decodeFrame = { _, pcm ->
            // The codec leaves junk in the buffer when a frame is damaged
            pcm.fill(7)
            queue.removeFirst()
        })
    }

    @Test
    fun goodFramesPassThrough() {
        decoder(listOf(SpeexDecodeResult.Success)).use {
            assertContentEquals(byteArrayOf(7, 7, 7, 7), it.decode(byteArrayOf(1)))
        }
    }

    @Test
    fun badFramesBecomeSilenceAndDecodingContinues() {
        decoder(
            listOf(SpeexDecodeResult.CorruptStream, SpeexDecodeResult.EndOfStream, SpeexDecodeResult.Success)
        ).use {
            assertContentEquals(ByteArray(4), it.decode(byteArrayOf(1)))
            assertContentEquals(ByteArray(4), it.decode(byteArrayOf(2)))
            assertContentEquals(byteArrayOf(7, 7, 7, 7), it.decode(byteArrayOf(3)))
        }
    }

    @Test
    fun closeReleasesTheCodec() {
        var closed = false
        WatchSpeexDecoder(frameSize = 1, decodeFrame = { _, _ -> SpeexDecodeResult.Success }, onClose = { closed = true })
            .close()
        assertTrue(closed)
    }
}
