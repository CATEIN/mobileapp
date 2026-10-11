package io.rebble.libpebblecommon.connection.endpointmanager.audio

import io.rebble.libpebblecommon.connection.endpointmanager.audio.VoiceSessionManager.Companion.partialResultTail
import io.rebble.libpebblecommon.packets.DictationPartialResult
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

class VoicePartialResultTest {
    @Test
    fun shortTextIsSentWhole() {
        assertEquals("Hello world", partialResultTail("Hello world", maxBytes = 11))
    }

    @Test
    fun longTextKeepsWholeTrailingWords() {
        assertEquals("three four", partialResultTail("one two three four", maxBytes = 12))
    }

    @Test
    fun cutNeverSplitsAMultiByteCharacter() {
        // No space in range, so the cut lands on a character boundary instead.
        assertEquals("éé", partialResultTail("ééé", maxBytes = 5))
    }

    @OptIn(ExperimentalUnsignedTypes::class)
    @Test
    fun packetLayoutMatchesFirmware() {
        val bytes = DictationPartialResult(0x1234u, "Hi").serialize()
        // header (length, endpoint 11000), msg id, flags, session id, text without terminator
        assertContentEquals(
            ubyteArrayOf(0x00u, 0x09u, 0x2Au, 0xF8u, 0x04u, 0u, 0u, 0u, 0u, 0x34u, 0x12u, 'H'.code.toUByte(), 'i'.code.toUByte()),
            bytes,
        )
    }
}
