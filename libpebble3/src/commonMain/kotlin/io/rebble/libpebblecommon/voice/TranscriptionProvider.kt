package io.rebble.libpebblecommon.voice

import kotlinx.coroutines.flow.Flow
import kotlin.time.Duration.Companion.seconds

interface TranscriptionProvider {
    /**
     * @param onPartialResult called with the full text recognized so far, while audio is still
     * arriving, by providers that stream.
     */
    suspend fun transcribe(
        encoderInfo: VoiceEncoderInfo,
        audioFrames: Flow<UByteArray>,
        isNotificationReply: Boolean,
        onPartialResult: suspend (String) -> Unit = {},
    ): TranscriptionResult
    suspend fun canServeSession(): Boolean
}

/**
 * The maximum amount of time the Pebble firmware will wait for a transcription result before timing
 * out and cancelling the session.
 */
val PEBBLE_FW_TRANSCRIPTION_TIMEOUT = 15.seconds
