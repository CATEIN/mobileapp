package coredevices.pebble.services.asr4all

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * A pinned ASR4ALL ExecuTorch bundle (https://huggingface.co/futo-org).
 *
 * The ExecuTorch Android bindings don't expose tensor shapes, so the geometry the decoder needs is
 * pinned here alongside the files it was read from (via the Python runtime's `MethodMeta`).
 */
data class Asr4allModelSpec(
    /** Stored in [coredevices.util.STTConfig.asr4allModel]. */
    val id: String,
    val repo: String,
    /** Hugging Face commit the files and hashes below were taken from. */
    val revision: String,
    val encoder: ModelFile,
    val metadata: ModelFile,
    val streamMethod: String,
    val pcecMethod: String,
    val pcecFlushMethod: String,
    /** PCM samples per `stream` call (commit + lookahead frames). */
    val windowSamples: Int,
    /** Frames committed per `stream` call. */
    val commitFrames: Int,
    val logitsSize: Int,
    val hiddenDim: Int,
    val filterCacheSize: Int,
    val attnCacheSize: Int,
    /** Tokens per PCEC call. */
    val pcecCommit: Int,
    val pcecCacheSize: Int,
    /** Width of the PCEC flush output, which carries the model's settle lag. */
    val pcecFlushSlots: Int,
) {
    data class ModelFile(val path: String, val sha256: String, val size: Long)

    fun url(file: ModelFile) = "https://huggingface.co/$repo/resolve/$revision/${file.path}"

    /** Tokens the PCEC delay line holds back (`(flush_slots - 1) / 2 - commit`). */
    val pcecLookahead: Int get() = (pcecFlushSlots - 1) / 2 - pcecCommit

    companion object {
        /** asr4all-s, INT8 XNNPACK, lowest-latency tier (640 ms commits). */
        val Small = Asr4allModelSpec(
            id = "s",
            repo = "futo-org/asr4all-s",
            revision = "735bdd8ae688094926068990ccf9b26a80cc43c3",
            encoder = ModelFile(
                path = "executorch/xnnpack_int8/asr_encoder.pte",
                sha256 = "a027754d46abdc7ffcdcc621a1430da7896e4504484657228a22d6eccf283381",
                size = 38_489_024,
            ),
            metadata = ModelFile(
                path = "executorch/xnnpack_int8/metadata.json",
                sha256 = "025de70620e2e90fd0e3de947fea4783e0e637ec54d877a5d25c8ad3866e6f48",
                size = 90_364,
            ),
            streamMethod = "stream_c16r4",
            pcecMethod = "pcec_c16",
            pcecFlushMethod = "pcec_c16_flush",
            windowSamples = 22_400,
            commitFrames = 16,
            logitsSize = 257,
            hiddenDim = 384,
            filterCacheSize = 1_006_080,
            attnCacheSize = 329_090,
            pcecCommit = 16,
            pcecCacheSize = 166_976,
            pcecFlushSlots = 65,
        )

        /** asr4all-m, INT8 XNNPACK, lowest-latency tier. Twice the acoustic model of [Small]. */
        val Medium = Asr4allModelSpec(
            id = "m",
            repo = "futo-org/asr4all-m",
            revision = "86f67fdc60fecb4f9662ccd1d99948291db4d905",
            encoder = ModelFile(
                path = "executorch/xnnpack_int8/asr_encoder.pte",
                sha256 = "f50318acb5de7bfc75229f73edcc9276b3553efc9c3e34a7f3f1cf66f76067b4",
                size = 72_395_968,
            ),
            metadata = ModelFile(
                path = "executorch/xnnpack_int8/metadata.json",
                sha256 = "321501d89e6249a76b8c0f896f9c3e90089f0f9d6135ab82802e5752990cab41",
                size = 90_363,
            ),
            streamMethod = "stream_c16r4",
            pcecMethod = "pcec_c16",
            pcecFlushMethod = "pcec_c16_flush",
            windowSamples = 24_960,
            commitFrames = 16,
            logitsSize = 257,
            hiddenDim = 512,
            filterCacheSize = 1_728_000,
            attnCacheSize = 548_354,
            pcecCommit = 16,
            pcecCacheSize = 166_976,
            pcecFlushSlots = 65,
        )

        /** asr4all-l, INT8 XNNPACK, lowest-latency tier. */
        val Large = Asr4allModelSpec(
            id = "l",
            repo = "futo-org/asr4all-l",
            revision = "2da6534551f007201368514ec08b0b9a60ba39dc",
            encoder = ModelFile(
                path = "executorch/xnnpack_int8/asr_encoder.pte",
                sha256 = "50113d5aa74c916cce33828a3ec69c86aa64b58513a437ecf84fecfae098d351",
                size = 123_044_416,
            ),
            metadata = ModelFile(
                path = "executorch/xnnpack_int8/metadata.json",
                sha256 = "05c6fd156f1fa99ea44457ab0d0a8d23b954c272fb614a034476123874e6d125",
                size = 90_362,
            ),
            streamMethod = "stream_c16r4",
            pcecMethod = "pcec_c16",
            pcecFlushMethod = "pcec_c16_flush",
            windowSamples = 27_520,
            commitFrames = 16,
            logitsSize = 257,
            hiddenDim = 640,
            filterCacheSize = 2_545_920,
            attnCacheSize = 822_402,
            pcecCommit = 16,
            pcecCacheSize = 166_976,
            pcecFlushSlots = 65,
        )

        val all = listOf(Small, Medium, Large)

        fun forId(id: String?): Asr4allModelSpec = all.firstOrNull { it.id == id } ?: Small
    }
}

/** The subset of the bundle's `metadata.json` the decoder reads. */
@Serializable
data class Asr4allMetadata(
    @SerialName("sample_rate") val sampleRate: Int,
    @SerialName("samples_per_frame") val samplesPerFrame: Int,
    val vocab: List<String>,
    @SerialName("blank_id") val blankId: Int,
    val pcec: Pcec,
) {
    @Serializable
    data class Pcec(
        val vocab: List<String>,
        @SerialName("blank_id") val blankId: Int,
        val ctrl: Ctrl,
        @SerialName("pad_in") val padIn: Int,
        @SerialName("pause_blackout") val pauseBlackout: Int,
        @SerialName("real_pause") val realPause: Boolean = false,
    )

    @Serializable
    data class Ctrl(
        val cap: Int,
        val allcaps: Int,
        val period: Int,
        val comma: Int,
        val question: Int,
    )
}

/** Opaque per-utterance model state (the streaming caches), owned by an [Asr4allRuntime]. */
interface Asr4allState

/** The three methods of the bundle the decoder calls. Implementations are not thread-safe. */
interface Asr4allRuntime : AutoCloseable {
    val spec: Asr4allModelSpec

    class StreamOutput(
        /** `[commitFrames * logitsSize]` */
        val logits: FloatArray,
        /** `[commitFrames * hiddenDim]` */
        val hidden: FloatArray,
    )

    fun newStreamState(): Asr4allState
    fun newPcecState(): Asr4allState

    /** One `windowSamples` window; advances [state]. */
    fun stream(window: FloatArray, state: Asr4allState): Asr4allRuntime.StreamOutput

    /** One full block of `pcecCommit` tokens; advances [state]. Returns `2 * pcecCommit` slots. */
    fun pcec(block: PcecBlock, state: Asr4allState): LongArray

    /** Renders what is still inside the PCEC delay line; does not advance [state]. */
    fun pcecFlush(block: PcecBlock, nReal: Int, state: Asr4allState): LongArray
}

/** One PCEC input block of `pcecCommit` tokens, padded past the real ones. */
class PcecBlock(
    val ids: LongArray,
    val pause: LongArray,
    /** `[pcecCommit * hiddenDim]` */
    val acoustic: FloatArray,
    val keep: FloatArray,
)
