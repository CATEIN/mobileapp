package coredevices.pebble.services.asr4all

import coredevices.pebble.services.AudioLevel
import coredevices.pebble.services.pcm16ToFloat
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

private object FakeState : Asr4allState

class Asr4allStreamDecoderTest {
    private val ctrl = Asr4allMetadata.Ctrl(cap = 10, allcaps = 11, period = 12, comma = 13, question = 14)
    private val pcecMeta = Asr4allMetadata.Pcec(
        vocab = listOf(" hello", " world", "s", " how"),
        blankId = 15,
        ctrl = ctrl,
        padIn = 7,
        pauseBlackout = 9,
        realPause = true,
    )
    private val meta = Asr4allMetadata(
        sampleRate = 16_000,
        samplesPerFrame = 10,
        vocab = listOf("a", "b", " c"),
        blankId = 3,
        pcec = pcecMeta,
    )

    // 5-frame windows committing 2 frames; PCEC blocks of 2 tokens with a 1-token delay line.
    private val spec = Asr4allModelSpec.Small.copy(
        windowSamples = 50,
        commitFrames = 2,
        logitsSize = 4,
        hiddenDim = 2,
        pcecCommit = 2,
        pcecFlushSlots = 7,
    )

    /** Emits [frameIds] in order (blank once exhausted); hidden row for frame f is [f, 2f]. */
    private inner class FakeRuntime(private val frameIds: List<Int>) : Asr4allRuntime {
        override val spec = this@Asr4allStreamDecoderTest.spec
        val windows = mutableListOf<FloatArray>()
        val pcecBlocks = mutableListOf<PcecBlock>()
        var pcecResults: (call: Int) -> LongArray = { LongArray(4) { pcecMeta.blankId.toLong() } }
        var flushResult: LongArray = LongArray(7) { pcecMeta.blankId.toLong() }
        val flushBlocks = mutableListOf<Pair<PcecBlock, Int>>()
        private var frame = 0

        override fun newStreamState(): Asr4allState = FakeState
        override fun newPcecState(): Asr4allState = FakeState

        override fun stream(window: FloatArray, state: Asr4allState): Asr4allRuntime.StreamOutput {
            windows += window.copyOf()
            val logits = FloatArray(spec.commitFrames * spec.logitsSize)
            val hidden = FloatArray(spec.commitFrames * spec.hiddenDim)
            for (t in 0 until spec.commitFrames) {
                val id = frameIds.getOrElse(frame) { meta.blankId }
                logits[t * spec.logitsSize + id] = 1f
                hidden[t * spec.hiddenDim] = frame.toFloat()
                hidden[t * spec.hiddenDim + 1] = 2f * frame
                frame++
            }
            return Asr4allRuntime.StreamOutput(logits, hidden)
        }

        override fun pcec(block: PcecBlock, state: Asr4allState): LongArray {
            pcecBlocks += block
            return pcecResults(pcecBlocks.size - 1)
        }

        override fun pcecFlush(block: PcecBlock, nReal: Int, state: Asr4allState): LongArray {
            flushBlocks += block to nReal
            return flushResult
        }

        override fun close() {}
    }

    private fun samples(count: Int, from: Int = 0) = FloatArray(count) { (from + it).toFloat() }

    @Test
    fun windowsAdvanceByCommitAndTailIsZeroPadded() {
        val runtime = FakeRuntime(emptyList())
        val decoder = Asr4allStreamDecoder(runtime, meta)
        assertFalse(decoder.accept(samples(33)))
        assertTrue(decoder.accept(samples(42, from = 33)))
        assertEquals(2, runtime.windows.size)
        decoder.finish()
        assertEquals(4, runtime.windows.size)
        assertEquals(listOf(0f, 20f, 40f, 60f), runtime.windows.map { it[0] })
        assertEquals(74f, runtime.windows[3][14])
        assertEquals(0f, runtime.windows[3][15])
    }

    @Test
    fun acousticTextCollapsesRepeatsAndBlanks() {
        val decoder = Asr4allStreamDecoder(FakeRuntime(listOf(0, 0, 3, 0, 1, 1, 2)), meta)
        decoder.accept(samples(70))
        decoder.finish()
        assertEquals("aab c", decoder.acousticText())
    }

    @Test
    fun tokensPoolFramesAndCarryLeadingPause() {
        // Tokens: 0 (frames 0-1), 1 after 3 blanks (frames 5-6), 2 (frame 7, held open until flush).
        val runtime = FakeRuntime(listOf(0, 0, 3, 3, 3, 1, 1, 2))
        val decoder = Asr4allStreamDecoder(runtime, meta)
        decoder.accept(samples(80))
        decoder.finish()
        assertEquals(1, runtime.pcecBlocks.size)
        val block = runtime.pcecBlocks[0]
        assertContentEquals(longArrayOf(0, 1), block.ids)
        assertContentEquals(longArrayOf(0, 2), block.pause)
        // Token 0 averages frames 0..4 (repeat + trailing blanks), token 1 frames 5..6.
        assertContentEquals(floatArrayOf(2f, 4f, 5.5f, 11f), block.acoustic)

        decoder.text()
        val (flushBlock, nReal) = runtime.flushBlocks.last()
        assertEquals(1, nReal)
        assertContentEquals(longArrayOf(2, 7), flushBlock.ids)
        assertContentEquals(longArrayOf(0, 9), flushBlock.pause)
    }

    @Test
    fun pcecWarmUpSlotsAreSkipped() {
        val runtime = FakeRuntime(listOf(0, 1, 0, 1, 0))
        runtime.pcecResults = { call ->
            if (call == 0) longArrayOf(1, 1, 0, 13) else longArrayOf(1, 15, 12, 15)
        }
        runtime.flushResult = longArrayOf(15, 15, 3, 14, 15, 15, 15)
        val decoder = Asr4allStreamDecoder(runtime, meta)
        decoder.accept(samples(50))
        decoder.finish()
        // Call 0 drops the 2 warm-up slots of the 1-token delay line; call 1 and the flush don't.
        assertEquals("Hello, world. How?", decoder.text())
    }

    @Test
    fun renderAppliesCasingAndPunctuation() {
        val slots = listOf(0, 13, 1, 2, 12, 15, 11, 3, 3, 14)
        assertEquals("Hello, worlds. HOW?", Asr4allStreamDecoder.renderSlots(pcecMeta, slots))
    }

    @Test
    fun pauseBuckets() {
        assertEquals(listOf(0, 1, 2, 2, 7, 8), listOf(0, 1, 2, 3, 64, 200).map(Asr4allStreamDecoder::pauseBucket))
    }

    @Test
    fun pcm16LittleEndianToFloat() {
        val pcm = byteArrayOf(0x00, 0x80.toByte(), 0xFF.toByte(), 0x7F, 0x00, 0x40)
        assertContentEquals(floatArrayOf(-1f, 32767f / 32768f, 0.5f), pcm16ToFloat(pcm))
    }

    @Test
    fun audioLevelReportsDurationRmsAndPeak() {
        val level = AudioLevel()
        level.add(FloatArray(8_000) { if (it % 2 == 0) 0.5f else -0.5f })
        level.add(FloatArray(8_000))
        assertEquals("1000 ms, rms -9 dBFS, peak -6 dBFS", level.toString())
        assertEquals("0 ms, rms -100 dBFS, peak -100 dBFS", AudioLevel().toString())
    }

    @Test
    fun modelSpecsResolveByIdWithSmallAsDefault() {
        assertEquals(Asr4allModelSpec.Medium, Asr4allModelSpec.forId("m"))
        assertEquals(Asr4allModelSpec.Large, Asr4allModelSpec.forId("l"))
        assertEquals(Asr4allModelSpec.Small, Asr4allModelSpec.forId(null))
        assertEquals(Asr4allModelSpec.Small, Asr4allModelSpec.forId("unknown"))
        assertEquals(16, Asr4allModelSpec.Medium.pcecLookahead)
    }
}
