package coredevices.pebble.services.asr4all

/**
 * Streaming decode of one utterance: acoustic CTC on fixed windows, then the PCEC pass
 * (punctuation, capitalisation, error correction) over the committed tokens.
 *
 * Port of `Engine` / `_Pcec` from FUTO's `executorch/asr4all_executorch.py`. Not thread-safe.
 */
class Asr4allStreamDecoder(
    private val runtime: Asr4allRuntime,
    private val meta: Asr4allMetadata,
) {
    private val spec = runtime.spec
    private val spf = meta.samplesPerFrame
    private val windowFrames = spec.windowSamples / spf

    private val streamState = runtime.newStreamState()
    private val pcec = Pcec(runtime, meta.pcec, spec)

    /** Audio from [pcmOffset] (in samples) onwards; earlier samples are already committed. */
    private var pcm = FloatArray(0)
    private var pcmOffset = 0
    private var totalSamples = 0
    private var nextFrame = 0
    private val frameIds = ArrayList<Int>()

    /**
     * Appends 16 kHz mono samples in [-1, 1] and decodes every complete window.
     * @return whether any new audio was decoded
     */
    fun accept(samples: FloatArray): Boolean {
        append(samples)
        var decoded = false
        while ((nextFrame + windowFrames) * spf <= totalSamples) {
            step()
            decoded = true
        }
        return decoded
    }

    /** Decodes the tail, zero-padding the missing lookahead. Call once, after the last [accept]. */
    fun finish() {
        while (nextFrame * spf < totalSamples) {
            step()
        }
        pcec.flush()
        pcec.drain()
    }

    /** Punctuated, cased transcript so far; after [finish] this is the final result. */
    fun text(): String = renderSlots(meta.pcec, pcec.slots + pcec.tail())

    /** Lower-case acoustic transcript, ahead of [text] by the PCEC lag. */
    fun acousticText(): String {
        val out = StringBuilder()
        var prev = -1
        for (id in frameIds) {
            if (id != prev && id != meta.blankId && id < meta.vocab.size) out.append(meta.vocab[id])
            prev = id
        }
        return out.split(WHITESPACE).filter { it.isNotEmpty() }.joinToString(" ")
    }

    private fun append(samples: FloatArray) {
        val keepFrom = nextFrame * spf - pcmOffset
        val retained = pcm.size - keepFrom
        val next = FloatArray(retained + samples.size)
        pcm.copyInto(next, 0, keepFrom, pcm.size)
        samples.copyInto(next, retained)
        pcm = next
        pcmOffset += keepFrom
        totalSamples += samples.size
    }

    private fun step() {
        val start = nextFrame * spf - pcmOffset
        val window = FloatArray(spec.windowSamples)
        val end = minOf(pcm.size, start + spec.windowSamples)
        if (end > start) pcm.copyInto(window, 0, start, end)
        val out = runtime.stream(window, streamState)

        val valid = minOf(spec.commitFrames, totalSamples / spf - nextFrame)
        val ids = IntArray(maxOf(valid, 0)) { t -> argmax(out.logits, t * spec.logitsSize, spec.logitsSize) }
        frameIds.addAll(ids.asList())
        if (ids.isNotEmpty()) {
            pcec.pool(ids, out.hidden, meta.blankId)
            pcec.drain()
        }
        nextFrame += spec.commitFrames
    }

    private class Pcec(
        private val runtime: Asr4allRuntime,
        private val meta: Asr4allMetadata.Pcec,
        spec: Asr4allModelSpec,
    ) {
        private val k = spec.pcecCommit
        private val lookahead = spec.pcecLookahead
        private val dim = spec.hiddenDim
        private val state = runtime.newPcecState()

        val slots = ArrayList<Int>()
        private val tokens = ArrayList<Token>()
        private var consumed = 0
        private var open: OpenToken? = null
        private var blanks = 0
        private var prev = -1

        private class Token(val id: Int, val acoustic: FloatArray, val pause: Int)

        private class OpenToken(val id: Int, val acc: DoubleArray, val leadingBlanks: Int) {
            var frames = 1
        }

        fun pool(frameIds: IntArray, hidden: FloatArray, blank: Int) {
            for ((t, fid) in frameIds.withIndex()) {
                val row = t * dim
                if (fid != prev && fid != blank) {
                    flush()
                    open = OpenToken(fid, DoubleArray(dim) { hidden[row + it].toDouble() }, blanks)
                    blanks = 0
                } else {
                    open?.let { o ->
                        for (i in 0 until dim) o.acc[i] += hidden[row + i]
                        o.frames++
                    }
                }
                prev = fid
                if (fid == blank) blanks++
            }
        }

        fun flush() {
            val o = open ?: return
            open = null
            val acoustic = FloatArray(dim) { (o.acc[it] / o.frames).toFloat() }
            tokens.add(Token(o.id, acoustic, pauseBucket(o.leadingBlanks)))
        }

        fun drain() {
            while (tokens.size >= k) {
                val out = runtime.pcec(block(k), state)
                val skip = minOf(2 * k, maxOf(0, 2 * lookahead - 2 * consumed))
                for (i in skip until 2 * k) slots.add(out[i].toInt())
                tokens.subList(0, k).clear()
                consumed += k
            }
        }

        fun tail(): List<Int> {
            val n = tokens.size
            val out = runtime.pcecFlush(block(n), n, state)
            val valid = minOf(out.size, 2 * lookahead + 2 * n + 1)
            val skip = maxOf(0, 2 * lookahead - 2 * consumed)
            return (skip until valid).map { out[it].toInt() }
        }

        private fun block(n: Int): PcecBlock {
            val ids = LongArray(k) { meta.padIn.toLong() }
            val pause = LongArray(k) { meta.pauseBlackout.toLong() }
            val acoustic = FloatArray(k * dim)
            for (i in 0 until n) {
                val token = tokens[i]
                ids[i] = token.id.toLong()
                if (meta.realPause) pause[i] = token.pause.toLong()
                token.acoustic.copyInto(acoustic, i * dim)
            }
            return PcecBlock(ids, pause, acoustic, keep = FloatArray(k))
        }
    }

    companion object {
        private val WHITESPACE = Regex("\\s+")

        /** Blank-frame counts separating the pause buckets PCEC was trained on. */
        private val PAUSE_THRESHOLDS = intArrayOf(1, 2, 4, 8, 16, 32, 64, 128)

        internal fun pauseBucket(blanks: Int): Int = PAUSE_THRESHOLDS.count { blanks >= it }

        private fun argmax(values: FloatArray, offset: Int, length: Int): Int {
            var best = 0
            for (i in 1 until length) {
                if (values[offset + i] > values[offset + best]) best = i
            }
            return best
        }

        /** PCEC lattice slots -> punctuated, cased text. */
        internal fun renderSlots(meta: Asr4allMetadata.Pcec, slots: List<Int>): String {
            class Word(val pieces: StringBuilder, val case: Int?, var punct: String = "")

            val ctrl = meta.ctrl
            val punctuation = mapOf(ctrl.period to ".", ctrl.comma to ",", ctrl.question to "?")
            val words = ArrayList<Word>()
            var pendingCase: Int? = null
            var prev = -1
            for (id in slots) {
                if (id == prev || id == meta.blankId) {
                    prev = id
                    continue
                }
                prev = id
                when {
                    id == ctrl.cap || id == ctrl.allcaps -> pendingCase = id
                    id in punctuation -> words.lastOrNull()?.punct = punctuation.getValue(id)
                    id < meta.vocab.size && meta.vocab[id].isNotEmpty() -> {
                        val piece = meta.vocab[id]
                        if (piece.startsWith(" ") || words.isEmpty()) {
                            words.add(Word(StringBuilder(piece), pendingCase))
                            pendingCase = null
                        } else {
                            words.last().pieces.append(piece)
                        }
                    }
                }
            }
            val out = ArrayList<String>()
            var sentenceStart = true
            for (word in words) {
                var w = word.pieces.toString().trim()
                if (w.isEmpty()) continue
                w = when {
                    word.case == ctrl.allcaps -> w.uppercase()
                    word.case == ctrl.cap || sentenceStart -> w.replaceFirstChar { it.uppercaseChar() }
                    else -> w
                }
                sentenceStart = word.punct == "." || word.punct == "?"
                out.add(w + word.punct)
            }
            return out.joinToString(" ")
        }
    }
}
