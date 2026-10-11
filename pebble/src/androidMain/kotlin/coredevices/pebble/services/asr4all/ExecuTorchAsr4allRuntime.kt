package coredevices.pebble.services.asr4all

import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor

class ExecuTorchAsr4allRuntime(
    override val spec: Asr4allModelSpec,
    encoderPath: String,
) : Asr4allRuntime {
    private val module = Module.load(encoderPath, Module.LOAD_MODE_MMAP).also { module ->
        listOf(spec.streamMethod, spec.pcecMethod, spec.pcecFlushMethod).forEach(module::loadMethod)
    }

    // Outputs are views into the method's arena, which the next call overwrites, so caches are
    // copied out into buffers the state owns.
    private class FloatCache(shape: LongArray) {
        private val buffer = Tensor.allocateFloatBuffer(Tensor.numel(shape).toInt())
        val input: EValue = EValue.from(Tensor.fromBlob(buffer, shape))

        fun update(output: EValue) {
            buffer.rewind()
            output.toTensor().copyDataInto(buffer)
        }
    }

    private class LongCache(shape: LongArray) {
        private val buffer = Tensor.allocateLongBuffer(Tensor.numel(shape).toInt())
        val input: EValue = EValue.from(Tensor.fromBlob(buffer, shape))

        fun update(output: EValue) {
            buffer.rewind()
            output.toTensor().copyDataInto(buffer)
        }
    }

    private class StreamState(val filter: FloatCache, val attn: FloatCache) : Asr4allState
    private class PcecState(val cache: FloatCache, val seen: LongCache) : Asr4allState

    override fun newStreamState(): Asr4allState = StreamState(
        filter = FloatCache(longArrayOf(spec.filterCacheSize.toLong())),
        attn = FloatCache(longArrayOf(spec.attnCacheSize.toLong())),
    )

    override fun newPcecState(): Asr4allState = PcecState(
        cache = FloatCache(longArrayOf(1, spec.pcecCacheSize.toLong())),
        seen = LongCache(longArrayOf(1)),
    )

    override fun stream(window: FloatArray, state: Asr4allState): Asr4allRuntime.StreamOutput {
        state as StreamState
        val out = module.execute(
            spec.streamMethod,
            EValue.from(Tensor.fromBlob(window, longArrayOf(1, spec.windowSamples.toLong()))),
            state.filter.input,
            state.attn.input,
        )
        state.filter.update(out[1])
        state.attn.update(out[2])
        return Asr4allRuntime.StreamOutput(
            logits = out[0].toTensor().dataAsFloatArray,
            hidden = out[3].toTensor().dataAsFloatArray,
        )
    }

    override fun pcec(block: PcecBlock, state: Asr4allState): LongArray {
        state as PcecState
        val out = module.execute(
            spec.pcecMethod,
            *block.inputs(),
            state.cache.input,
            state.seen.input,
        )
        state.cache.update(out[1])
        state.seen.update(out[2])
        return out[0].toTensor().dataAsLongArray
    }

    override fun pcecFlush(block: PcecBlock, nReal: Int, state: Asr4allState): LongArray {
        state as PcecState
        val out = module.execute(
            spec.pcecFlushMethod,
            *block.inputs(),
            EValue.from(Tensor.fromBlob(longArrayOf(nReal.toLong()), longArrayOf())),
            state.cache.input,
            state.seen.input,
        )
        return out[0].toTensor().dataAsLongArray
    }

    private fun PcecBlock.inputs(): Array<EValue> {
        val k = spec.pcecCommit.toLong()
        return arrayOf(
            EValue.from(Tensor.fromBlob(ids, longArrayOf(1, k))),
            EValue.from(Tensor.fromBlob(pause, longArrayOf(1, k))),
            EValue.from(Tensor.fromBlob(acoustic, longArrayOf(1, k, spec.hiddenDim.toLong()))),
            EValue.from(Tensor.fromBlob(keep, longArrayOf(1, k))),
        )
    }

    override fun close() {
        module.destroy()
    }
}
