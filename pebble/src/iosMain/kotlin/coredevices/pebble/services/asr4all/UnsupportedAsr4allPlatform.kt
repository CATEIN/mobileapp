package coredevices.pebble.services.asr4all

import kotlinx.io.files.Path

object UnsupportedAsr4allPlatform : Asr4allPlatform {
    override val modelDirectory: Path? = null
    override fun sha256Hex(path: Path): String = error("ASR4ALL is not supported on iOS")
    override fun openRuntime(spec: Asr4allModelSpec, encoder: Path): Asr4allRuntime =
        error("ASR4ALL is not supported on iOS")
}
