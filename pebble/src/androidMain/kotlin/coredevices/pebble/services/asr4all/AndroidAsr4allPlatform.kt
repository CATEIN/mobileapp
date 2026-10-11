package coredevices.pebble.services.asr4all

import android.os.Process
import io.rebble.libpebblecommon.connection.AppContext
import kotlinx.io.files.Path
import java.io.File
import java.security.MessageDigest

class AndroidAsr4allPlatform(appContext: AppContext) : Asr4allPlatform {
    // ExecuTorch only ships 64-bit native libraries.
    override val modelDirectory = if (Process.is64Bit()) {
        Path(appContext.context.noBackupFilesDir.resolve("asr4all").path)
    } else {
        null
    }

    override fun sha256Hex(path: Path): String {
        val digest = MessageDigest.getInstance("SHA-256")
        File(path.toString()).inputStream().use { input ->
            val buffer = ByteArray(64 * 1024)
            while (true) {
                val n = input.read(buffer)
                if (n == -1) break
                digest.update(buffer, 0, n)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    override fun openRuntime(spec: Asr4allModelSpec, encoder: Path): Asr4allRuntime =
        ExecuTorchAsr4allRuntime(spec, encoder.toString())
}
