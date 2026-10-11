package coredevices.pebble.services.asr4all

import co.touchlab.kermit.Logger
import coredevices.pebble.services.asr4all.Asr4allModelSpec.ModelFile
import coredevices.util.CoreConfigFlow
import io.ktor.client.HttpClient
import io.ktor.client.request.prepareGet
import io.ktor.client.statement.bodyAsChannel
import io.ktor.http.isSuccess
import io.ktor.utils.io.readAvailable
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.IO
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.io.buffered
import kotlinx.io.files.Path
import kotlinx.io.files.SystemFileSystem
import kotlinx.io.readString
import kotlinx.serialization.json.Json

/** Platform hooks for ASR4ALL. [modelDirectory] is null where there is no runtime. */
interface Asr4allPlatform {
    val modelDirectory: Path?
    fun sha256Hex(path: Path): String
    fun openRuntime(spec: Asr4allModelSpec, encoder: Path): Asr4allRuntime
}

sealed interface Asr4allModelState {
    data object Unsupported : Asr4allModelState
    data object NotDownloaded : Asr4allModelState
    data class Downloading(val progress: Float) : Asr4allModelState
    data object Ready : Asr4allModelState
    data class Failed(val message: String) : Asr4allModelState
}

/** A decoder for one utterance; closing it releases the model. */
class Asr4allSession(
    private val runtime: Asr4allRuntime,
    val decoder: Asr4allStreamDecoder,
) : AutoCloseable {
    override fun close() = runtime.close()
}

class Asr4allModelStore(
    private val platform: Asr4allPlatform,
    private val httpClient: HttpClient,
    private val json: Json,
    private val coreConfigFlow: CoreConfigFlow,
) {
    companion object {
        private val logger = Logger.withTag("Asr4allModelStore")
    }

    /** The model selected in settings. */
    val spec: Asr4allModelSpec
        get() = Asr4allModelSpec.forId(coreConfigFlow.value.sttConfig.asr4allModel)

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val _state = MutableStateFlow<Asr4allModelState>(Asr4allModelState.NotDownloaded)
    val state: StateFlow<Asr4allModelState> = _state.asStateFlow()
    private var downloadJob: Job? = null
    /** The model [state] reports a download (or failed download) for. */
    private var downloadSpec: Asr4allModelSpec? = null
    private val metadataLock = Mutex()
    private var metadata: Pair<Asr4allModelSpec, Asr4allMetadata>? = null

    private fun Asr4allModelSpec.directory(): Path? =
        platform.modelDirectory?.let { Path(it, repo.substringAfter('/'), revision) }

    private fun ModelFile.localPath(dir: Path) = Path(dir, path.substringAfterLast('/'))

    private fun Asr4allModelSpec.isInstalled(dir: Path) = listOf(encoder, metadata).all {
        SystemFileSystem.metadataOrNull(it.localPath(dir))?.size == it.size
    }

    fun refresh(): Asr4allModelState {
        val spec = spec
        val dir = spec.directory()
        val current = _state.value
        val downloading = current is Asr4allModelState.Downloading || current is Asr4allModelState.Failed
        if (!downloading || downloadSpec != spec) {
            _state.value = when {
                dir == null -> Asr4allModelState.Unsupported
                spec.isInstalled(dir) -> Asr4allModelState.Ready
                else -> Asr4allModelState.NotDownloaded
            }
        }
        return _state.value
    }

    /** Downloads the selected model and then removes the others. */
    fun download() {
        val spec = spec
        val dir = spec.directory() ?: return
        if (downloadJob?.isActive == true && downloadSpec == spec) return
        val previous = downloadJob
        downloadJob = scope.launch {
            previous?.cancelAndJoin()
            downloadSpec = spec
            _state.value = Asr4allModelState.Downloading(0f)
            try {
                SystemFileSystem.createDirectories(dir)
                val files = listOf(spec.metadata, spec.encoder)
                val total = files.sumOf { it.size }.toFloat()
                var done = 0L
                for (file in files) {
                    val target = file.localPath(dir)
                    if (SystemFileSystem.metadataOrNull(target)?.size == file.size) {
                        done += file.size
                        continue
                    }
                    downloadFile(spec, file, target) { bytes ->
                        val progress = ((done + bytes) / total * 100).toInt() / 100f
                        _state.value = Asr4allModelState.Downloading(progress)
                    }
                    done += file.size
                }
                deleteModelsOtherThan(spec)
                _state.value = Asr4allModelState.Ready
            } catch (e: CancellationException) {
                _state.value = Asr4allModelState.NotDownloaded
                throw e
            } catch (e: Exception) {
                logger.e(e) { "ASR4ALL model download failed" }
                _state.value = Asr4allModelState.Failed(e.message ?: "Download failed")
            }
        }
    }

    private suspend fun downloadFile(
        spec: Asr4allModelSpec,
        file: ModelFile,
        target: Path,
        onProgress: (Long) -> Unit,
    ) {
        val part = Path("$target.part")
        val url = spec.url(file)
        logger.i { "Downloading $url" }
        httpClient.prepareGet(url).execute { response ->
            check(response.status.isSuccess()) { "HTTP ${response.status.value} for ${file.path}" }
            val channel = response.bodyAsChannel()
            val buffer = ByteArray(64 * 1024)
            var written = 0L
            SystemFileSystem.sink(part).buffered().use { sink ->
                while (true) {
                    val n = channel.readAvailable(buffer)
                    if (n == -1) break
                    sink.write(buffer, 0, n)
                    written += n
                    onProgress(written)
                }
            }
        }
        val size = SystemFileSystem.metadataOrNull(part)?.size
        val hash = platform.sha256Hex(part)
        if (size != file.size || !hash.equals(file.sha256, ignoreCase = true)) {
            SystemFileSystem.delete(part, mustExist = false)
            error("${file.path} failed verification (size=$size, sha256=$hash)")
        }
        SystemFileSystem.atomicMove(part, target)
    }

    private fun deleteModelsOtherThan(keep: Asr4allModelSpec) {
        val root = platform.modelDirectory ?: return
        for (repo in SystemFileSystem.list(root)) {
            for (revision in SystemFileSystem.list(repo)) {
                if (repo.name == keep.repo.substringAfter('/') && revision.name == keep.revision) continue
                SystemFileSystem.list(revision).forEach { SystemFileSystem.delete(it) }
                SystemFileSystem.delete(revision)
                logger.i { "Deleted unused model $revision" }
            }
            if (SystemFileSystem.list(repo).isEmpty()) SystemFileSystem.delete(repo)
        }
    }

    fun isReady(): Boolean = refresh() == Asr4allModelState.Ready

    suspend fun openSession(): Asr4allSession = withContext(Dispatchers.IO) {
        val spec = spec
        val dir = checkNotNull(spec.directory()) { "ASR4ALL is not supported on this platform" }
        check(spec.isInstalled(dir)) { "ASR4ALL model is not downloaded" }
        val meta = metadataLock.withLock {
            metadata?.takeIf { it.first == spec }?.second ?: json.decodeFromString<Asr4allMetadata>(
                SystemFileSystem.source(spec.metadata.localPath(dir)).buffered().use { it.readString() }
            ).also { metadata = spec to it }
        }
        val runtime = platform.openRuntime(spec, spec.encoder.localPath(dir))
        try {
            Asr4allSession(runtime, Asr4allStreamDecoder(runtime, meta))
        } catch (e: Exception) {
            runtime.close()
            throw e
        }
    }
}
