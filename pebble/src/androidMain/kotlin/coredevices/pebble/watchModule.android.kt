package coredevices.pebble

import android.content.Context
import coredevices.pebble.services.asr4all.AndroidAsr4allPlatform
import coredevices.pebble.services.asr4all.Asr4allPlatform
import io.rebble.libpebblecommon.connection.AppContext
import org.koin.core.module.Module
import org.koin.core.module.dsl.singleOf
import org.koin.dsl.bind
import org.koin.dsl.module

actual val platformWatchModule: Module = module {
    single {
        val context = get<Context>()
        AppContext(context.applicationContext)
    }
    single<Platform> { Platform.Android }
    singleOf(::PebbleAndroidDelegate)
    singleOf(::AndroidAsr4allPlatform) bind Asr4allPlatform::class
}