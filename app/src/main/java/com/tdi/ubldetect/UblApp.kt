package com.tdi.ubldetect

import android.app.Activity
import android.app.Application
import android.content.ComponentCallbacks2
import android.util.Log
import com.tdi.ubldetect.infer.DelegateKind
import com.tdi.ubldetect.infer.ModelPool
import com.tdi.ubldetect.infer.TfliteRunner
import java.util.concurrent.Executors

/**
 * Holds the loaded models for as long as the app is in front of someone.
 *
 * The models outlive the camera screen on purpose. Which one to run depends
 * on the mode and the resolution, never on the category, so returning to the
 * picker and choosing another category used to close a 57 MB interpreter and
 * open the same file again, for a wait of about 3.5 s, twice that for QPDS.
 *
 * They do not outlive the app being put away. [onTrimMemory] releases them
 * when the app is no longer visible, so a phone in somebody's pocket is not
 * holding 143 MiB of shelf detector.
 */
class UblApp : Application() {

    /** Loads off the main thread, one at a time, so warm ups cannot pile up. */
    private val warmUpThread = Executors.newSingleThreadExecutor()

    val models = ModelPool<TfliteRunner> { asset, delegate ->
        TfliteRunner(this, asset, delegate)
    }

    /**
     * Starts loading [assets] before anything asks for them.
     *
     * Called when a list of categories appears: reading it and choosing takes
     * a few seconds, which is about what a model load costs, so the wait
     * happens while the person is deciding rather than after they tap.
     */
    fun warmUp(assets: List<String>, delegate: DelegateKind) {
        warmUpThread.execute {
            val began = android.os.SystemClock.elapsedRealtime()
            models.warm(assets, delegate)
            Log.i(
                TAG,
                "warmed ${assets.size} in ${android.os.SystemClock.elapsedRealtime() - began}ms",
            )
        }
    }

    override fun onTrimMemory(level: Int) {
        super.onTrimMemory(level)
        // UI_HIDDEN means the app went to the background; the levels above it
        // mean the system wants memory back. Either way these are no longer
        // earning their 143 MiB, and the pool refuses while a camera is
        // actually holding them.
        if (level >= ComponentCallbacks2.TRIM_MEMORY_UI_HIDDEN) {
            if (models.releaseIfIdle()) Log.i(TAG, "released models at trim level $level")
        }
    }

    private companion object {
        const val TAG = "UblApp"
    }
}

/** The pool, from anywhere with an Activity. */
val Activity.models: ModelPool<TfliteRunner> get() = (application as UblApp).models
