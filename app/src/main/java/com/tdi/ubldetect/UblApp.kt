package com.tdi.ubldetect

import android.app.Activity
import android.app.Application
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import com.tdi.ubldetect.infer.DelegateKind
import com.tdi.ubldetect.infer.ModelPool
import com.tdi.ubldetect.infer.TfliteRunner
import java.util.concurrent.Executors

/**
 * Holds the loaded models between camera screens, briefly.
 *
 * The models outlive the camera screen on purpose. Which one to run depends
 * on the mode and the resolution, never on the category, so returning to the
 * picker and choosing another category used to close a 57 MB interpreter and
 * open the same file again, for a wait of about 3.5 s, twice that for QPDS.
 * Held, that second category costs about 1 ms.
 *
 * Briefly, because the measured price is not the 143 MiB of model file. An
 * interpreter on the GPU delegate also holds its working buffers, and on this
 * device the QPDS pair came to roughly 760 MB: the picker sat at 1171 MB PSS
 * holding them against 71 MB with nothing loaded. That is fine for the few
 * seconds it takes to choose the next slab and indefensible as a resting
 * state, so they are released once nobody has wanted them for
 * [IDLE_GRACE_MS], and immediately when the app stops being visible.
 */
class UblApp : Application() {

    /** Loads off the main thread, one at a time, so warm ups cannot pile up. */
    private val warmUpThread = Executors.newSingleThreadExecutor()

    private val handler = Handler(Looper.getMainLooper())

    private val releaseWhenIdle = Runnable {
        release("after idle grace")
    }

    /**
     * Closes the models, off the main thread.
     *
     * Every trigger for this arrives on the main thread (a timer, an activity
     * stopping, a memory warning) and releasing takes the pool's lock, which
     * a load holds for seconds. Doing it here would freeze the screen that is
     * being left, so it is handed to the thread that does the loading.
     */
    private fun release(why: String) {
        warmUpThread.execute {
            if (models.releaseIfIdle()) Log.i(TAG, "released models $why")
        }
    }

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

    /**
     * Started activities, to tell "in front of someone" from "put away".
     *
     * Counted here rather than taken from `onTrimMemory`, which was the first
     * attempt and is dead code on this device: `TRIM_MEMORY_UI_HIDDEN` and
     * the levels above it are deprecated and no longer delivered, so pressing
     * home released nothing and the pool kept 143 MiB until the process died.
     */
    private var startedActivities = 0

    /**
     * Takes a hold on the models for a camera screen.
     *
     * Cancels any pending release, so leaving one camera and opening another
     * inside the grace period finds them still loaded.
     */
    fun beginUse() {
        handler.removeCallbacks(releaseWhenIdle)
        models.beginUse()
    }

    /**
     * Gives back a hold, and starts the clock.
     *
     * Called from the analysis thread, so the release is posted rather than
     * run here.
     */
    fun endUse() {
        models.endUse()
        handler.removeCallbacks(releaseWhenIdle)
        handler.postDelayed(releaseWhenIdle, IDLE_GRACE_MS)
    }

    override fun onCreate() {
        super.onCreate()
        registerActivityLifecycleCallbacks(object : ActivityLifecycleCallbacks {
            override fun onActivityStarted(activity: Activity) {
                startedActivities++
            }

            override fun onActivityStopped(activity: Activity) {
                startedActivities--
                if (startedActivities > 0) return
                // Nothing of this app is on screen. Going between the picker
                // and the camera passes through here with the next screen
                // already started, so this is the app being put away rather
                // than a screen change.
                release("app not visible")
            }

            override fun onActivityCreated(activity: Activity, state: Bundle?) = Unit
            override fun onActivityResumed(activity: Activity) = Unit
            override fun onActivityPaused(activity: Activity) = Unit
            override fun onActivitySaveInstanceState(activity: Activity, out: Bundle) = Unit
            override fun onActivityDestroyed(activity: Activity) = Unit
        })
    }

    /** Still delivered, and still worth acting on. */
    override fun onLowMemory() {
        super.onLowMemory()
        release("under memory pressure")
    }

    private companion object {
        const val TAG = "UblApp"

        /**
         * How long models stay loaded with nothing holding them.
         *
         * Long enough to cover going back and picking the next slab, which is
         * the whole point, and short enough that a phone left on the picker
         * is not holding most of a gigabyte.
         */
        const val IDLE_GRACE_MS = 30_000L
    }
}

/** The pool, from anywhere with an Activity. */
val Activity.models: ModelPool<TfliteRunner> get() = (application as UblApp).models
