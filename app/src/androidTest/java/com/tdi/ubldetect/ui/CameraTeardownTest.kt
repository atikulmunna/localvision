package com.tdi.ubldetect.ui

import android.Manifest
import android.content.Intent
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.test.core.app.ActivityScenario
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.rule.GrantPermissionRule
import com.tdi.ubldetect.catalog.Mode
import com.tdi.ubldetect.infer.TfliteRunner
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Leaving the camera screen must not take the process with it.
 *
 * Reported from the device: going back to pick a different category crashed
 * the app. The log showed SIGSEGV, a null pointer dereference inside
 * libtensorflowlite_jni.so on the analysis thread.
 *
 * The cause is a teardown race. `executor.shutdown()` stops new work being
 * accepted but does not wait for the frame already running, so the activity
 * closed the interpreter while native code was still inside `run()`. The
 * interpreter is a native object; using it after close is undefined, and here
 * it was a segfault.
 *
 * This is the same class of bug as the model swap in Task 11, which was fixed
 * by serialising onto the analysis thread. Teardown was simply not covered by
 * that fix.
 *
 * The test opens and closes the camera repeatedly, because the race needs a
 * frame to be in flight at the moment of closing and one attempt can miss it.
 */
@RunWith(AndroidJUnit4::class)
class CameraTeardownTest {

    @get:Rule
    val cameraPermission: GrantPermissionRule =
        GrantPermissionRule.grant(Manifest.permission.CAMERA)

    private fun cameraIntent(mode: Mode, key: String) =
        Intent(ApplicationProvider.getApplicationContext(), CameraActivity::class.java)
            .putExtra(CameraActivity.EXTRA_MODE, mode.name)
            .putExtra(CameraActivity.EXTRA_KEY, key)

    private fun openThenClose(mode: Mode, key: String, settleMs: Long) {
        ActivityScenario.launch<CameraActivity>(cameraIntent(mode, key)).use {
            // Long enough for the camera to bind and a frame to be mid-flight
            // when the scenario closes, which is the moment that crashed.
            Thread.sleep(settleMs)
        }
    }

    @Test
    fun leavingThePosmCameraDoesNotCrashTheProcess() {
        repeat(3) { openThenClose(Mode.POSM, "hair_care", 1500) }
    }

    @Test
    fun leavingTheQpdsCameraDoesNotCrashTheProcess() {
        // QPDS holds two interpreters and its frames take longer, so the
        // window for closing one mid-inference is wider.
        repeat(3) { openThenClose(Mode.QPDS, "DOVE_LUX_INTERNATIONAL", 2000) }
    }

    @Test
    fun closingImmediatelyAfterOpeningIsAlsoSafe() {
        // The other end of the race: tearing down while the camera is still
        // binding and no frame has arrived yet.
        repeat(3) { openThenClose(Mode.POSM, "hair_care", 150) }
    }

    /**
     * Leaving during a model swap must still release the models.
     *
     * The teardown wait was sized for a frame, at "roughly 400 ms", but model
     * loads run on the same single thread and cost about 3.5 s each, so QPDS
     * spends some 7 s loading two. Tap a resolution and press back inside
     * that window and the wait expires with the load still running. The
     * activity then closes a list of runners that the loading thread has not
     * filled in yet, so both new interpreters stay open for the life of the
     * process, and if the timing lands the other way it closes one the
     * loading thread is still using.
     *
     * The existing tests cannot see this: none of them swaps, and a leak
     * raises nothing.
     */
    @Test
    fun leavingDuringAResolutionSwapReleasesEveryInterpreter() {
        repeat(2) {
            ActivityScenario.launch<CameraActivity>(
                cameraIntent(Mode.QPDS, "DOVE_LUX_INTERNATIONAL"),
            ).use { scenario ->
                // The first load has to finish, or the swap is a no-op.
                Thread.sleep(SWAP_SETTLE_MS)
                scenario.onActivity { activity ->
                    val chip = findByText(activity.window.decorView, "960")
                    assertNotNull("no 960 chip in the controls panel", chip)
                    chip!!.performClick()
                }
                // Mid-swap: two interpreters are opening on the analysis
                // thread and the activity is about to go away under them.
                Thread.sleep(300)
            }
            // Release is queued behind the load, so it completes after
            // onDestroy returns rather than inside it.
            waitForNoLiveInterpreters()
            assertEquals("interpreters left open after teardown", 0, TfliteRunner.live.get())
        }
    }

    private fun waitForNoLiveInterpreters() {
        val deadline = System.currentTimeMillis() + RELEASE_WAIT_MS
        while (TfliteRunner.live.get() != 0 && System.currentTimeMillis() < deadline) {
            Thread.sleep(100)
        }
    }

    /** The controls are built in code with no ids, so the label is the handle. */
    private fun findByText(root: View, text: String): View? {
        if (root is TextView && root.text == text) return root
        if (root is ViewGroup) {
            for (i in 0 until root.childCount) {
                findByText(root.getChildAt(i), text)?.let { return it }
            }
        }
        return null
    }

    private companion object {
        /** Long enough for QPDS to finish opening both models. */
        const val SWAP_SETTLE_MS = 9000L

        /** Generous: the release it is waiting on is behind a 3.5 s load. */
        const val RELEASE_WAIT_MS = 15000L
    }
}
