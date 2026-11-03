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
import org.junit.Assert.assertTrue
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
     * Leaving during a model swap must not strand an interpreter.
     *
     * The teardown wait was sized for a frame, at "roughly 400 ms", but model
     * loads run on the same single thread and cost about 3.5 s each. Tap a
     * resolution and press back inside that window and the wait expired with
     * the load still running, so the activity closed a runner list the
     * loading thread had not filled in yet. Measured at the time: four
     * interpreters left open after one cycle.
     *
     * The assertion is non-growth rather than zero, because models are now
     * deliberately kept between screens: the pool holds the working set so
     * the next category does not reload it. A leak shows up as a count that
     * climbs with every cycle; the working set does not.
     */
    @Test
    fun leavingDuringAResolutionSwapDoesNotStrandInterpreters() {
        val counts = mutableListOf<Int>()
        repeat(3) {
            ActivityScenario.launch<CameraActivity>(
                cameraIntent(Mode.POSM, "hair_care"),
            ).use { scenario ->
                // The first load has to finish, or the swap is a no-op.
                Thread.sleep(SWAP_SETTLE_MS)
                scenario.onActivity { activity ->
                    val chip = findByText(activity.window.decorView, "960")
                    assertNotNull("no 960 chip in the controls panel", chip)
                    chip!!.performClick()
                }
                // Mid-swap: a 57 MB model is opening on the analysis thread
                // and the activity is about to go away under it.
                Thread.sleep(300)
            }
            // The release is queued behind the load, so it lands after
            // onDestroy returns rather than inside it.
            Thread.sleep(RELEASE_WAIT_MS)
            counts += TfliteRunner.live.get()
        }

        assertTrue(
            "interpreters grew across cycles: $counts",
            counts.all { it <= WORKING_SET_MAX },
        )
        assertEquals("interpreters grew across cycles: $counts", counts[0], counts[2])
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
        /** Long enough for the first model to finish opening. */
        const val SWAP_SETTLE_MS = 6000L

        /** Generous: what it waits for is queued behind a 3.5 s load. */
        const val RELEASE_WAIT_MS = 9000L

        /**
         * Most interpreters the pool should ever hold.
         *
         * Two: QPDS's segmenter and classifier. A POSM swap closes the old
         * resolution before opening the new one, so even mid-swap it is one.
         */
        const val WORKING_SET_MAX = 2
    }
}
