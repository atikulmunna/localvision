package com.tdi.ubldetect.ui

import android.Manifest
import android.content.Intent
import androidx.test.core.app.ActivityScenario
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.rule.GrantPermissionRule
import com.tdi.ubldetect.catalog.Mode
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
}
