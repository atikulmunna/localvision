package com.tdi.ubldetect

import android.view.View
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Content must not render underneath the status bar.
 *
 * Targeting SDK 35 or newer makes Android lay the app out edge to edge, so a
 * view placed at the top of the window draws behind the clock and the status
 * icons unless it is inset. That is how the skeleton first appeared on the
 * device: the title overlapped the system clock.
 *
 * This is pinned at the base rather than fixed per screen, because every
 * screen that follows (the picker, the overlay, the controls) sits in the same
 * window and would inherit the same defect.
 */
@RunWith(AndroidJUnit4::class)
class SystemBarInsetsTest {

    @Test
    fun contentIsInsetBelowTheStatusBar() {
        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            scenario.onActivity { activity ->
                val content = activity.findViewById<View>(android.R.id.content)
                val root = (content as android.view.ViewGroup).getChildAt(0)
                val bars = ViewCompat.getRootWindowInsets(content)
                    ?.getInsets(WindowInsetsCompat.Type.systemBars())

                // The device has a status bar, so there is a real inset to respect.
                assertTrue("expected a non-zero status bar inset", (bars?.top ?: 0) > 0)
                assertEquals(
                    "content must be padded clear of the status bar",
                    bars!!.top,
                    root.paddingTop,
                )
            }
        }
    }

    @Test
    fun contentIsInsetAboveTheNavigationBar() {
        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            scenario.onActivity { activity ->
                val content = activity.findViewById<View>(android.R.id.content)
                val root = (content as android.view.ViewGroup).getChildAt(0)
                val bars = ViewCompat.getRootWindowInsets(content)
                    ?.getInsets(WindowInsetsCompat.Type.systemBars())
                assertEquals(bars!!.bottom, root.paddingBottom)
            }
        }
    }
}
