package com.tdi.ubldetect.ui

import android.view.View
import androidx.core.graphics.Insets
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat

/**
 * Keeps content clear of the status and navigation bars.
 *
 * Targeting SDK 35 or newer makes Android lay the window out edge to edge, so
 * anything at the top of the screen draws behind the clock and the status
 * icons unless it is inset. The skeleton's title did exactly that on the
 * S23+, overlapping the system clock, which is a 94 pixel status bar here.
 *
 * Applied per view rather than globally on purpose. The camera preview is
 * meant to be full bleed, filling the screen behind the bars; it is the
 * chrome drawn on top of it, the overlay labels and the controls, that has to
 * stay out from under them. So a screen insets its chrome and leaves its
 * preview alone.
 */
fun View.padForSystemBars(
    left: Boolean = true,
    top: Boolean = true,
    right: Boolean = true,
    bottom: Boolean = true,
) {
    // Captured before the listener runs, so repeated inset passes stay
    // additive-free: padding is always base plus the current inset, never
    // the previous padding plus the inset again.
    val base = Insets.of(paddingLeft, paddingTop, paddingRight, paddingBottom)

    ViewCompat.setOnApplyWindowInsetsListener(this) { view, windowInsets ->
        val bars = windowInsets.getInsets(WindowInsetsCompat.Type.systemBars())
        view.setPadding(
            base.left + if (left) bars.left else 0,
            base.top + if (top) bars.top else 0,
            base.right + if (right) bars.right else 0,
            base.bottom + if (bottom) bars.bottom else 0,
        )
        windowInsets
    }
    ViewCompat.requestApplyInsets(this)
}
