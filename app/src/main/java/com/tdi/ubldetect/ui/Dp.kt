package com.tdi.ubldetect.ui

import android.content.Context
import android.util.TypedValue
import android.view.View

/**
 * Density-independent pixels.
 *
 * Everything in this app's UI was originally written with raw pixel values,
 * which on a 450 dpi phone made 16 "padding" about 5dp and left every screen
 * looking cramped. Sizes go through here instead.
 */
fun Context.dp(value: Float): Int = TypedValue.applyDimension(
    TypedValue.COMPLEX_UNIT_DIP,
    value,
    resources.displayMetrics,
).toInt()

fun View.dp(value: Float): Int = context.dp(value)

/** Text sizes are set in sp, so they follow the reader's font preference. */
fun Context.sp(value: Float): Float = TypedValue.applyDimension(
    TypedValue.COMPLEX_UNIT_SP,
    value,
    resources.displayMetrics,
)
