package com.tdi.ubldetect

import android.app.Activity
import android.os.Bundle
import android.widget.TextView
import com.tdi.ubldetect.ui.padForSystemBars

/**
 * Placeholder launcher, replaced by PickerActivity in Task 10.
 *
 * It exists so the skeleton can be installed and launched on the device
 * before any camera or model code is written.
 */
class MainActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val label = TextView(this).apply {
            text = "UBL Detect Viewer: skeleton"
            textSize = 20f
        }
        setContentView(label)
        // The window is laid out edge to edge, so this would otherwise draw
        // behind the status bar.
        label.padForSystemBars()
    }
}
