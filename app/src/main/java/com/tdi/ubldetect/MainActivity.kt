package com.tdi.ubldetect

import android.app.Activity
import android.os.Bundle
import android.widget.TextView

/**
 * Placeholder launcher, replaced by PickerActivity in Task 10.
 *
 * It exists so the skeleton can be installed and launched on the device
 * before any camera or model code is written.
 */
class MainActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(TextView(this).apply {
            text = "UBL Detect Viewer: skeleton"
            textSize = 20f
        })
    }
}
