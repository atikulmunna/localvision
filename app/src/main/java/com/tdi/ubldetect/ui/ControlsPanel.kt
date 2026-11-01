package com.tdi.ubldetect.ui

import android.content.Context
import android.graphics.Color
import android.view.Gravity
import android.view.ViewGroup
import android.widget.Button
import android.widget.LinearLayout
import android.widget.SeekBar
import android.widget.TextView
import com.tdi.ubldetect.infer.DelegateKind

/**
 * The controls strip along the bottom of the camera screen.
 *
 * Every control here exists to answer a question that is otherwise
 * unanswerable from inside the app:
 *
 * - **Confidence** separates "the model did not see it" from "the model saw
 *   it faintly".
 * - **Allowlist off** separates "not detected" from "detected as something
 *   this category does not expect", which look identical while filtering is
 *   on and are completely different problems.
 * - **Resolution** trades lag against recall, which is the lever the spec
 *   singles out.
 * - **Delegate** is the only way to compare GPU against CPU on real frames.
 * - **Freeze** makes labels readable without holding the phone still.
 * - **Save** keeps a frame and its labels for arguing about later.
 */
class ControlsPanel(
    context: Context,
    private var state: ControlsState,
    private val onChange: (ControlsState) -> Unit,
    private val onSave: () -> Unit,
) : LinearLayout(context) {

    private val readout = TextView(context).apply {
        setTextColor(Color.WHITE)
        textSize = 13f
    }

    init {
        orientation = VERTICAL
        setBackgroundColor(Color.argb(170, 0, 0, 0))
        setPadding(PAD, PAD, PAD, PAD)
        layoutParams = ViewGroup.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT,
        )

        addView(readout)

        addView(
            SeekBar(context).apply {
                max = 90
                progress = ((state.scoreThreshold - 0.05f) * 100).toInt()
                setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                    override fun onProgressChanged(bar: SeekBar?, value: Int, fromUser: Boolean) {
                        if (fromUser) push(state.copy(scoreThreshold = 0.05f + value / 100f))
                    }
                    override fun onStartTrackingTouch(bar: SeekBar?) = Unit
                    override fun onStopTrackingTouch(bar: SeekBar?) = Unit
                })
            },
        )

        // Equal weights, not wrap content. Six fixed-width buttons overflowed
        // 1080 pixels: FREEZE wrapped one letter per line and SAVE was pushed
        // off the screen entirely.
        addView(
            LinearLayout(context).apply {
                orientation = HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                for (size in ControlsState.OFFERED_SIZES) {
                    addView(chip("$size") { push(state.copy(imgsz = size)) })
                }
                addView(chip("FILTER") { push(state.copy(useAllowlist = !state.useAllowlist)) })
                addView(
                    chip("GPU") {
                        val next = if (state.delegate == DelegateKind.GPU) {
                            DelegateKind.CPU_XNNPACK
                        } else {
                            DelegateKind.GPU
                        }
                        push(state.copy(delegate = next))
                    },
                )
                addView(chip("HOLD") { push(state.copy(frozen = !state.frozen)) })
                addView(chip("SAVE") { onSave() })
            },
        )

        render()
    }

    private fun chip(label: String, onClick: () -> Unit) = Button(context).apply {
        text = label
        textSize = 11f
        setPadding(0, 0, 0, 0)
        minWidth = 0
        minimumWidth = 0
        setOnClickListener { onClick() }
        layoutParams = LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
            .apply { marginStart = 4; marginEnd = 4 }
    }

    private fun push(next: ControlsState) {
        state = next
        render()
        onChange(next)
    }

    /** Called back by the activity when it adopts (or declines) a change. */
    fun adopt(next: ControlsState) {
        state = next
        render()
    }

    private fun render() {
        readout.text = buildString {
            append("conf ")
            append((state.scoreThreshold * 100).toInt())
            append("%   ")
            append(state.imgsz)
            append("   filter ")
            append(if (state.useAllowlist) "on" else "off")
            append("   ")
            append(if (state.delegate == DelegateKind.GPU) "gpu" else "cpu")
            if (state.frozen) append("   frozen")
        }
    }

    private companion object {
        const val PAD = 12
    }
}
