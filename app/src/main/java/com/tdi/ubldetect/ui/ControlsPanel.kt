package com.tdi.ubldetect.ui

import android.content.Context
import android.graphics.drawable.GradientDrawable
import android.util.TypedValue
import android.view.Gravity
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.SeekBar
import android.widget.TextView
import com.tdi.ubldetect.R
import com.tdi.ubldetect.infer.DelegateKind

/**
 * The controls strip along the bottom of the camera screen.
 *
 * Dark and translucent on purpose. It sits over a live preview that swings
 * between a white poster and a dark shelf, and light chrome would vanish
 * against the first.
 *
 * Every control answers a question that is otherwise unanswerable from
 * inside the app:
 *
 * - **Confidence** separates "the model did not see it" from "the model saw
 *   it faintly".
 * - **Filter off** separates "not detected" from "detected as something this
 *   category does not expect", which look identical while filtering is on.
 * - **Resolution** trades lag against recall.
 * - **GPU** is the only way to compare delegates on real frames.
 * - **Hold** makes labels readable without holding the phone still.
 * - **Save** keeps a frame and its labels for arguing about later.
 *
 * Chips show their state rather than only acting: the active resolution and
 * an engaged toggle carry the accent, so the readout is not the only place
 * that says what is on.
 */
class ControlsPanel(
    context: Context,
    private var state: ControlsState,
    private val onChange: (ControlsState) -> Unit,
    private val onSave: () -> Unit,
) : LinearLayout(context) {

    private val readout = TextView(context).apply {
        setTextColor(context.color(R.color.chrome_text))
        setTextSize(TypedValue.COMPLEX_UNIT_SP, 12f)
        alpha = 0.85f
    }

    private val chips = mutableMapOf<String, TextView>()

    init {
        orientation = VERTICAL
        setBackgroundColor(context.color(R.color.chrome))
        setPadding(dp(16f), dp(12f), dp(16f), dp(12f))
        layoutParams = ViewGroup.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT,
        )

        addView(readout)

        addView(
            SeekBar(context).apply {
                max = 90
                progress = ((state.scoreThreshold - 0.05f) * 100).toInt()
                setPadding(0, dp(8f), 0, dp(4f))
                setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                    override fun onProgressChanged(bar: SeekBar?, value: Int, fromUser: Boolean) {
                        if (fromUser) push(state.copy(scoreThreshold = 0.05f + value / 100f))
                    }
                    override fun onStartTrackingTouch(bar: SeekBar?) = Unit
                    override fun onStopTrackingTouch(bar: SeekBar?) = Unit
                })
            },
        )

        addView(
            LinearLayout(context).apply {
                orientation = HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                // Equal weights: six fixed-width buttons overflowed 1080 px,
                // wrapping one letter per line and pushing SAVE off screen.
                for (size in ControlsState.OFFERED_SIZES) {
                    addView(chip("$size") { push(state.copy(imgsz = size)) })
                }
                addView(chip("Filter") { push(state.copy(useAllowlist = !state.useAllowlist)) })
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
                addView(chip("Boxes") { push(state.copy(showBoxes = !state.showBoxes)) })
                addView(chip("Hold") { push(state.copy(frozen = !state.frozen)) })
                addView(chip("Save") { onSave() })
            },
        )

        render()
    }

    private fun chip(label: String, onClick: () -> Unit) = TextView(context).apply {
        text = label
        gravity = Gravity.CENTER
        setTextSize(TypedValue.COMPLEX_UNIT_SP, 12f)
        setTextColor(context.color(R.color.chrome_text))
        setPadding(0, dp(9f), 0, dp(9f))
        isClickable = true
        setOnClickListener { onClick() }
        layoutParams = LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply {
            marginStart = dp(3f)
            marginEnd = dp(3f)
        }
        chips[label] = this
    }

    private fun setChipActive(label: String, active: Boolean) {
        chips[label]?.background = GradientDrawable().apply {
            cornerRadius = dp(9f).toFloat()
            setColor(context.color(if (active) R.color.chip_active else R.color.chip))
        }
    }

    private fun push(next: ControlsState) {
        state = next
        render()
        onChange(next)
    }

    /** Called back by the activity when it adopts, or declines, a change. */
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
        }
        for (size in ControlsState.OFFERED_SIZES) {
            setChipActive("$size", state.imgsz == size)
        }
        setChipActive("Filter", state.useAllowlist)
        setChipActive("Boxes", state.showBoxes)
        setChipActive("GPU", state.delegate == DelegateKind.GPU)
        setChipActive("Hold", state.frozen)
        setChipActive("Save", false)
    }
}

internal fun Context.color(id: Int) = resources.getColor(id, theme)
