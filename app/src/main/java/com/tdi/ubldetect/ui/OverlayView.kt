package com.tdi.ubldetect.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF as AndroidRectF
import android.util.AttributeSet
import android.view.View
import com.tdi.ubldetect.R
import com.tdi.ubldetect.imaging.PreviewMapping
import com.tdi.ubldetect.imaging.ViewMapping
import com.tdi.ubldetect.model.FrameResult

/**
 * Draws detection boxes and labels over the camera preview.
 *
 * Decides nothing. [OverlayState] holds what should be on screen and why;
 * this paints it and maps source-frame coordinates into view coordinates with
 * [PreviewMapping], which is the fill-centre crop the preview itself uses and
 * deliberately not the letterbox the model saw.
 *
 * Allocates nothing in `onDraw`. It runs on every frame of a 30 frames per
 * second preview, so paints and the scratch rectangle are created once.
 */
class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyle: Int = 0,
) : View(context, attrs, defStyle) {

    val state = OverlayState()

    private var mapping: ViewMapping = ViewMapping(1f, 0f, 0f)
    private var labeller: (Int) -> String = { "class $it" }
    private var status: String = ""

    /**
     * Shown instead of the timings until the models are ready.
     *
     * The camera binds in about 125 ms but a cold model load takes seconds,
     * so without this the person watches a live preview draw no boxes and
     * reasonably concludes the detector is broken.
     */
    var loadingMessage: String? = "Loading model"
        set(value) {
            field = value
            invalidate()
        }

    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = dp(2.5f).toFloat()
        color = context.color(R.color.accent_bright)
    }
    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.BLACK
        textSize = context.sp(13f)
        isFakeBoldText = true
    }
    private val labelBackPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = context.color(R.color.accent_bright)
    }
    private val statusPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = context.color(R.color.chrome_text)
        textSize = context.sp(12f)
    }
    private val statusBackPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = context.color(R.color.chrome)
    }
    private val holdPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = context.color(R.color.accent_bright)
        textSize = context.sp(12f)
        isFakeBoldText = true
    }

    private val scratch = AndroidRectF()

    /**
     * @param result the newest completed frame, or null when there is none
     * @param m how that frame maps onto this view
     * @param labeller class id to display name, from the chosen catalog entry
     */
    fun submit(result: FrameResult?, m: ViewMapping, labeller: (Int) -> String) {
        mapping = m
        this.labeller = labeller
        state.submit(result)
        status = buildStatus(result)
        invalidate()
    }

    fun freeze() {
        state.freeze()
        invalidate()
    }

    fun unfreeze() {
        state.unfreeze()
        invalidate()
    }

    private fun buildStatus(result: FrameResult?): String {
        if (result == null) return status
        val timings = result.timingsMs.entries.joinToString(" ") { "${it.key} ${it.value}ms" }
        return "${result.detections.size} found  $timings"
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        for (d in state.drawn) {
            val r = PreviewMapping.mapRect(d.box, mapping)
            scratch.set(r.left, r.top, r.right, r.bottom)
            canvas.drawRect(scratch, boxPaint)

            val text = "${labeller(d.classId)}  ${(d.score * 100).toInt()}%"
            val padH = dp(6f).toFloat()
            val w = labelPaint.measureText(text)
            val tall = dp(19f).toFloat()
            // Above the box, or inside it when the box is against the top edge.
            val baseline = if (r.top > tall + dp(4f)) r.top - dp(5f) else r.top + tall + dp(3f)
            // Clamped so a long name on a box near the right edge stays on
            // screen instead of running off it, which is where labels went
            // before: readable only up to "...PHASE1 AU".
            val left = (r.left).coerceIn(0f, (width - w - padH * 2).coerceAtLeast(0f))
            scratch.set(left, baseline - tall, left + w + padH * 2, baseline + dp(5f))
            canvas.drawRoundRect(scratch, dp(4f).toFloat(), dp(4f).toFloat(), labelBackPaint)
            canvas.drawText(text, left + padH, baseline, labelPaint)
        }

        // Padding does not move the canvas, it only reserves space, so the
        // status line has to honour it explicitly. Without this it draws at
        // y=0 and sits on top of the system clock, which is exactly the
        // overlap this app already had once.
        val top = paddingTop.toFloat()
        val line = loadingMessage ?: status
        if (line.isNotEmpty()) {
            // A pill rather than a full-width bar: it covers less of the
            // preview, which is the thing the person is actually looking at.
            val padH = dp(10f).toFloat()
            val w = statusPaint.measureText(line)
            val left = paddingLeft + dp(12f).toFloat()
            scratch.set(left, top + dp(6f), left + w + padH * 2, top + dp(32f))
            canvas.drawRoundRect(scratch, dp(13f).toFloat(), dp(13f).toFloat(), statusBackPaint)
            canvas.drawText(line, left + padH, top + dp(24f), statusPaint)
        }

        if (state.frozen) {
            val label = "HOLD"
            val padH = dp(10f).toFloat()
            val w = holdPaint.measureText(label)
            val right = width - paddingRight - dp(12f).toFloat()
            scratch.set(right - w - padH * 2, top + dp(6f), right, top + dp(32f))
            canvas.drawRoundRect(scratch, dp(13f).toFloat(), dp(13f).toFloat(), statusBackPaint)
            canvas.drawText(label, right - w - padH, top + dp(24f), holdPaint)
        }
    }
}
