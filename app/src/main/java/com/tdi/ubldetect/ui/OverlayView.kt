package com.tdi.ubldetect.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF as AndroidRectF
import android.util.AttributeSet
import android.view.View
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

    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 4f
        color = Color.rgb(0, 229, 160)
    }
    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.BLACK
        textSize = 34f
        isFakeBoldText = true
    }
    private val labelBackPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.rgb(0, 229, 160)
    }
    private val statusPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 30f
    }
    private val statusBackPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(150, 0, 0, 0)
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
            val w = labelPaint.measureText(text)
            // Above the box, or inside it when the box is against the top edge.
            val baseline = if (r.top > 40f) r.top - 10f else r.top + 36f
            canvas.drawRect(r.left, baseline - 30f, r.left + w + 16f, baseline + 8f, labelBackPaint)
            canvas.drawText(text, r.left + 8f, baseline, labelPaint)
        }

        if (status.isNotEmpty()) {
            canvas.drawRect(0f, 0f, width.toFloat(), 44f, statusBackPaint)
            canvas.drawText(status, 16f, 32f, statusPaint)
        }

        if (state.frozen) {
            canvas.drawText("FROZEN", width - 140f, 32f, statusPaint)
        }
    }
}
