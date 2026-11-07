package com.tdi.ubldetect.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF as AndroidRectF
import android.text.TextPaint
import android.text.TextUtils
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
     * One line of the floating tally, measured and shortened ready to draw.
     *
     * Built in [submit] rather than in `onDraw`, which runs on every frame of
     * a 30 fps preview and allocates nothing by design.
     */
    private class TallyLine(val name: CharSequence, val count: String, val nameWidth: Float)

    private var tallyLines: List<TallyLine> = emptyList()
    private var tallyWidth = 0f

    /**
     * Whether to draw the boxes and their labels.
     *
     * The tally is deliberately not affected. With boxes off the preview is
     * clear and the counts still say what is up there, which is the reason
     * to turn them off in the first place.
     */
    var showBoxes: Boolean = true
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
    private val tallyBackPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = context.color(R.color.float_surface)
    }
    private val tallyBorderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = dp(1f).toFloat()
        color = context.color(R.color.float_border)
    }
    // A TextPaint, not a Paint: TextUtils.ellipsize measures with one.
    private val tallyNamePaint = TextPaint(Paint.ANTI_ALIAS_FLAG).apply {
        color = context.color(R.color.chrome_text)
        textSize = context.sp(12f)
    }
    private val tallyCountPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
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
        // Only when the result was taken. A refused one, because the
        // overlay is frozen or because it arrived out of order, must not
        // leave a live count and live timings sitting above a held picture.
        if (state.submit(result)) {
            status = buildStatus(state.drawn.size, result)
            buildTally()
        }
        invalidate()
    }

    fun freeze() {
        state.freeze()
        invalidate()
    }

    /**
     * Rebuilds the tally lines for whatever is currently drawn.
     *
     * Long names are shortened to fit rather than wrapped or allowed to run
     * off: half the POSM names are longer than a phone is wide, and the count
     * is the column worth keeping.
     */
    private fun buildTally() {
        val rows = Tally.of(state.drawn, labeller, TALLY_MAX_ROWS)
        if (rows.isEmpty()) {
            tallyLines = emptyList()
            tallyWidth = 0f
            return
        }
        val countWidth = rows.maxOf { tallyCountPaint.measureText(it.count.toString()) }
        val room = width * TALLY_WIDTH_SHARE - dp(20f) - dp(14f) - countWidth
        val nameRoom = room.coerceAtLeast(dp(60f).toFloat())

        var widest = 0f
        tallyLines = rows.map { row ->
            val name = TextUtils.ellipsize(
                row.label, tallyNamePaint, nameRoom, TextUtils.TruncateAt.END,
            )
            val w = tallyNamePaint.measureText(name, 0, name.length)
            widest = maxOf(widest, w)
            TallyLine(name, row.count.toString(), w)
        }
        tallyWidth = dp(20f) + widest + dp(14f) + countWidth
    }

    fun unfreeze() {
        state.unfreeze()
        invalidate()
    }

    private fun buildStatus(drawn: Int, result: FrameResult?): String {
        if (result == null) return status
        val timings = result.timingsMs.entries.joinToString(" ") { "${it.key} ${it.value}ms" }
        return "$drawn found  $timings"
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        for (d in if (showBoxes) state.drawn else emptyList()) {
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
        // Loading has its own card now, drawn by LoadingView over the top of
        // this one, so the pill is only ever the live timings.
        val line = status
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

        // The tally floats under the status pill, left aligned with it.
        if (tallyLines.isNotEmpty()) {
            val left = paddingLeft + dp(12f).toFloat()
            val rowH = dp(19f).toFloat()
            val panelTop = top + dp(40f)
            val panelBottom = panelTop + dp(14f) + rowH * tallyLines.size
            scratch.set(left, panelTop, left + tallyWidth, panelBottom)
            val radius = dp(10f).toFloat()
            canvas.drawRoundRect(scratch, radius, radius, tallyBackPaint)
            canvas.drawRoundRect(scratch, radius, radius, tallyBorderPaint)

            val textLeft = left + dp(10f)
            val countRight = left + tallyWidth - dp(10f)
            var baseline = panelTop + dp(7f) + rowH - dp(5f)
            for (line in tallyLines) {
                canvas.drawText(line.name, 0, line.name.length, textLeft, baseline, tallyNamePaint)
                val cw = tallyCountPaint.measureText(line.count)
                canvas.drawText(line.count, countRight - cw, baseline, tallyCountPaint)
                baseline += rowH
            }
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

    private companion object {
        /**
         * Lines the tally will show before folding the rest into "+N more".
         *
         * A crowded shelf can carry twenty kinds, and listing them all would
         * cover the preview, which is the thing being looked at.
         */
        const val TALLY_MAX_ROWS = 8

        /** How much of the width the panel may take before names are cut. */
        const val TALLY_WIDTH_SHARE = 0.62f
    }
}
