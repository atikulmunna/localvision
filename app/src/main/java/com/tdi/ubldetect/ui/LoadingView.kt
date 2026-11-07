package com.tdi.ubldetect.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import android.view.animation.AnimationUtils
import com.tdi.ubldetect.R
import com.tdi.ubldetect.infer.LoadProgress
import kotlin.math.abs
import kotlin.math.cos

/**
 * What the screen shows while the models are opening.
 *
 * There is a real wait here and it cannot be designed away. Opening the QPDS
 * pair costs about four seconds on this device, almost all of it the GPU
 * driver preparing the model, and it is paid once per mode rather than once
 * per slab. What can be fixed is what the wait looks like: before this, the
 * screen showed a live preview that drew no boxes with a small grey pill
 * reading "Loading model" in the corner, which reads as a broken detector
 * rather than a busy one.
 *
 * Nothing but the spinner and two short lines, drawn straight onto the
 * preview. No panel, no scrim, no explanation: the camera is already live
 * behind this and covering it to say so was the wrong trade. Legibility
 * comes from a shadow under each mark rather than from a box, which is what
 * keeps it readable over a white poster without hiding the shelf.
 *
 * ## The count is real
 *
 * The count comes from the pool's own [LoadProgress], so it moves when a
 * model actually finishes rather than on a timer. It can sit on 0/2 for
 * three seconds and that is honest: both models are genuinely still opening.
 * The spinner carries the "still working" message in the meantime, which is
 * what a spinner is for.
 *
 * Allocates nothing in `onDraw`. It animates at display rate for several
 * seconds, so the paints and the rectangle are built once.
 */
class LoadingView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyle: Int = 0,
) : View(context, attrs, defStyle) {

    private val ringPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = dp(3f).toFloat()
        strokeCap = Paint.Cap.ROUND
        color = context.color(R.color.ring_track)
    }
    private val arcPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = dp(3f).toFloat()
        strokeCap = Paint.Cap.ROUND
        color = context.color(R.color.accent_bright)
    }

    /**
     * A darker, wider stroke under the arc, in place of a shadow.
     *
     * `setShadowLayer` is honoured for text on a hardware accelerated
     * canvas and quietly ignored for everything else, so the spinner would
     * have had no shadow at all over a bright shelf. Two strokes do work.
     */
    private val arcHaloPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = dp(5.5f).toFloat()
        strokeCap = Paint.Cap.ROUND
        color = context.color(R.color.spinner_halo)
    }
    private val titlePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = context.color(R.color.chrome_text)
        textSize = context.sp(16f)
        textAlign = Paint.Align.CENTER
        isFakeBoldText = true
        setShadowLayer(dp(5f).toFloat(), 0f, dp(1f).toFloat(), Color.BLACK)
    }
    private val countPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = context.color(R.color.chrome_text_dim)
        textSize = context.sp(13f)
        textAlign = Paint.Align.CENTER
        setShadowLayer(dp(5f).toFloat(), 0f, dp(1f).toFloat(), Color.BLACK)
    }

    private val scratch = RectF()

    private var title = ""
    private var progress = LoadProgress(0, 0)

    /**
     * Where the count comes from, read on each frame rather than pushed.
     *
     * The pool publishes progress from whichever thread is doing the
     * opening. Pulling it here means the view is never showing a count the
     * pool has already moved past, and nothing has to post to the main
     * thread while it is busy.
     */
    var progressSource: (() -> LoadProgress)? = null

    /** Shows the spinner, from the start of its animation. */
    fun show(title: String) {
        this.title = title
        if (visibility != VISIBLE) {
            alpha = 0f
            visibility = VISIBLE
            animate().alpha(1f).setDuration(FADE_IN_MS).start()
        }
        postInvalidateOnAnimation()
    }

    /** Sets the count directly, for a test or a caller without a pool. */
    fun update(progress: LoadProgress) {
        this.progress = progress
        if (visibility == VISIBLE) postInvalidateOnAnimation()
    }

    /**
     * Fades out and stops drawing.
     *
     * Faded rather than removed, because the models often arrive while this
     * is still fading in, and something that flashes on and off is worse
     * than nothing at all. Idempotent: the camera calls it on every path
     * that could leave the loading state.
     */
    fun hide() {
        if (visibility != VISIBLE) return
        animate().alpha(0f).setDuration(FADE_OUT_MS).withEndAction {
            visibility = GONE
            alpha = 1f
        }.start()
    }

    override fun onDraw(canvas: Canvas) {
        progressSource?.let { progress = it() }

        val ring = dp(42f)
        val cx = width / 2f
        var y = height / 2f - dp(64f)

        drawSpinner(canvas, cx, y + ring / 2f, ring / 2f - arcPaint.strokeWidth)
        y += ring + dp(22f)

        canvas.drawText(title, cx, y + titlePaint.textSize * 0.85f, titlePaint)
        y += titlePaint.textSize + dp(10f)

        canvas.drawText(
            "loading ${progress.done}/${progress.total}",
            cx,
            y + countPaint.textSize * 0.85f,
            countPaint,
        )

        postInvalidateOnAnimation()
    }

    /**
     * One arc, turning, with a sweep that breathes.
     *
     * A fixed arc turning at a fixed rate reads as a loop rather than as
     * work. Varying the sweep is what the platform's own indeterminate
     * spinner does, and it is the difference between "animated" and "alive".
     */
    private fun drawSpinner(canvas: Canvas, cx: Float, cy: Float, r: Float) {
        scratch.set(cx - r, cy - r, cx + r, cy + r)
        canvas.drawArc(scratch, 0f, 360f, false, ringPaint)

        val now = AnimationUtils.currentAnimationTimeMillis()
        val turn = (now % TURN_MS) / TURN_MS.toFloat()
        val breathe = (now % BREATHE_MS) / BREATHE_MS.toFloat()
        // cos gives a smooth there-and-back with no seam at the wrap.
        val sweep = MIN_SWEEP +
            (MAX_SWEEP - MIN_SWEEP) * (0.5f - 0.5f * cos(breathe * 2f * Math.PI).toFloat())
        // The start walks back by half the growth, so the arc grows from
        // both ends rather than dragging one end around the circle.
        val start = turn * 360f - abs(sweep) / 2f
        canvas.drawArc(scratch, start, sweep, false, arcHaloPaint)
        canvas.drawArc(scratch, start, sweep, false, arcPaint)
    }

    private companion object {
        const val FADE_IN_MS = 160L
        const val FADE_OUT_MS = 220L

        /** One turn of the arc. */
        const val TURN_MS = 1100L

        /**
         * One growth and shrink of the sweep.
         *
         * Deliberately not a multiple of [TURN_MS]: equal periods make the
         * arc grow at the same point on the circle every time, which the eye
         * picks up as a stutter.
         */
        const val BREATHE_MS = 1700L

        const val MIN_SWEEP = 24f
        const val MAX_SWEEP = 290f
    }
}
