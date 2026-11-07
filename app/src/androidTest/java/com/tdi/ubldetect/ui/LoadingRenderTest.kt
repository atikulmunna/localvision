package com.tdi.ubldetect.ui

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.view.View
import android.widget.FrameLayout
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.tdi.ubldetect.infer.LoadProgress
import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Draws the loading spinner, so the wait can be looked at rather than
 * imagined.
 *
 * It only exists for the few seconds a model takes to open, which is exactly
 * when nobody is holding a phone and a screenshot tool at once. This renders
 * it off screen over a stand-in preview and writes a PNG.
 */
@RunWith(AndroidJUnit4::class)
class LoadingRenderTest {

    private val ctx = InstrumentationRegistry.getInstrumentation().targetContext

    private val w = 1080
    private val h = 2340

    private fun dpPx(value: Int) = (value * ctx.resources.displayMetrics.density).toInt()

    /**
     * On the main thread, because [LoadingView.show] fades the card in and
     * an animator may only be started from a Looper thread. The overlay's
     * render test gets away with the instrumentation thread; this one does
     * not, and the failure is an exception rather than a wrong pixel.
     */
    /**
     * Draws the card the way the camera screen does, and for two reasons.
     *
     * On the main thread, because [LoadingView.show] fades the card in and
     * an animator may only be started from a Looper thread.
     *
     * Through a parent, because `View.draw` paints whatever it is asked to
     * paint: visibility is honoured by the ViewGroup drawing its children,
     * so a card called hidden still draws itself if you ask it directly.
     * Testing it without a parent asserted nothing about what reaches the
     * screen.
     */
    private fun render(progress: LoadProgress, show: Boolean = true): Bitmap {
        lateinit var bitmap: Bitmap
        InstrumentationRegistry.getInstrumentation().runOnMainSync {
            val view = LoadingView(ctx)
            view.visibility = View.GONE
            if (show) view.show("QPDS Model")
            view.update(progress)
            // The fade is still running, and drawing honours alpha, so a
            // card captured mid fade would be a flaky assertion about
            // opacity.
            view.alpha = 1f

            val parent = FrameLayout(ctx)
            parent.addView(view)
            parent.measure(
                View.MeasureSpec.makeMeasureSpec(w, View.MeasureSpec.EXACTLY),
                View.MeasureSpec.makeMeasureSpec(h, View.MeasureSpec.EXACTLY),
            )
            parent.layout(0, 0, w, h)

            bitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
            val canvas = Canvas(bitmap)
            canvas.drawColor(Color.rgb(190, 195, 200)) // stand-in for a bright shelf
            parent.draw(canvas)
        }
        return bitmap
    }

    private fun save(bitmap: Bitmap, name: String) {
        val out = File(ctx.filesDir, name)
        out.outputStream().use { bitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }
        assertTrue("nothing written to $out", out.length() > 0)
    }

    /** How many pixels are close to the accent, which is only ours. */
    private fun accentPixels(bitmap: Bitmap): Int {
        var found = 0
        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)
        for (p in pixels) {
            if (Color.red(p) < 80 && Color.green(p) > 160 && Color.blue(p) > 100) found++
        }
        return found
    }

    @Test
    fun theShelfStaysVisibleBehindTheSpinner() {
        // The point of having no panel: the camera is already live, and
        // covering it to say the models are loading was the wrong trade.
        val bitmap = render(LoadProgress(1, 2))
        save(bitmap, "loading.png")
        val shelf = Color.rgb(190, 195, 200)
        assertEquals("something is covering the top of the preview", shelf, bitmap.getPixel(40, 40))
        assertEquals(
            "something is covering the middle of the preview",
            shelf,
            bitmap.getPixel(w - 40, h / 2),
        )
    }

    @Test
    fun theSpinnerIsDrawn() {
        val bitmap = render(LoadProgress(1, 2))
        assertTrue("no accent pixels, so no spinner", accentPixels(bitmap) > 300)
    }

    @Test
    fun theCountSaysWhatHasOpened() {
        // 0/2 and 1/2 are different pixels. Without this the count could be
        // wired to nothing and the test would still pass on the spinner.
        val zero = render(LoadProgress(0, 2))
        val one = render(LoadProgress(1, 2))
        val band = h / 2 + dpPx(30)
        var differing = 0
        for (x in 0 until w) {
            if (zero.getPixel(x, band) != one.getPixel(x, band)) differing++
        }
        assertTrue(
            "0/2 and 1/2 drew the same row of pixels",
            differing > 0,
        )
    }

    @Test
    fun hiddenItDrawsNothingAtAll() {
        // It sits on top of the controls and swallows taps, so a spinner
        // left behind would block the panel for the rest of the session.
        val bitmap = render(LoadProgress(2, 2), show = false)
        assertEquals(
            "something was drawn over the preview",
            Color.rgb(190, 195, 200),
            bitmap.getPixel(w / 2, h / 2),
        )
        assertEquals(0, accentPixels(bitmap))
    }
}
