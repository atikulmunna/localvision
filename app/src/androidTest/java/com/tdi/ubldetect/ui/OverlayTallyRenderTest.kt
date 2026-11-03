package com.tdi.ubldetect.ui

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.view.View
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.tdi.ubldetect.imaging.LetterboxTransform
import com.tdi.ubldetect.imaging.RectF
import com.tdi.ubldetect.imaging.ViewMapping
import com.tdi.ubldetect.model.Detection
import com.tdi.ubldetect.model.FrameResult
import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Draws the overlay with known detections, so the floating tally can be seen.
 *
 * A panel over a live preview cannot be checked by pointing a phone at a
 * desk: with nothing detected there is correctly nothing to draw. This
 * renders the view off screen against a fixed set of detections and writes a
 * PNG, which is both an assertion and something a person can look at.
 */
@RunWith(AndroidJUnit4::class)
class OverlayTallyRenderTest {

    private val ctx = InstrumentationRegistry.getInstrumentation().targetContext

    private val names = mapOf(
        11 to "dove_mask_25",
        12 to "sunsilk_serum_25",
        13 to "SUNSILK RELAUNCH BLACK 5PLUS1 POSTER PHASE1 AUG26",
        14 to "lux_soap",
    )

    private fun detection(cls: Int, i: Int) = Detection(
        RectF(40f + i * 30, 300f + i * 40, 240f + i * 30, 560f + i * 40),
        cls,
        0.80f + i * 0.01f,
    )

    private fun render(detections: List<Detection>, frozen: Boolean = false): Bitmap {
        val view = OverlayView(ctx)
        val w = 1080
        val h = 2340
        view.measure(
            View.MeasureSpec.makeMeasureSpec(w, View.MeasureSpec.EXACTLY),
            View.MeasureSpec.makeMeasureSpec(h, View.MeasureSpec.EXACTLY),
        )
        view.layout(0, 0, w, h)
        view.loadingMessage = null
        if (frozen) view.freeze()
        view.submit(
            FrameResult(
                detections = detections,
                frameTimestampNs = 0,
                transform = LetterboxTransform(1f, 0f, 0f, 640),
                timingsMs = mapOf("yuv" to 90L, "infer" to 179L),
            ),
            ViewMapping(1f, 0f, 0f),
        ) { names[it] ?: "class $it" }

        val bitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)
        canvas.drawColor(Color.rgb(90, 95, 100)) // stand-in for the preview
        view.draw(canvas)
        return bitmap
    }

    private fun write(bitmap: Bitmap, name: String): File {
        val dir = File(ctx.filesDir, "render").apply { mkdirs() }
        val f = File(dir, name)
        f.outputStream().use { bitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }
        return f
    }

    @Test
    fun tallyIsDrawnBelowTheStatusPill() {
        val dets = List(3) { detection(11, it) } +
            List(2) { detection(12, it) } +
            listOf(detection(13, 0), detection(14, 0))
        val bitmap = render(dets)
        write(bitmap, "tally.png")

        // The panel sits under the status pill at the top left. Its surface
        // is translucent black over the stand-in preview, so those pixels
        // must differ from the bare background.
        val background = Color.rgb(90, 95, 100)
        assertNotEquals(background, bitmap.getPixel(40, 300))
    }

    @Test
    fun aCrowdedShelfDoesNotFillTheScreenWithRows() {
        // 20 kinds fold into 8 lines, so the panel cannot run past the
        // controls and cover the preview it is describing.
        val dets = (1..20).flatMap { cls -> List(cls) { detection(cls, 0) } }
        val bitmap = render(dets)
        write(bitmap, "tally_crowded.png")

        val background = Color.rgb(90, 95, 100)
        // Well below any 8 row panel, and above the controls: bare preview.
        assertEquals(background, bitmap.getPixel(40, 900))
    }

    @Test
    fun aHeldFrameKeepsItsTally() {
        // Hold freezes the overlay, so the table has to describe the frame
        // being held rather than emptying out.
        val view = OverlayView(ctx)
        view.measure(
            View.MeasureSpec.makeMeasureSpec(1080, View.MeasureSpec.EXACTLY),
            View.MeasureSpec.makeMeasureSpec(2340, View.MeasureSpec.EXACTLY),
        )
        view.layout(0, 0, 1080, 2340)
        view.loadingMessage = null
        val labeller: (Int) -> String = { names[it] ?: "class $it" }
        val result = FrameResult(
            List(3) { detection(11, it) },
            0,
            LetterboxTransform(1f, 0f, 0f, 640),
            mapOf("infer" to 179L),
        )
        view.submit(result, ViewMapping(1f, 0f, 0f), labeller)
        view.freeze()

        val held = Bitmap.createBitmap(1080, 2340, Bitmap.Config.ARGB_8888)
        Canvas(held).also { it.drawColor(Color.rgb(90, 95, 100)) }.let { view.draw(it) }

        // A later empty frame must not clear what is on screen.
        view.submit(
            FrameResult(emptyList(), 1, LetterboxTransform(1f, 0f, 0f, 640), emptyMap()),
            ViewMapping(1f, 0f, 0f),
            labeller,
        )
        val after = Bitmap.createBitmap(1080, 2340, Bitmap.Config.ARGB_8888)
        Canvas(after).also { it.drawColor(Color.rgb(90, 95, 100)) }.let { view.draw(it) }
        write(after, "tally_held.png")

        assertTrue("the held tally was cleared", held.sameAs(after))
    }
}
