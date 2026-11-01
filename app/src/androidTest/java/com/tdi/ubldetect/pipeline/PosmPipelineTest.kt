package com.tdi.ubldetect.pipeline

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.tdi.ubldetect.catalog.Catalog
import com.tdi.ubldetect.catalog.CatalogEntry
import com.tdi.ubldetect.catalog.Mode
import com.tdi.ubldetect.imaging.RgbFrame
import com.tdi.ubldetect.infer.DelegateKind
import com.tdi.ubldetect.infer.TfliteRunner
import java.nio.ByteBuffer
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * The POSM path end to end on the device, against a real shop photo.
 *
 * Every piece below it is unit tested in isolation; this is the one that
 * proves they agree with each other and with the exported model.
 *
 * The fixture is a real image from the UBL pipeline. The expectation comes
 * from running that image through the same exported .tflite in Python with
 * cv2 preprocessing, which is the honest reference: class 13,
 * lifebuoy_relaunch_bunting, in skin_cleansing, at roughly (86, 246) in
 * source pixels.
 *
 * Deliberately NOT compared against YOLO(...).predict() on the .pt. That
 * path runs the one2many head through NMS while the export ships the
 * NMS-free one2one head, so the two are entitled to disagree on marginal
 * detections. See docs/model-contract.md.
 */
@RunWith(AndroidJUnit4::class)
class PosmPipelineTest {

    private val ctx = InstrumentationRegistry.getInstrumentation().targetContext

    private fun catalog(): Catalog =
        Catalog.parse(ctx.assets.open("catalog.json").bufferedReader().readText())

    private fun entry(key: String): CatalogEntry =
        catalog().entries(Mode.POSM).first { it.key == key }

    /** The fixture photo as the upright RGB the camera path would produce. */
    private fun fixtureFrame(name: String): RgbFrame {
        val bitmap = InstrumentationRegistry.getInstrumentation().context.assets.open(name)
            .use { BitmapFactory.decodeStream(it) }
        val w = bitmap.width
        val h = bitmap.height
        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)
        val buf = ByteBuffer.allocate(w * h * 3)
        for (i in pixels.indices) {
            val p = pixels[i]
            buf.put(i * 3, ((p shr 16) and 0xFF).toByte())
            buf.put(i * 3 + 1, ((p shr 8) and 0xFF).toByte())
            buf.put(i * 3 + 2, (p and 0xFF).toByte())
        }
        bitmap.recycle()
        return RgbFrame(buf, w, h)
    }

    private fun runner(asset: String) =
        TfliteRunner(ctx, asset, DelegateKind.CPU_XNNPACK)

    @Test
    fun processesAFixtureShelfImageAndLabelsAtLeastOneMaterial() {
        runner("models/posm_640.tflite").use { r ->
            val result = PosmPipeline(r, entry("skin_cleansing"), 0.30f)
                .process(fixtureFrame("posm_fixture.jpg"), 1L, useAllowlist = true)

            assertTrue("expected at least one detection", result.detections.isNotEmpty())
            assertTrue(result.timingsMs.containsKey("infer"))
            assertTrue(result.timingsMs.containsKey("prep"))
        }
    }

    @Test
    fun findsTheClassTheReferenceFoundOnThisImage() {
        runner("models/posm_640.tflite").use { r ->
            val result = PosmPipeline(r, entry("skin_cleansing"), 0.30f)
                .process(fixtureFrame("posm_fixture.jpg"), 1L, useAllowlist = true)
            assertTrue(
                "expected class 13, got ${result.detections.map { it.classId }}",
                result.detections.any { it.classId == 13 },
            )
        }
    }

    @Test
    fun boxLandsWhereTheReferencePutIt() {
        // This is the preprocessing guard, and it is the test that caught the
        // real one. Resampling without antialiasing (cv2's behaviour, and
        // Ultralytics') against resampling with it (PIL's) do not merely shift
        // this box, they change which class wins. A filtered downscale
        // "improvement" in InputTensor fails here.
        runner("models/posm_640.tflite").use { r ->
            val result = PosmPipeline(r, entry("skin_cleansing"), 0.30f)
                .process(fixtureFrame("posm_fixture.jpg"), 1L, useAllowlist = true)
            val box = result.detections.first { it.classId == 13 }.box
            assertEquals(86.2f, box.left, 20f)
            assertEquals(246.1f, box.top, 20f)
        }
    }

    @Test
    fun allowlistOffReturnsAtLeastAsManyDetectionsAsAllowlistOn() {
        runner("models/posm_640.tflite").use { r ->
            val p = PosmPipeline(r, entry("skin_cleansing"), 0.30f)
            val frame = fixtureFrame("posm_fixture.jpg")
            val on = p.process(frame, 1L, useAllowlist = true)
            val off = p.process(frame, 2L, useAllowlist = false)
            assertTrue(off.detections.size >= on.detections.size)
        }
    }

    @Test
    fun aCategoryThatDoesNotExpectTheseClassesFiltersThemOut() {
        // The same photo through the nutrition allowlist: class 13 is a skin
        // cleansing material, so it must not be labelled as a nutrition one.
        runner("models/posm_640.tflite").use { r ->
            val result = PosmPipeline(r, entry("nutrition"), 0.30f)
                .process(fixtureFrame("posm_fixture.jpg"), 1L, useAllowlist = true)
            assertTrue(result.detections.none { it.classId == 13 })
        }
    }

    @Test
    fun theResultCarriesTheFramesOwnTimestampAndTransform() {
        // The overlay needs both to place late results correctly.
        runner("models/posm_640.tflite").use { r ->
            val result = PosmPipeline(r, entry("skin_cleansing"), 0.30f)
                .process(fixtureFrame("posm_fixture.jpg"), 4242L, useAllowlist = true)
            assertEquals(4242L, result.frameTimestampNs)
            assertEquals(640, result.transform.inputSize)
        }
    }
}
