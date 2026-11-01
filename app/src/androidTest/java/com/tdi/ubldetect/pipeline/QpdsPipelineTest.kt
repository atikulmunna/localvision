package com.tdi.ubldetect.pipeline

import android.graphics.BitmapFactory
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.tdi.ubldetect.catalog.Catalog
import com.tdi.ubldetect.catalog.CatalogEntry
import com.tdi.ubldetect.catalog.Mode
import com.tdi.ubldetect.imaging.RgbFrame
import com.tdi.ubldetect.infer.ClsRunner
import com.tdi.ubldetect.infer.DelegateKind
import com.tdi.ubldetect.infer.TfliteRunner
import com.tdi.ubldetect.ui.ControlsState
import java.nio.ByteBuffer
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * The QPDS two-stage path on the device, against a real shelf photo.
 *
 * The expectation comes from running the same image through the same
 * exported models in Python with cv2 preprocessing while writing
 * docs/model-contract.md: 24 instances at `seg_conf` 0.20, and the top
 * instance classified as `dove_serum_bar` (class 15) at 1.000. The photo is a
 * DOVE and LUX International shelf, and the DOVE_LUX_INTERNATIONAL slab
 * expects exactly that class.
 */
@RunWith(AndroidJUnit4::class)
class QpdsPipelineTest {

    private val ctx = InstrumentationRegistry.getInstrumentation().targetContext

    private fun slab(key: String): CatalogEntry =
        Catalog.parse(ctx.assets.open("catalog.json").bufferedReader().readText())
            .entries(Mode.QPDS).first { it.key == key }

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

    private fun <T> withPipeline(
        state: ControlsState = ControlsState(),
        key: String = "DOVE_LUX_INTERNATIONAL",
        block: (Pair<QpdsPipeline, ControlsState>) -> T,
    ): T =
        TfliteRunner(ctx, "models/qpds_seg_640.tflite", DelegateKind.CPU_XNNPACK).use { seg ->
            TfliteRunner(ctx, "models/qpds_cls_224.tflite", DelegateKind.CPU_XNNPACK).use { clsRunner ->
                block(QpdsPipeline(seg, ClsRunner(clsRunner), slab(key)) to state)
            }
        }

    @Test
    fun labelsProductsOnAFixtureShelfImage() {
        withPipeline { (p, st) ->
            val r = p.process(fixtureFrame("qpds_fixture.jpg"), 1L, st)
            assertTrue("expected at least one labelled product", r.detections.isNotEmpty())
            assertTrue(r.timingsMs.containsKey("prep"))
            assertTrue(r.timingsMs.containsKey("seg"))
            assertTrue(r.timingsMs.keys.any { it.startsWith("cls") })
        }
    }

    @Test
    fun findsTheProductThePythonReferenceIdentified() {
        withPipeline { (p, st) ->
            val r = p.process(fixtureFrame("qpds_fixture.jpg"), 1L, st)
            assertTrue(
                "expected class 15 dove_serum_bar, got ${r.detections.map { it.classId }}",
                r.detections.any { it.classId == 15 },
            )
        }
    }

    @Test
    fun everyLabelComesFromTheSlabsOwnSkus() {
        // The whole point of picking a slab. A label from outside this set
        // would mean the allowlist is not reaching the classifier.
        val expected = slab("DOVE_LUX_INTERNATIONAL").classIds
        withPipeline { (p, st) ->
            val r = p.process(fixtureFrame("qpds_fixture.jpg"), 1L, st)
            assertTrue(r.detections.all { it.classId in expected })
        }
    }

    @Test
    fun instanceCapBoundsTheWorkDone() {
        // Stage two costs one inference per instance, so an uncapped crowded
        // shelf would stall the loop.
        withPipeline(ControlsState(instanceCap = 3)) { (p, st) ->
            val r = p.process(fixtureFrame("qpds_fixture.jpg"), 1L, st)
            assertTrue("expected at most 3, got ${r.detections.size}", r.detections.size <= 3)
        }
    }

    @Test
    fun theResultCarriesTheFramesTimestampAndTransform() {
        withPipeline { (p, st) ->
            val r = p.process(fixtureFrame("qpds_fixture.jpg"), 99L, st)
            assertEquals(99L, r.frameTimestampNs)
            assertEquals(640, r.transform.inputSize)
        }
    }

    @Test
    fun extraTimingsAreCarriedThrough() {
        withPipeline { (p, st) ->
            val r = p.process(fixtureFrame("qpds_fixture.jpg"), 1L, st, mapOf("yuv" to 7L))
            assertEquals(7L, r.timingsMs["yuv"])
        }
    }
}
