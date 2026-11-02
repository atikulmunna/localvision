package com.tdi.ubldetect.pipeline

import android.graphics.BitmapFactory
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.tdi.ubldetect.catalog.Catalog
import com.tdi.ubldetect.catalog.CatalogEntry
import com.tdi.ubldetect.catalog.Mode
import com.tdi.ubldetect.imaging.RgbFrame
import com.tdi.ubldetect.infer.DelegateKind
import com.tdi.ubldetect.infer.TfliteRunner
import com.tdi.ubldetect.ui.ControlsState
import java.io.File
import java.nio.ByteBuffer
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Runs the app's own pipeline over fixture images and writes its labels out.
 *
 * This is the app side of the Task 16 comparison. `tools/reference_labels.py`
 * produces the other side from the same `.tflite` files in Python, and
 * `tools/diff_labels.py` says whether a difference is a preprocessing bug or
 * quantization drift.
 *
 * It is a test rather than a debug activity on purpose: the harness already
 * grants permissions, runs on the device, and gives somewhere to put the
 * output, and it avoids exporting an activity whose only job is debugging.
 *
 * Pull the results with:
 *   adb shell run-as com.tdi.ubldetect cat files/labeldump/posm.json
 */
@RunWith(AndroidJUnit4::class)
class LabelDumpTest {

    private val ctx = InstrumentationRegistry.getInstrumentation().targetContext

    private fun catalog(): Catalog =
        Catalog.parse(ctx.assets.open("catalog.json").bufferedReader().readText())

    private fun entry(mode: Mode, key: String): CatalogEntry =
        catalog().entries(mode).first { it.key == key }

    /** Every image bundled in the test APK's assets. */
    private fun fixtureNames(): List<String> =
        InstrumentationRegistry.getInstrumentation().context.assets.list("")
            .orEmpty()
            .filter { it.endsWith(".jpg", ignoreCase = true) }
            .sorted()

    private fun frame(name: String): RgbFrame {
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

    private fun write(fileName: String, perImage: Map<String, List<JSONObject>>) {
        val root = JSONObject()
        perImage.forEach { (image, dets) ->
            root.put(image, JSONObject().put("detections", JSONArray(dets)))
        }
        val dir = File(ctx.filesDir, "labeldump").apply { mkdirs() }
        File(dir, fileName).writeText(root.toString(2))
    }

    private fun dump(pipeline: FramePipeline, state: ControlsState, out: String) {
        val names = fixtureNames()
        assertTrue("no fixture images bundled in the test APK", names.isNotEmpty())

        val perImage = names.associateWith { name ->
            pipeline.process(frame(name), 1L, state).detections.map { d ->
                JSONObject()
                    .put("classId", d.classId)
                    .put("score", d.score.toDouble())
                    .put("left", d.box.left.toDouble())
                    .put("top", d.box.top.toDouble())
                    .put("right", d.box.right.toDouble())
                    .put("bottom", d.box.bottom.toDouble())
            }
        }
        write(out, perImage)
    }

    @Test
    fun dumpPosmLabelsWithoutTheAllowlist() {
        // No allowlist, so this compares the detector itself rather than the
        // catalog's filtering. Restricting here would hide a geometry error
        // behind an empty result.
        TfliteRunner(ctx, "models/posm_640.tflite", DelegateKind.CPU_XNNPACK).use { r ->
            dump(
                PosmPipeline(r, entry(Mode.POSM, "skin_cleansing")),
                ControlsState(useAllowlist = false),
                "posm.json",
            )
        }
    }

    @Test
    fun dumpQpdsLabelsForOneSlab() {
        // QPDS has to stay restricted: stage two is a choice among the slab's
        // SKUs, so an open classifier would be answering a different question.
        TfliteRunner(ctx, "models/qpds_seg_640.tflite", DelegateKind.CPU_XNNPACK).use { seg ->
            TfliteRunner(ctx, "models/qpds_cls_224.tflite", DelegateKind.CPU_XNNPACK).use { cls ->
                dump(
                    QpdsPipeline(seg, com.tdi.ubldetect.infer.ClsRunner(cls),
                        entry(Mode.QPDS, "DOVE_LUX_INTERNATIONAL")),
                    ControlsState(),
                    "qpds.json",
                )
            }
        }
    }
}
