package com.tdi.ubldetect

import android.graphics.BitmapFactory
import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.tdi.ubldetect.catalog.Catalog
import com.tdi.ubldetect.catalog.CatalogEntry
import com.tdi.ubldetect.catalog.Mode
import com.tdi.ubldetect.imaging.RgbFrame
import com.tdi.ubldetect.imaging.YuvPlanes
import com.tdi.ubldetect.imaging.YuvToRgb
import com.tdi.ubldetect.infer.ClsRunner
import com.tdi.ubldetect.infer.DelegateKind
import com.tdi.ubldetect.infer.TfliteRunner
import com.tdi.ubldetect.pipeline.PosmPipeline
import com.tdi.ubldetect.pipeline.QpdsPipeline
import com.tdi.ubldetect.ui.ControlsState
import java.io.File
import java.nio.ByteBuffer
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Measures what the app actually costs per frame, so the spec's estimates
 * can be replaced with numbers.
 *
 * Run over fixture images rather than the live camera on purpose: a live
 * frame's cost depends on what the phone happens to be pointed at (stage two
 * costs one inference per detected product), so live timings are not
 * comparable between runs. A fixed image makes the comparison between
 * resolutions and delegates mean something.
 *
 * Results land in `files/bench/results.json`. Drive this with `am instrument`
 * rather than `connectedAndroidTest`, which uninstalls the app afterwards and
 * takes the results with it.
 */
@RunWith(AndroidJUnit4::class)
class BenchmarkTest {

    private val ctx = InstrumentationRegistry.getInstrumentation().targetContext
    private val results = JSONArray()

    private fun entry(mode: Mode, key: String): CatalogEntry =
        Catalog.parse(ctx.assets.open("catalog.json").bufferedReader().readText())
            .entries(mode).first { it.key == key }

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

    private fun record(row: JSONObject) {
        results.put(row)
        android.util.Log.i("Benchmark", row.toString())
    }

    /** Mean of the per-stage timings across [runs], after warming up. */
    private fun timeStages(
        runs: Int,
        warmup: Int = 3,
        once: () -> Map<String, Long>,
    ): Pair<Map<String, Double>, Double> {
        repeat(warmup) { once() }
        val totals = HashMap<String, Long>()
        var wall = 0L
        repeat(runs) {
            val t0 = SystemClock.elapsedRealtime()
            once().forEach { (k, v) -> totals[k] = (totals[k] ?: 0L) + v }
            wall += SystemClock.elapsedRealtime() - t0
        }
        return totals.mapValues { it.value.toDouble() / runs } to wall.toDouble() / runs
    }

    private fun benchPosm(imgsz: Int, delegate: DelegateKind, runs: Int = 20) {
        val loadStart = SystemClock.elapsedRealtime()
        TfliteRunner(ctx, "models/posm_$imgsz.tflite", delegate).use { runner ->
            val loadMs = SystemClock.elapsedRealtime() - loadStart
            val pipeline = PosmPipeline(runner, entry(Mode.POSM, "skin_cleansing"))
            val f = frame("posm_fixture.jpg")
            val state = ControlsState(imgsz = imgsz, useAllowlist = false)
            val (stages, wall) = timeStages(runs) {
                pipeline.process(f, 1L, state).timingsMs
            }
            record(
                JSONObject()
                    .put("pipeline", "POSM")
                    .put("imgsz", imgsz)
                    .put("delegate", runner.activeDelegate.name)
                    .put("requested", delegate.name)
                    .put("loadMs", loadMs)
                    .put("runs", runs)
                    .put("wallMs", wall)
                    .put("stages", JSONObject(stages.mapValues { it.value })),
            )
        }
    }

    private fun benchQpds(imgsz: Int, delegate: DelegateKind, runs: Int = 10) {
        val loadStart = SystemClock.elapsedRealtime()
        TfliteRunner(ctx, "models/qpds_seg_$imgsz.tflite", delegate).use { seg ->
            TfliteRunner(ctx, "models/qpds_cls_224.tflite", delegate).use { cls ->
                val loadMs = SystemClock.elapsedRealtime() - loadStart
                val pipeline = QpdsPipeline(
                    seg, ClsRunner(cls), entry(Mode.QPDS, "DOVE_LUX_INTERNATIONAL"),
                )
                val f = frame("qpds_fixture.jpg")
                val state = ControlsState(imgsz = imgsz)
                val (stages, wall) = timeStages(runs) {
                    pipeline.process(f, 1L, state).timingsMs
                }
                record(
                    JSONObject()
                        .put("pipeline", "QPDS")
                        .put("imgsz", imgsz)
                        .put("delegate", seg.activeDelegate.name)
                        .put("requested", delegate.name)
                        .put("loadMs", loadMs)
                        .put("runs", runs)
                        .put("wallMs", wall)
                        .put("stages", JSONObject(stages.mapValues { it.value })),
                )
            }
        }
    }

    /**
     * The YUV conversion, which is paid on every frame before any model runs
     * and does not appear in either pipeline's own timings.
     *
     * Measured at the analysis stream's real size, because its cost is
     * proportional to source pixels.
     */
    private fun benchYuvConversion(width: Int, height: Int, runs: Int = 30) {
        val y = ByteBuffer.allocate(width * height)
        val cw = (width + 1) / 2
        val ch = (height + 1) / 2
        val u = ByteBuffer.allocate(cw * ch)
        val v = ByteBuffer.allocate(cw * ch)
        val planes = YuvPlanes(y, u, v, width, height, width, cw, 1)
        val out = ByteBuffer.allocate(width * height * 3)

        repeat(3) { YuvToRgb.convert(planes, 90, out) }
        val t0 = SystemClock.elapsedRealtime()
        repeat(runs) { YuvToRgb.convert(planes, 90, out) }
        val each = (SystemClock.elapsedRealtime() - t0).toDouble() / runs

        record(
            JSONObject()
                .put("pipeline", "YuvToRgb")
                .put("source", "${width}x$height")
                .put("runs", runs)
                .put("wallMs", each),
        )
    }

    @Test
    fun benchmarkEveryCombination() {
        for (delegate in listOf(DelegateKind.GPU, DelegateKind.CPU_XNNPACK)) {
            for (imgsz in ControlsState.OFFERED_SIZES) {
                benchPosm(imgsz, delegate)
            }
            // qpds_seg is shipped at 640 only; 960 would report as missing.
            benchQpds(640, delegate)
        }
        benchYuvConversion(720, 1280)
        benchYuvConversion(480, 640)

        val dir = File(ctx.filesDir, "bench").apply { mkdirs() }
        File(dir, "results.json").writeText(results.toString(2))
    }
}
