package com.tdi.ubldetect.ui

import android.content.Context
import android.graphics.Bitmap
import com.tdi.ubldetect.imaging.RgbFrame
import com.tdi.ubldetect.model.FrameResult
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import org.json.JSONArray
import org.json.JSONObject

/**
 * Writes a frozen frame and its labels to app-private storage.
 *
 * App-private, not the gallery: these are shelf photographs taken inside
 * somebody's shop, and this app has no business putting them where other
 * apps can read them. They come back out with `adb pull` when wanted.
 *
 * The JSON sidecar carries the detections and the per-stage timings, so a
 * saved frame is enough to argue about later: what was detected, how
 * confidently, where, and how long each stage took.
 */
class FrameSaver(private val context: Context) {

    fun save(frame: RgbFrame, result: FrameResult?, labeller: (Int) -> String): File {
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val dir = File(context.filesDir, "frames").apply { mkdirs() }

        val jpeg = File(dir, "$stamp.jpg")
        toBitmap(frame).use { bitmap ->
            jpeg.outputStream().use { bitmap.value.compress(Bitmap.CompressFormat.JPEG, 92, it) }
        }

        val labels = JSONArray()
        result?.detections?.forEach { d ->
            labels.put(
                JSONObject()
                    .put("label", labeller(d.classId))
                    .put("classId", d.classId)
                    .put("score", d.score)
                    .put("left", d.box.left)
                    .put("top", d.box.top)
                    .put("right", d.box.right)
                    .put("bottom", d.box.bottom),
            )
        }
        File(dir, "$stamp.json").writeText(
            JSONObject()
                .put("width", frame.width)
                .put("height", frame.height)
                .put("timingsMs", JSONObject(result?.timingsMs?.mapKeys { it.key } ?: emptyMap<String, Long>()))
                .put("detections", labels)
                .toString(2),
        )
        return jpeg
    }

    /** Wraps the bitmap so it is always recycled, even if compression throws. */
    private class Recycling(val value: Bitmap) : AutoCloseable {
        override fun close() = value.recycle()
    }

    private fun toBitmap(frame: RgbFrame): Recycling {
        val pixels = IntArray(frame.width * frame.height)
        val src = frame.buffer
        for (i in pixels.indices) {
            val r = src.get(i * 3).toInt() and 0xFF
            val g = src.get(i * 3 + 1).toInt() and 0xFF
            val b = src.get(i * 3 + 2).toInt() and 0xFF
            pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
        return Recycling(
            Bitmap.createBitmap(pixels, frame.width, frame.height, Bitmap.Config.ARGB_8888),
        )
    }

    companion object {
        /** Where saved frames land, for the message shown after a save. */
        fun directory(context: Context): File = File(context.filesDir, "frames")
    }
}
