package com.tdi.ubldetect.infer

import com.tdi.ubldetect.imaging.Letterbox
import com.tdi.ubldetect.imaging.LetterboxTransform
import com.tdi.ubldetect.imaging.RectF
import com.tdi.ubldetect.model.Detection
import kotlin.math.roundToInt

/**
 * Reads the NMS-free detection head.
 *
 * The contract, measured from the exported model and recorded in
 * docs/model-contract.md: `[1, 300, 6]`, each row `x1, y1, x2, y2, score,
 * class`, boxes in letterboxed input pixels, scores already through a
 * sigmoid, rows sorted by score descending, unused tail zero filled.
 *
 * No non-maximum suppression, because the end2end head emits one row per
 * detection rather than a cloud of candidates. That is a property of
 * exporting with `nms=False`; the default export emits the raw one2many head
 * instead, which is duplicate-heavy and a different shape, so a re-export
 * without the flag fails Task 8's shape assertion rather than silently
 * producing stacks of boxes here.
 */
object Yolo26DetectDecoder {

    private const val STRIDE = 6

    /**
     * @param allowed class ids to keep, or null to keep every class, which is
     *   what the allowlist toggle turns off
     * @param t the transform the frame was letterboxed with, used to put boxes
     *   back into source coordinates
     */
    fun decode(
        raw: FloatArray,
        shape: IntArray,
        scoreThreshold: Float,
        allowed: Set<Int>?,
        t: LetterboxTransform,
    ): List<Detection> {
        val rowCount = shape[1]
        val out = ArrayList<Detection>()

        for (i in 0 until rowCount) {
            val o = i * STRIDE
            val score = raw[o + 4]

            // Rows are score-sorted, so the first row below the threshold ends
            // the useful part of the tensor. On a real frame only a handful of
            // the 300 slots carry anything, so this skips most of the work
            // every frame rather than decoding a zero-filled tail.
            if (score < scoreThreshold) break

            // Rounded, not truncated: the class arrives as a float, and
            // truncating 15.999999 would name a different product.
            val classId = raw[o + 5].roundToInt()
            if (allowed != null && classId !in allowed) continue

            out.add(
                Detection(
                    box = Letterbox.rectToSource(
                        RectF(raw[o], raw[o + 1], raw[o + 2], raw[o + 3]),
                        t,
                    ),
                    classId = classId,
                    score = score,
                ),
            )
        }
        return out
    }
}
