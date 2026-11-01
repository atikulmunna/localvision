package com.tdi.ubldetect.infer

import com.tdi.ubldetect.imaging.Letterbox
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Decoding the NMS-free head.
 *
 * Per docs/model-contract.md, measured from the exported file: the output is
 * [1, 300, 6] where each row is x1, y1, x2, y2, score, class, the boxes are in
 * letterboxed input pixels, the scores are already through a sigmoid, the rows
 * arrive sorted by score descending and the unused tail is zero filled.
 *
 * There is no NMS here, and that is a property of the export rather than an
 * omission: `nms=False` asks for the end2end head, which emits one row per
 * detection. The default export instead emits the raw one2many head, which is
 * duplicate-heavy and would need suppression. If someone re-exports without
 * that flag, `noNmsIsApplied` is not what breaks: the contract test in Task 8
 * breaks first, on the output shape.
 */
class Yolo26DetectDecoderTest {

    private val shape = intArrayOf(1, 300, 6)
    private val t = Letterbox.compute(1920, 1080, 640)

    /** Builds a [1,300,6] buffer from rows of (x1,y1,x2,y2,score,class). */
    private fun rows(vararg entries: FloatArray): FloatArray {
        val raw = FloatArray(300 * 6)
        entries.forEachIndexed { i, e -> e.copyInto(raw, i * 6) }
        return raw
    }

    private fun row(score: Float, cls: Int = 0, x1: Float = 100f, y1: Float = 200f) =
        floatArrayOf(x1, y1, x1 + 50f, y1 + 40f, score, cls.toFloat())

    @Test
    fun keepsOnlyCandidatesAtOrAboveThreshold() {
        val out = Yolo26DetectDecoder.decode(
            rows(row(0.95f), row(0.30f), row(0.01f)), shape, 0.30f, null, t,
        )
        assertEquals(2, out.size)
    }

    @Test
    fun noNmsIsApplied() {
        // Two identical overlapping boxes of the same class both survive. The
        // end2end head does not emit duplicates, so suppressing them here
        // would be removing detections the model deliberately kept.
        val out = Yolo26DetectDecoder.decode(
            rows(row(0.90f, cls = 3), row(0.80f, cls = 3)), shape, 0.30f, null, t,
        )
        assertEquals(2, out.size)
    }

    @Test
    fun allowlistDropsOtherClasses() {
        val out = Yolo26DetectDecoder.decode(
            rows(row(0.90f, cls = 3), row(0.80f, cls = 41)), shape, 0.30f, setOf(3), t,
        )
        assertEquals(listOf(3), out.map { it.classId })
    }

    @Test
    fun nullAllowlistKeepsEveryClass() {
        val out = Yolo26DetectDecoder.decode(
            rows(row(0.90f, cls = 3), row(0.80f, cls = 41)), shape, 0.30f, null, t,
        )
        assertEquals(2, out.size)
    }

    @Test
    fun boxesComeBackInSourceCoordinates() {
        // The centre of the model square is the centre of the frame.
        val tt = Letterbox.compute(1080, 1920, 640)
        val out = Yolo26DetectDecoder.decode(
            rows(floatArrayOf(315f, 315f, 325f, 325f, 0.9f, 0f)), shape, 0.30f, null, tt,
        )
        assertEquals(540f, out[0].box.centerX(), 2f)
        assertEquals(960f, out[0].box.centerY(), 2f)
    }

    @Test
    fun allBelowThresholdYieldsEmptyListNotNull() {
        assertEquals(0, Yolo26DetectDecoder.decode(rows(row(0.01f)), shape, 0.30f, null, t).size)
    }

    @Test
    fun stopsAtTheFirstRowBelowThresholdInsteadOfWalkingAllThreeHundred() {
        // Rows are score-sorted, so a row below threshold means every later row
        // is too. On the probe image only 7 of 300 slots carried anything, so
        // walking the zero-filled tail every frame is wasted work. A stray high
        // score placed after a low one must therefore NOT be picked up: that is
        // the observable consequence of stopping early, and it is correct.
        val raw = rows(row(0.90f), row(0.10f))
        row(0.99f).copyInto(raw, 50 * 6)
        val out = Yolo26DetectDecoder.decode(raw, shape, 0.30f, null, t)
        assertEquals(1, out.size)
    }

    @Test
    fun classIdRoundsRatherThanTruncates() {
        // Column 5 is a float. Truncating 15.999999 gives class 15, which is a
        // different product entirely.
        val out = Yolo26DetectDecoder.decode(
            rows(floatArrayOf(10f, 10f, 20f, 20f, 0.9f, 15.999999f)), shape, 0.30f, null, t,
        )
        assertEquals(16, out[0].classId)
    }

    @Test
    fun reproducesTheMeasuredContractDetection() {
        // docs/model-contract.md, the exported model's top row on a real photo
        // as both the app and cv2 produce it: raw (28.76, 221.98, 370.47,
        // 291.46) score 0.559 class 13, un-letterboxing to about (86.2, 246.1).
        val tt = Letterbox.compute(1919, 1080, 640)
        val out = Yolo26DetectDecoder.decode(
            rows(floatArrayOf(28.761932f, 221.97615f, 370.47034f, 291.4572f, 0.5594f, 13f)),
            shape, 0.30f, null, tt,
        )
        assertEquals(1, out.size)
        assertEquals(13, out[0].classId)
        assertEquals(0.5594f, out[0].score, 1e-4f)
        assertEquals(86.2f, out[0].box.left, 0.5f)
        assertEquals(246.1f, out[0].box.top, 0.5f)
    }

    @Test
    fun zeroFilledTailIsNotDecodedAsADetectionAtTheOrigin() {
        // Every unused slot is six zeros, which reads as a zero-area box of
        // class 0 at score 0. Without the threshold these would be 300 phantom
        // detections stacked at the top left corner.
        val out = Yolo26DetectDecoder.decode(FloatArray(300 * 6), shape, 0.30f, null, t)
        assertTrue(out.isEmpty())
    }
}
