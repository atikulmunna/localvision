package com.tdi.ubldetect.imaging

import com.tdi.ubldetect.infer.MaskGrid
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Building the crop the classifier was trained on.
 *
 * This mirrors `_detect_products_two_stage` in UBL-standalone's
 * `core/detection.py`. The geometry is not a free choice: QPDS-cls learned on
 * crops built exactly this way, so an unmasked or differently padded crop is
 * off-distribution and the classifier's answers stop meaning what they say.
 *
 * Two details differ from the detector's letterbox and both matter:
 *
 * - padding here is **black**, where the detector pads 114 grey
 * - the background inside the box is **blacked out by the mask**, so a
 *   neighbouring product leaning into the box does not vote on the label
 */
class CropBuilderTest {

    private val size = 224

    private fun solid(width: Int, height: Int, r: Int, g: Int, b: Int): RgbFrame {
        val buf = ByteBuffer.allocate(width * height * 3)
        for (i in 0 until width * height) {
            buf.put(i * 3, r.toByte())
            buf.put(i * 3 + 1, g.toByte())
            buf.put(i * 3 + 2, b.toByte())
        }
        return RgbFrame(buf, width, height)
    }

    private fun out() = ByteBuffer.allocateDirect(size * size * 3 * 4).order(ByteOrder.nativeOrder())

    /** Channel value at an output pixel. NCHW, so channels are whole planes. */
    private fun at(buf: ByteBuffer, channel: Int, x: Int, y: Int): Float =
        buf.asFloatBuffer().get(channel * size * size + y * size + x)

    private fun fullMask(w: Int = 8, h: Int = 8) = MaskGrid(FloatArray(w * h) { 1f }, w, h)

    private fun uniformMask(v: Float, w: Int = 8, h: Int = 8) = MaskGrid(FloatArray(w * h) { v }, w, h)

    /** Mask covering only the left half of the box. */
    private fun leftHalfMask(w: Int = 8, h: Int = 8) =
        MaskGrid(FloatArray(w * h) { if (it % w < w / 2) 1f else 0f }, w, h)

    @Test
    fun pixelsOutsideTheMaskBecomeBlack() {
        val b = out()
        assertTrue(
            CropBuilder.build(
                solid(100, 100, 255, 255, 255), RectF(0f, 0f, 100f, 100f),
                leftHalfMask(), 0.5f, size, b,
            ),
        )
        assertEquals(1f, at(b, 0, 40, 112), 1e-3f)   // inside the mask
        assertEquals(0f, at(b, 0, 200, 112), 1e-3f)  // outside it
    }

    @Test
    fun outputIsSquareAndAspectPreservedWithBlackPadding() {
        // A 200 by 100 box into 224: content is 224 wide and 112 tall, centred,
        // with black above and below.
        val b = out()
        CropBuilder.build(solid(300, 300, 255, 255, 255), RectF(0f, 0f, 200f, 100f), fullMask(), 0.5f, size, b)
        assertEquals(0f, at(b, 0, 112, 5), 1e-3f)     // top padding
        assertTrue(at(b, 0, 112, 112) > 0.9f)          // centred content
        assertEquals(0f, at(b, 0, 112, 218), 1e-3f)    // bottom padding
    }

    @Test
    fun paddingIsBlackNotTheDetectorsGrey() {
        // 114 grey is right for the detector and wrong here. The classifier
        // trained on black-padded crops.
        val b = out()
        CropBuilder.build(solid(300, 300, 255, 255, 255), RectF(0f, 0f, 200f, 100f), fullMask(), 0.5f, size, b)
        assertEquals(0f, at(b, 1, 112, 5), 1e-3f)
        assertEquals(0f, at(b, 2, 112, 5), 1e-3f)
    }

    @Test
    fun zeroSizeBoxAfterTruncationIsRejected() {
        // The server skips degenerate crops; so does this, rather than
        // feeding the classifier an empty image and believing the answer.
        assertFalse(
            CropBuilder.build(
                solid(100, 100, 255, 255, 255), RectF(10.2f, 10.2f, 10.8f, 10.8f),
                fullMask(), 0.5f, size, out(),
            ),
        )
    }

    @Test
    fun boxExtendingPastTheFrameEdgeIsClampedNotWrapped() {
        val b = out()
        assertTrue(
            CropBuilder.build(
                solid(100, 100, 255, 255, 255), RectF(80f, 80f, 140f, 140f),
                fullMask(), 0.5f, size, b,
            ),
        )
        // The clamped 20 by 20 region letterboxes into the square without
        // reading out of bounds, which would otherwise be a crash or garbage.
        assertTrue(at(b, 0, 112, 112) > 0.9f)
    }

    @Test
    fun aBoxEntirelyOutsideTheFrameIsRejected() {
        assertFalse(
            CropBuilder.build(
                solid(100, 100, 255, 255, 255), RectF(200f, 200f, 260f, 260f),
                fullMask(), 0.5f, size, out(),
            ),
        )
    }

    @Test
    fun maskThresholdDecidesInclusion() {
        // A uniform 0.4 mask at threshold 0.5 blacks the whole crop.
        val b = out()
        CropBuilder.build(
            solid(100, 100, 255, 255, 255), RectF(0f, 0f, 100f, 100f),
            uniformMask(0.4f), 0.5f, size, b,
        )
        assertEquals(0f, at(b, 0, 112, 112), 1e-3f)
    }

    @Test
    fun maskAtOrAboveThresholdIsIncluded() {
        val b = out()
        CropBuilder.build(
            solid(100, 100, 255, 255, 255), RectF(0f, 0f, 100f, 100f),
            uniformMask(0.5f), 0.5f, size, b,
        )
        assertTrue(at(b, 0, 112, 112) > 0.9f)
    }

    @Test
    fun layoutIsNchwAndNormalised() {
        val b = out()
        CropBuilder.build(
            solid(100, 100, 255, 128, 0), RectF(0f, 0f, 100f, 100f),
            fullMask(), 0.5f, size, b,
        )
        assertEquals(1f, at(b, 0, 112, 112), 1e-2f)
        assertEquals(128f / 255f, at(b, 1, 112, 112), 1e-2f)
        assertEquals(0f, at(b, 2, 112, 112), 1e-2f)
    }

    @Test
    fun theBufferIsReusedBetweenCrops() {
        // One buffer per pipeline, not one per instance: a crowded shelf
        // builds dozens of these per frame.
        val b = out()
        CropBuilder.build(solid(100, 100, 10, 10, 10), RectF(0f, 0f, 100f, 100f), fullMask(), 0.5f, size, b)
        val first = System.identityHashCode(b)
        CropBuilder.build(solid(100, 100, 250, 250, 250), RectF(0f, 0f, 100f, 100f), fullMask(), 0.5f, size, b)
        assertEquals(first, System.identityHashCode(b))
        assertTrue(at(b, 0, 112, 112) > 0.9f)
    }
}
