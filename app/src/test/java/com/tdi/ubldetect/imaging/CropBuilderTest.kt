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

    /**
     * A mask over the whole of a 100 by 100 frame at imgsz 640.
     *
     * The grid declares the model-pixel region it covers, because that is
     * what decides where its cells land on the frame. Spanning the frame
     * keeps these tests about masking and padding rather than about geometry;
     * [maskGeometryFollowsTheRegionTheGridCoversNotTheBox] covers the case
     * where the two differ.
     */
    private fun maskOverWholeFrame(values: FloatArray, w: Int, h: Int) =
        MaskGrid(values, w, h, RectF(0f, 0f, 640f, 640f), Letterbox.compute(100, 100, 640))

    private fun fullMask(w: Int = 8, h: Int = 8) =
        maskOverWholeFrame(FloatArray(w * h) { 1f }, w, h)

    private fun uniformMask(v: Float, w: Int = 8, h: Int = 8) =
        maskOverWholeFrame(FloatArray(w * h) { v }, w, h)

    /** Mask covering only the left half of the frame. */
    private fun leftHalfMask(w: Int = 8, h: Int = 8) =
        maskOverWholeFrame(FloatArray(w * h) { if (it % w < w / 2) 1f else 0f }, w, h)

    /**
     * The mask is placed by the region it covers, not by the box.
     *
     * This is the bug this geometry was changed for. The decoder builds a
     * mask over whole proto cells, so the grid covers a region that is
     * truncated to the cell below the box and clamped to the model square.
     * Indexing it as if it spanned the box exactly put the mask in the wrong
     * place: off by up to a cell normally, and for a box hanging off the edge
     * of the frame, offset and stretched by tens of percent. Wrong pixels get
     * blacked out and the classifier is handed an off-distribution crop,
     * which this model answers confidently and wrongly.
     *
     * Here a 2 by 1 grid covers the left half of the model square, so its
     * first cell is product and its second is not. The box is the right half
     * of the frame, which lies entirely inside the grid's second cell, so the
     * whole crop must be blacked out. Stretching the grid over the box
     * instead would make half of it product.
     */
    @Test
    fun maskGeometryFollowsTheRegionTheGridCoversNotTheBox() {
        val t = Letterbox.compute(100, 100, 640)
        val mask = MaskGrid(
            floatArrayOf(1f, 0f), 2, 1,
            RectF(0f, 0f, 640f, 640f), t,
        )
        val b = out()
        assertTrue(
            CropBuilder.build(
                solid(100, 100, 255, 255, 255), RectF(50f, 0f, 100f, 100f),
                mask, 0.5f, size, b,
            ),
        )
        // Every pixel of the right-hand box sits in the grid's empty cell.
        assertEquals(0f, at(b, 0, 112, 60), 1e-3f)
        assertEquals(0f, at(b, 0, 60, 112), 1e-3f)
        assertEquals(0f, at(b, 0, 180, 180), 1e-3f)
    }

    /**
     * A mask cell outside the grid is clamped, not read out of bounds.
     *
     * Boxes legitimately hang off the model square: the measured rows ran
     * from -32 to 691 at imgsz 640. Those source pixels are outside the
     * frame and are already excluded from the crop, but the index has to stay
     * in range on the way there.
     */
    @Test
    fun aSourcePixelOutsideTheGridsRegionIsClampedIntoIt() {
        val t = Letterbox.compute(100, 100, 640)
        val mask = MaskGrid(FloatArray(4) { 1f }, 2, 2, RectF(0f, 0f, 320f, 320f), t)
        val b = out()
        assertTrue(
            CropBuilder.build(
                solid(100, 100, 255, 255, 255), RectF(-20f, -20f, 120f, 120f),
                mask, 0.5f, size, b,
            ),
        )
        // The grid covers only the top left quarter of the frame, so every
        // pixel beyond it clamps onto an edge cell, which here is product.
        assertEquals(1f, at(b, 0, 200, 200), 1e-3f)
    }

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
