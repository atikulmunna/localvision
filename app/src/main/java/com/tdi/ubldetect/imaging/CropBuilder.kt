package com.tdi.ubldetect.imaging

import com.tdi.ubldetect.infer.MaskGrid
import java.nio.ByteBuffer
import kotlin.math.floor

/**
 * Builds the masked crop that QPDS-cls was trained on.
 *
 * Mirrors `_detect_products_two_stage` in UBL-standalone's
 * `core/detection.py`, step for step, because the classifier learned on crops
 * made exactly this way. An unmasked or differently padded crop is
 * off-distribution, and the failure is quiet: it returns confident answers
 * that happen to be wrong.
 *
 * The server's sequence is: resize the mask to full image size with nearest
 * neighbour, crop the integer-truncated box, set everything outside the mask
 * to black, then letterbox onto a 384 black canvas, which the classifier then
 * resizes to 224. Since that canvas is already square, letterboxing straight
 * to 224 is geometrically identical and resamples once instead of twice.
 *
 * Two things differ from [InputTensor] and both are deliberate:
 *
 * - **Padding is black**, where the detector pads 114 grey.
 * - **Background inside the box is removed**, so a neighbouring product
 *   leaning into the box cannot vote on the label.
 */
object CropBuilder {

    /**
     * @param box the instance box in source-frame coordinates
     * @param mask that instance's mask, which knows which part of the frame
     *   it covers
     * @param maskThreshold cells at or above this count as product
     * @param out reused between instances; NCHW float, `outSize^2 * 3 * 4` bytes
     * @return false for a crop with no area, which the caller skips, as the
     *   server does
     */
    fun build(
        frame: RgbFrame,
        box: RectF,
        mask: MaskGrid,
        maskThreshold: Float,
        outSize: Int,
        out: ByteBuffer,
    ): Boolean {
        // Integer truncation then clamping, matching the server's
        // `map(int, box)` followed by a numpy slice.
        val x0 = box.left.toInt().coerceIn(0, frame.width)
        val y0 = box.top.toInt().coerceIn(0, frame.height)
        val x1 = box.right.toInt().coerceIn(0, frame.width)
        val y1 = box.bottom.toInt().coerceIn(0, frame.height)

        val cropW = x1 - x0
        val cropH = y1 - y0
        if (cropW <= 0 || cropH <= 0) return false

        val plane = outSize * outSize
        out.clear()
        out.limit(plane * 3 * 4)
        val floats = out.asFloatBuffer()

        // Letterbox the crop into the square, black everywhere else.
        val scale = minOf(outSize.toFloat() / cropW, outSize.toFloat() / cropH)
        val padX = (outSize - cropW * scale) / 2f
        val padY = (outSize - cropH * scale) / 2f

        val src = frame.buffer

        for (oy in 0 until outSize) {
            val cy = (oy + 0.5f - padY) / scale - 0.5f
            val rowInside = cy > -0.5f && cy < cropH - 0.5f
            for (ox in 0 until outSize) {
                val i = oy * outSize + ox
                if (!rowInside) {
                    floats.put(i, 0f)
                    floats.put(plane + i, 0f)
                    floats.put(2 * plane + i, 0f)
                    continue
                }
                val cx = (ox + 0.5f - padX) / scale - 0.5f
                if (cx <= -0.5f || cx >= cropW - 0.5f) {
                    floats.put(i, 0f)
                    floats.put(plane + i, 0f)
                    floats.put(2 * plane + i, 0f)
                    continue
                }

                val sx = x0 + cx
                val sy = y0 + cy

                // The mask places itself. It covers whole proto cells, not
                // the box, so indexing it by position within the box put it
                // in the wrong place.
                if (mask.atSource(sx, sy) < maskThreshold) {
                    floats.put(i, 0f)
                    floats.put(plane + i, 0f)
                    floats.put(2 * plane + i, 0f)
                    continue
                }

                sampleBilinear(src, frame.width, frame.height, sx, sy, floats, i, plane)
            }
        }

        out.position(0)
        return true
    }

    private fun sampleBilinear(
        src: ByteBuffer,
        srcW: Int,
        srcH: Int,
        sx: Float,
        sy: Float,
        dst: java.nio.FloatBuffer,
        index: Int,
        plane: Int,
    ) {
        val x0 = clampFloor(sx, srcW)
        val y0 = clampFloor(sy, srcH)
        val x1 = if (x0 + 1 < srcW) x0 + 1 else x0
        val y1 = if (y0 + 1 < srcH) y0 + 1 else y0
        val fx = (sx - x0).coerceIn(0f, 1f)
        val fy = (sy - y0).coerceIn(0f, 1f)

        val i00 = (y0 * srcW + x0) * 3
        val i10 = (y0 * srcW + x1) * 3
        val i01 = (y1 * srcW + x0) * 3
        val i11 = (y1 * srcW + x1) * 3

        for (c in 0 until 3) {
            val v00 = (src.get(i00 + c).toInt() and 0xFF).toFloat()
            val v10 = (src.get(i10 + c).toInt() and 0xFF).toFloat()
            val v01 = (src.get(i01 + c).toInt() and 0xFF).toFloat()
            val v11 = (src.get(i11 + c).toInt() and 0xFF).toFloat()
            val top = v00 + (v10 - v00) * fx
            val bottom = v01 + (v11 - v01) * fx
            dst.put(c * plane + index, (top + (bottom - top) * fy) / 255f)
        }
    }

    private fun clampFloor(v: Float, limit: Int): Int {
        val i = floor(v).toInt()
        return when {
            i < 0 -> 0
            i > limit - 1 -> limit - 1
            else -> i
        }
    }
}
