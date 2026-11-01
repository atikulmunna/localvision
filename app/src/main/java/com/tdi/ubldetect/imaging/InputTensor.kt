package com.tdi.ubldetect.imaging

import java.nio.ByteBuffer

/**
 * Packs an upright RGB frame into the model's input tensor.
 *
 * The format is not a choice. From docs/model-contract.md, measured from the
 * exported file: `[1, 3, N, N]` float32, **NCHW**, values 0.0 to 1.0, RGB,
 * letterboxed with **114 grey** padding. Each of those is a silent accuracy
 * loss if wrong rather than a crash, which is why they are pinned by tests
 * instead of trusted to a comment.
 *
 * The geometry is [Letterbox]'s, deliberately shared: the same transform
 * that put the frame in is used to take the boxes back out, so the two can
 * never drift apart.
 */
object InputTensor {

    /** Ultralytics' letterbox fill, and therefore what the model trained on. */
    const val PAD_VALUE = 114

    fun transformFor(srcWidth: Int, srcHeight: Int, inputSize: Int): LetterboxTransform =
        Letterbox.compute(srcWidth, srcHeight, inputSize)

    /**
     * Writes one frame into [out] and returns the transform used.
     *
     * Resampling is bilinear, matching the reference preprocessing. Nearest
     * neighbour is cheaper and visibly similar, but it is not what the model
     * was fed during training.
     *
     * @param out reused between frames; the caller owns one per resolution
     */
    fun letterboxToNchw(frame: RgbFrame, inputSize: Int, out: ByteBuffer): LetterboxTransform {
        val t = transformFor(frame.width, frame.height, inputSize)
        val n = inputSize
        val plane = n * n
        val pad = PAD_VALUE / 255f

        out.clear()
        out.limit(plane * 3 * 4)
        val floats = out.asFloatBuffer()

        val src = frame.buffer
        val sw = frame.width
        val sh = frame.height

        for (y in 0 until n) {
            // Model pixel back to source pixel, the inverse of the letterbox.
            val sy = (y + 0.5f - t.padY) / t.scale - 0.5f
            val rowInside = sy > -0.5f && sy < sh - 0.5f

            for (x in 0 until n) {
                val i = y * n + x
                if (!rowInside) {
                    floats.put(i, pad)
                    floats.put(plane + i, pad)
                    floats.put(2 * plane + i, pad)
                    continue
                }
                val sx = (x + 0.5f - t.padX) / t.scale - 0.5f
                if (sx <= -0.5f || sx >= sw - 0.5f) {
                    floats.put(i, pad)
                    floats.put(plane + i, pad)
                    floats.put(2 * plane + i, pad)
                    continue
                }

                val x0 = floorClamp(sx, sw)
                val y0 = floorClamp(sy, sh)
                val x1 = if (x0 + 1 < sw) x0 + 1 else x0
                val y1 = if (y0 + 1 < sh) y0 + 1 else y0
                val fx = (sx - x0).coerceIn(0f, 1f)
                val fy = (sy - y0).coerceIn(0f, 1f)

                val i00 = (y0 * sw + x0) * 3
                val i10 = (y0 * sw + x1) * 3
                val i01 = (y1 * sw + x0) * 3
                val i11 = (y1 * sw + x1) * 3

                for (c in 0 until 3) {
                    val v00 = (src.get(i00 + c).toInt() and 0xFF).toFloat()
                    val v10 = (src.get(i10 + c).toInt() and 0xFF).toFloat()
                    val v01 = (src.get(i01 + c).toInt() and 0xFF).toFloat()
                    val v11 = (src.get(i11 + c).toInt() and 0xFF).toFloat()
                    val top = v00 + (v10 - v00) * fx
                    val bottom = v01 + (v11 - v01) * fx
                    floats.put(c * plane + i, (top + (bottom - top) * fy) / 255f)
                }
            }
        }

        out.position(0)
        return t
    }

    private fun floorClamp(v: Float, limit: Int): Int {
        val i = kotlin.math.floor(v).toInt()
        return when {
            i < 0 -> 0
            i > limit - 1 -> limit - 1
            else -> i
        }
    }
}
