package com.tdi.ubldetect.imaging

import java.nio.ByteBuffer

/**
 * An upright RGB frame, tightly packed, three bytes per pixel.
 *
 * `width` and `height` are post-rotation, so they are what everything
 * downstream should letterbox and draw against.
 */
data class RgbFrame(val buffer: ByteBuffer, val width: Int, val height: Int)

/**
 * The three planes of a YUV_420_888 image, with their strides.
 *
 * Strides are carried explicitly rather than assumed. Camera planes are
 * commonly padded (a 100 pixel row in a 128 byte stride) and chroma is
 * commonly interleaved (pixelStride 2), and assuming either away produces a
 * sheared or falsely coloured image that still looks like a photograph.
 */
class YuvPlanes(
    val y: ByteBuffer,
    val u: ByteBuffer,
    val v: ByteBuffer,
    val width: Int,
    val height: Int,
    val yRowStride: Int,
    val uvRowStride: Int,
    val uvPixelStride: Int,
)

/**
 * Converts a camera frame to upright, tightly packed RGB in one pass.
 *
 * Rotation happens here and nowhere else. The camera hands back frames in
 * sensor orientation, which on a portrait phone is 90 degrees from what the
 * person sees. Correcting it in one place means nothing downstream has to
 * know about orientation at all: the letterbox, the decoders, the crop
 * builder and the overlay all work in upright coordinates. Correcting it in
 * two places means correcting it twice.
 *
 * The rotation is folded into the output index rather than done as a second
 * pass, so there is no intermediate bitmap and no second traversal.
 */
object YuvToRgb {

    /**
     * @param out reused across frames. At 30 frames a second, allocating a
     *   megabyte-plus buffer per frame is pure garbage collector churn, so the
     *   caller owns one buffer per resolution and passes it back in.
     * @return a frame whose width and height are already rotated
     */
    fun convert(planes: YuvPlanes, rotationDegrees: Int, out: ByteBuffer): RgbFrame {
        val w = planes.width
        val h = planes.height
        val rotation = ((rotationDegrees % 360) + 360) % 360
        val swap = rotation == 90 || rotation == 270
        val outW = if (swap) h else w
        val outH = if (swap) w else h

        out.clear()
        out.limit(outW * outH * 3)

        for (row in 0 until h) {
            val yRow = row * planes.yRowStride
            val uvRow = (row / 2) * planes.uvRowStride
            for (col in 0 until w) {
                val yv = planes.y.get(yRow + col).toInt() and 0xFF
                val uvCol = (col / 2) * planes.uvPixelStride
                val uv = planes.u.get(uvRow + uvCol).toInt() and 0xFF
                val vv = planes.v.get(uvRow + uvCol).toInt() and 0xFF

                // BT.601. With U and V at 128 this is the identity on Y, which
                // is what makes the greyscale geometry tests exact.
                val d = uv - 128
                val e = vv - 128
                val r = clamp(yv + 1.402f * e)
                val g = clamp(yv - 0.344136f * d - 0.714136f * e)
                val b = clamp(yv + 1.772f * d)

                // Destination for this source pixel after rotation.
                val dx: Int
                val dy: Int
                when (rotation) {
                    90 -> { dx = outW - 1 - row; dy = col }
                    180 -> { dx = outW - 1 - col; dy = outH - 1 - row }
                    270 -> { dx = row; dy = outH - 1 - col }
                    else -> { dx = col; dy = row }
                }

                val o = (dy * outW + dx) * 3
                out.put(o, r)
                out.put(o + 1, g)
                out.put(o + 2, b)
            }
        }

        out.position(0)
        return RgbFrame(out, outW, outH)
    }

    private fun clamp(v: Float): Byte {
        val i = (v + 0.5f).toInt()
        return when {
            i < 0 -> 0
            i > 255 -> 255.toByte()
            else -> i.toByte()
        }
    }
}
