package com.tdi.ubldetect.imaging

import java.nio.ByteBuffer
import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * Review Focus 1: rotation is applied here and nowhere else.
 *
 * The camera delivers frames in sensor orientation, which on a portrait phone
 * is 90 degrees away from what the person sees. If that correction is applied
 * in more than one place it gets applied twice; if it is applied in none, every
 * box is drawn rotated. So it happens once, during the conversion to RGB, and
 * everything downstream works in upright coordinates.
 *
 * These are greyscale cases on purpose: with U and V held at 128 the colour
 * conversion is the identity on Y, so a pixel's value is exactly predictable
 * and the assertions are about geometry rather than colour arithmetic.
 */
class YuvToRgbTest {

    /**
     * A YUV_420_888 frame with a chosen 8-bit grey at one pixel.
     *
     * `yRowStride` is deliberately settable: camera planes are usually padded,
     * and assuming they are tightly packed is the classic way to produce a
     * sheared image.
     */
    private fun greyPlanes(
        width: Int,
        height: Int,
        markX: Int = -1,
        markY: Int = -1,
        markValue: Int = 255,
        background: Int = 16,
        yRowStride: Int = width,
        uvPixelStride: Int = 1,
    ): YuvPlanes {
        val y = ByteBuffer.allocate(yRowStride * height)
        for (row in 0 until height) {
            for (col in 0 until width) {
                val v = if (row == markY && col == markX) markValue else background
                y.put(row * yRowStride + col, v.toByte())
            }
        }
        val cw = (width + 1) / 2
        val ch = (height + 1) / 2
        val uvRowStride = cw * uvPixelStride
        val u = ByteBuffer.allocate(uvRowStride * ch)
        val v = ByteBuffer.allocate(uvRowStride * ch)
        for (i in 0 until uvRowStride * ch) {
            u.put(i, 128.toByte())
            v.put(i, 128.toByte())
        }
        return YuvPlanes(y, u, v, width, height, yRowStride, uvRowStride, uvPixelStride)
    }

    private fun out(width: Int, height: Int) = ByteBuffer.allocate(width * height * 3)

    private fun red(f: RgbFrame, x: Int, y: Int): Int =
        f.buffer.get((y * f.width + x) * 3).toInt() and 0xFF

    @Test
    fun rotation0PreservesDimensions() {
        val f = YuvToRgb.convert(greyPlanes(640, 480), 0, out(640, 480))
        assertEquals(640, f.width)
        assertEquals(480, f.height)
    }

    @Test
    fun rotation90SwapsWidthAndHeight() {
        val f = YuvToRgb.convert(greyPlanes(640, 480), 90, out(640, 480))
        assertEquals(480, f.width)
        assertEquals(640, f.height)
    }

    @Test
    fun rotation270SwapsWidthAndHeight() {
        val f = YuvToRgb.convert(greyPlanes(640, 480), 270, out(640, 480))
        assertEquals(480, f.width)
        assertEquals(640, f.height)
    }

    @Test
    fun rotation180PreservesDimensions() {
        val f = YuvToRgb.convert(greyPlanes(640, 480), 180, out(640, 480))
        assertEquals(640, f.width)
        assertEquals(480, f.height)
    }

    @Test
    fun rotation90MovesTopLeftPixelToTopRight() {
        // The whole of Review Focus 1 in one assertion. Rotating a frame 90
        // degrees clockwise sends (0,0) to the top right corner. Getting the
        // direction backwards still produces a plausible image, just mirrored.
        val f = YuvToRgb.convert(greyPlanes(64, 48, markX = 0, markY = 0), 90, out(64, 48))
        assertEquals(255, red(f, f.width - 1, 0))
        assertEquals(16, red(f, 0, 0))
    }

    @Test
    fun rotation270MovesTopLeftPixelToBottomLeft() {
        val f = YuvToRgb.convert(greyPlanes(64, 48, markX = 0, markY = 0), 270, out(64, 48))
        assertEquals(255, red(f, 0, f.height - 1))
    }

    @Test
    fun rotation180MovesTopLeftPixelToBottomRight() {
        val f = YuvToRgb.convert(greyPlanes(64, 48, markX = 0, markY = 0), 180, out(64, 48))
        assertEquals(255, red(f, f.width - 1, f.height - 1))
    }

    @Test
    fun rotation0LeavesTheMarkWhereItWas() {
        val f = YuvToRgb.convert(greyPlanes(64, 48, markX = 3, markY = 5), 0, out(64, 48))
        assertEquals(255, red(f, 3, 5))
    }

    @Test
    fun paddedRowStrideIsHonouredRatherThanAssumedAway() {
        // A plane padded to 128 bytes for a 100 pixel row. Reading it as if it
        // were tightly packed shears the image progressively down the frame.
        val f = YuvToRgb.convert(
            greyPlanes(100, 40, markX = 7, markY = 11, yRowStride = 128),
            0,
            out(100, 40),
        )
        assertEquals(255, red(f, 7, 11))
        assertEquals(16, red(f, 8, 11))
    }

    @Test
    fun semiPlanarChromaPixelStrideIsHonoured() {
        // Most Android cameras hand back interleaved chroma, pixelStride 2.
        val f = YuvToRgb.convert(
            greyPlanes(64, 48, markX = 2, markY = 2, uvPixelStride = 2),
            0,
            out(64, 48),
        )
        assertEquals(255, red(f, 2, 2))
    }

    @Test
    fun outputIsTightlyPackedRgbWithNoRowStridePadding() {
        // Odd dimensions, because they expose stride assumptions in the output
        // as well as the input, and the chroma planes round up.
        val f = YuvToRgb.convert(greyPlanes(641, 481), 0, out(641, 481))
        assertEquals(641 * 481 * 3, f.buffer.limit())
    }

    @Test
    fun greyInputProducesEqualRgbChannels() {
        // U and V at 128 mean no colour, so the three channels must agree.
        // If they do not, the conversion coefficients are wrong.
        val f = YuvToRgb.convert(greyPlanes(16, 16, markX = 4, markY = 4, markValue = 200), 0, out(16, 16))
        val base = (4 * f.width + 4) * 3
        val r = f.buffer.get(base).toInt() and 0xFF
        val g = f.buffer.get(base + 1).toInt() and 0xFF
        val b = f.buffer.get(base + 2).toInt() and 0xFF
        assertEquals(200, r)
        assertEquals(r, g)
        assertEquals(r, b)
    }

    @Test
    fun theOutputBufferIsReusedRatherThanReallocated() {
        // One allocation per resolution, not one per frame: at 30 frames a
        // second a 1.4 MB allocation per frame is pure garbage collector churn.
        val buffer = out(64, 48)
        val first = YuvToRgb.convert(greyPlanes(64, 48), 0, buffer)
        val second = YuvToRgb.convert(greyPlanes(64, 48), 0, buffer)
        assertEquals(first.buffer, second.buffer)
    }
}
