package com.tdi.ubldetect.imaging

import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * Packing a frame into the model's input tensor.
 *
 * Four properties come straight from docs/model-contract.md, measured from
 * the exported file. Each is a silent accuracy loss if wrong, not a crash:
 *
 * - **NCHW**, all red then all green then all blue, because the LiteRT
 *   exporter traces PyTorch and keeps its layout. Feeding interleaved RGB
 *   produces an image the model has never seen and detections that look
 *   randomly poor.
 * - **0.0 to 1.0**, so bytes are divided by 255.
 * - **114 grey padding**, which is what Ultralytics letterboxes with and
 *   therefore what the model was trained on. Black padding is a different
 *   picture.
 * - Geometry identical to [Letterbox], so boxes decode back correctly.
 */
class InputTensorTest {

    private fun solid(width: Int, height: Int, r: Int, g: Int, b: Int): RgbFrame {
        val buf = ByteBuffer.allocate(width * height * 3)
        for (i in 0 until width * height) {
            buf.put(i * 3, r.toByte())
            buf.put(i * 3 + 1, g.toByte())
            buf.put(i * 3 + 2, b.toByte())
        }
        return RgbFrame(buf, width, height)
    }

    private fun out(n: Int) =
        ByteBuffer.allocateDirect(n * n * 3 * 4).order(ByteOrder.nativeOrder())

    private fun at(buf: ByteBuffer, n: Int, channel: Int, x: Int, y: Int): Float =
        buf.asFloatBuffer().get(channel * n * n + y * n + x)

    @Test
    fun fillsExactlyOneTensorWorthOfFloats() {
        val b = out(64)
        InputTensor.letterboxToNchw(solid(64, 64, 10, 20, 30), 64, b)
        assertEquals(64 * 64 * 3 * 4, b.limit())
    }

    @Test
    fun layoutIsChannelPlanesNotInterleavedPixels() {
        val b = out(32)
        InputTensor.letterboxToNchw(solid(32, 32, 255, 0, 0), 32, b)
        assertEquals(1f, at(b, 32, 0, 5, 5), 1e-3f)   // red plane
        assertEquals(0f, at(b, 32, 1, 5, 5), 1e-3f)   // green plane
        assertEquals(0f, at(b, 32, 2, 5, 5), 1e-3f)   // blue plane
    }

    @Test
    fun valuesAreNormalisedToZeroOne() {
        val b = out(32)
        InputTensor.letterboxToNchw(solid(32, 32, 128, 128, 128), 32, b)
        assertEquals(128f / 255f, at(b, 32, 0, 1, 1), 1e-4f)
    }

    @Test
    fun paddingIsOneHundredAndFourteenGreyNotBlack() {
        // A 2:1 frame into a square leaves a quarter of the height padded top
        // and bottom. Those rows must be the training pad colour.
        val b = out(64)
        InputTensor.letterboxToNchw(solid(64, 32, 255, 255, 255), 64, b)
        val expected = 114f / 255f
        assertEquals(expected, at(b, 64, 0, 32, 2), 1e-4f)
        assertEquals(expected, at(b, 64, 1, 32, 2), 1e-4f)
        assertEquals(expected, at(b, 64, 2, 32, 61), 1e-4f)
    }

    @Test
    fun imageContentLandsInsideThePadding() {
        val b = out(64)
        InputTensor.letterboxToNchw(solid(64, 32, 255, 255, 255), 64, b)
        assertEquals(1f, at(b, 64, 0, 32, 32), 1e-3f)
    }

    @Test
    fun squareFrameIsNotPaddedAtAll() {
        val b = out(32)
        InputTensor.letterboxToNchw(solid(16, 16, 200, 200, 200), 32, b)
        val expected = 200f / 255f
        assertEquals(expected, at(b, 32, 0, 0, 0), 1e-3f)
        assertEquals(expected, at(b, 32, 0, 31, 31), 1e-3f)
    }

    @Test
    fun geometryMatchesTheLetterboxUsedToDecodeBoxesBack() {
        // If these two ever disagree, every box is drawn at a believable but
        // wrong place, which reads as a model problem.
        val t = Letterbox.compute(1920, 1080, 640)
        assertEquals(t, InputTensor.transformFor(1920, 1080, 640))
    }

    @Test
    fun reusesTheCallersBufferAcrossFrames() {
        val b = out(32)
        InputTensor.letterboxToNchw(solid(32, 32, 10, 10, 10), 32, b)
        val firstIdentity = System.identityHashCode(b)
        InputTensor.letterboxToNchw(solid(32, 32, 20, 20, 20), 32, b)
        assertEquals(firstIdentity, System.identityHashCode(b))
        assertEquals(20f / 255f, at(b, 32, 0, 1, 1), 1e-4f)
    }
}
