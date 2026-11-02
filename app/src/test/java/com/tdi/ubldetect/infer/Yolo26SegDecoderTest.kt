package com.tdi.ubldetect.infer

import com.tdi.ubldetect.imaging.Letterbox
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Decoding the segmentation head.
 *
 * Contract measured from the exported file and recorded in
 * docs/model-contract.md: rows are `[1, 300, 38]`, being 4 box values, a
 * score, a class id that is always 0 because the model has one class, and 32
 * mask coefficients. Protos are `[1, 32, P, P]` where **P is N/4**, so 160 at
 * imgsz 640. The plan guessed 64; measuring is why these tests use 160.
 *
 * A mask is `sigmoid(coeffs . protos)`. On a real shelf photo that averaged
 * 0.948 inside the top instance's box against 0.374 across the whole grid,
 * which is the separation the crop builder relies on.
 */
class Yolo26SegDecoderTest {

    private val segShape = intArrayOf(1, 300, 38)
    private val protoShape = intArrayOf(1, 32, 160, 160)
    private val t = Letterbox.compute(1920, 1080, 640)

    /** Builds a [1,300,38] buffer from rows of (x1,y1,x2,y2,score,cls,coeffs...). */
    private fun segRows(vararg scores: Float): FloatArray {
        val raw = FloatArray(300 * 38)
        scores.forEachIndexed { i, s ->
            val o = i * 38
            raw[o] = 100f; raw[o + 1] = 100f; raw[o + 2] = 200f; raw[o + 3] = 200f
            raw[o + 4] = s
            raw[o + 5] = 0f
            for (c in 0 until 32) raw[o + 6 + c] = 0.1f * (c + 1)
        }
        return raw
    }

    private fun rowsWithCoeffs(score: Float, coeffs: FloatArray, box: FloatArray): FloatArray {
        val raw = FloatArray(300 * 38)
        box.copyInto(raw, 0)
        raw[4] = score
        raw[5] = 0f
        coeffs.copyInto(raw, 6)
        return raw
    }

    /** Protos whose first channel is all ones and the rest all zeros. */
    private fun unitProtos(): FloatArray {
        val p = FloatArray(32 * 160 * 160)
        for (i in 0 until 160 * 160) p[i] = 1f
        return p
    }

    @Test
    fun splitsBoxScoreAndThirtyTwoCoefficients() {
        val out = Yolo26SegDecoder.decode(segRows(0.9f), segShape, 0.20f, 40, t)
        assertEquals(1, out.size)
        assertEquals(32, out[0].coeffs.size)
        assertEquals(0.1f, out[0].coeffs[0], 1e-5f)
        assertEquals(3.2f, out[0].coeffs[31], 1e-5f)
    }

    @Test
    fun honoursTheSegConfidenceThreshold() {
        // 0.20 is the server's fixed_shelf seg_conf, copied not re-tuned
        val out = Yolo26SegDecoder.decode(segRows(0.50f, 0.20f, 0.19f), segShape, 0.20f, 40, t)
        assertEquals(2, out.size)
    }

    @Test
    fun capLimitsInstancesToTheHighestScoring() {
        // A crowded shelf must not stall the loop: the cap keeps the most
        // confident instances rather than whichever came first.
        val out = Yolo26SegDecoder.decode(segRows(0.9f, 0.8f, 0.7f), segShape, 0.20f, 2, t)
        assertEquals(listOf(0.9f, 0.8f), out.map { it.score })
    }

    @Test
    fun capOfZeroYieldsNothingRatherThanEverything() {
        assertTrue(Yolo26SegDecoder.decode(segRows(0.9f), segShape, 0.20f, 0, t).isEmpty())
    }

    @Test
    fun boxesComeBackInSourceCoordinates() {
        val tt = Letterbox.compute(1080, 1920, 640)
        val out = Yolo26SegDecoder.decode(
            rowsWithCoeffs(0.9f, FloatArray(32), floatArrayOf(315f, 315f, 325f, 325f)),
            segShape, 0.20f, 40, tt,
        )
        assertEquals(540f, out[0].box.centerX(), 2f)
        assertEquals(960f, out[0].box.centerY(), 2f)
    }

    @Test
    fun maskForCombinesProtosByCoefficientsAndSigmoids() {
        // One proto channel of all ones with coefficient 1.0, the rest zero,
        // so every cell is sigmoid(1.0).
        val coeffs = FloatArray(32).also { it[0] = 1f }
        val inst = Yolo26SegDecoder.decode(
            rowsWithCoeffs(0.9f, coeffs, floatArrayOf(0f, 0f, 640f, 640f)),
            segShape, 0.20f, 40, t,
        )[0]
        val m = Yolo26SegDecoder.maskFor(inst, unitProtos(), protoShape, t)
        assertEquals(0.7310586f, m.at(0, 0), 1e-4f)
    }

    @Test
    fun maskIsComputedOnlyWithinTheInstanceBox() {
        // Cost scales with the box, not the frame. A full-resolution upsample
        // per instance would be the obvious and wasteful way to do this.
        val coeffs = FloatArray(32).also { it[0] = 1f }
        val inst = Yolo26SegDecoder.decode(
            rowsWithCoeffs(0.9f, coeffs, floatArrayOf(0f, 0f, 320f, 160f)),
            segShape, 0.20f, 40, t,
        )[0]
        val m = Yolo26SegDecoder.maskFor(inst, unitProtos(), protoShape, t)
        assertTrue("grid ${m.width}x${m.height} should be smaller than 160", m.width < 160)
        assertEquals(320 / 4, m.width)
        assertEquals(160 / 4, m.height)
    }

    @Test
    fun protoGridIsQuarterOfTheInputNotAFixedSize() {
        // P = N/4: 160 at 640 and 240 at 960. A hardcoded 64, which the plan
        // assumed, would place every mask in the wrong part of the grid.
        val coeffs = FloatArray(32).also { it[0] = 1f }
        val inst = Yolo26SegDecoder.decode(
            rowsWithCoeffs(0.9f, coeffs, floatArrayOf(0f, 0f, 960f, 960f)),
            segShape, 0.20f, 40, Letterbox.compute(1920, 1080, 960),
        )[0]

        val t960 = Letterbox.compute(1920, 1080, 960)
        val m = Yolo26SegDecoder.maskFor(
            inst, FloatArray(32 * 240 * 240) { 1f }, intArrayOf(1, 32, 240, 240), t960,
        )
        assertEquals(240, m.width)
    }

    @Test
    fun theGridContainsTheWholeBoxWhenItDoesNotLandOnCellBoundaries() {
        // Truncating both ends left the box poking out of its own mask. A box
        // from model 10 to 330 spans cells 2.5 to 82.5, so the grid has to
        // run from cell 2 to cell 83: floor one side, ceil the other. Taking
        // floor on both gave cells 2 to 82, and every source pixel in the
        // last 20 model px of the box then indexed past the grid and got the
        // edge cell's value instead of its own.
        val coeffs = FloatArray(32).also { it[0] = 1f }
        val inst = Yolo26SegDecoder.decode(
            rowsWithCoeffs(0.9f, coeffs, floatArrayOf(10f, 10f, 330f, 330f)),
            segShape, 0.20f, 40, t,
        )[0]
        val m = Yolo26SegDecoder.maskFor(inst, unitProtos(), protoShape, t)
        assertEquals(81, m.width)
        assertEquals(81, m.height)
    }

    @Test
    fun aBoxReachingPastTheProtoGridIsClamped() {
        // Boxes can exceed the input square: the measured rows ranged from
        // -32 to 691 at imgsz 640. Indexing the proto grid with that directly
        // would read out of bounds.
        val coeffs = FloatArray(32).also { it[0] = 1f }
        val inst = Yolo26SegDecoder.decode(
            rowsWithCoeffs(0.9f, coeffs, floatArrayOf(-32f, -32f, 691f, 691f)),
            segShape, 0.20f, 40, t,
        )[0]
        val m = Yolo26SegDecoder.maskFor(inst, unitProtos(), protoShape, t)
        assertTrue(m.width in 1..160)
        assertTrue(m.height in 1..160)
    }

    @Test
    fun zeroFilledTailIsNotDecodedAsAnInstance() {
        assertTrue(Yolo26SegDecoder.decode(FloatArray(300 * 38), segShape, 0.20f, 40, t).isEmpty())
    }
}
