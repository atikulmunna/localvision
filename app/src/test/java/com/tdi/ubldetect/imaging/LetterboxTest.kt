package com.tdi.ubldetect.imaging

import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * The model sees a square. The camera does not.
 *
 * Every frame is scaled by the smaller ratio and centred in an N by N square,
 * so the model's coordinates are in that square's pixels and have to be mapped
 * back before anything can be drawn over the preview. Measured and recorded in
 * docs/model-contract.md: the exported POSM model returns boxes in letterboxed
 * input pixels, not normalized.
 */
class LetterboxTest {

    @Test
    fun landscapeSourcePadsTopAndBottom() {
        val t = Letterbox.compute(1920, 1080, 640)
        assertEquals(640f / 1920f, t.scale, 1e-6f)
        assertEquals(0f, t.padX, 1e-6f)
        assertEquals((640f - 1080f * t.scale) / 2f, t.padY, 1e-6f)
    }

    @Test
    fun portraitSourcePadsLeftAndRight() {
        val t = Letterbox.compute(1080, 1920, 640)
        assertEquals(640f / 1920f, t.scale, 1e-6f)
        assertEquals(0f, t.padY, 1e-6f)
        assertEquals((640f - 1080f * t.scale) / 2f, t.padX, 1e-6f)
    }

    @Test
    fun roundTripIsIdentity() {
        val t = Letterbox.compute(1920, 1080, 640)
        val p = Letterbox.toSource(320f, 320f, t)
        assertEquals(960f, p.x, 1e-3f)
        assertEquals(540f, p.y, 1e-3f)
    }

    @Test
    fun modelCornerMapsToSourceCorner() {
        // The padded edge maps to the source edge. Anything above it maps to a
        // negative coordinate, which the caller clamps; it is not silently wrong.
        val t = Letterbox.compute(1920, 1080, 640)
        assertEquals(0f, Letterbox.toSource(0f, t.padY, t).y, 1e-3f)
    }

    @Test
    fun squareSourceNeedsNoPadding() {
        val t = Letterbox.compute(1000, 1000, 640)
        assertEquals(0.64f, t.scale, 1e-6f)
        assertEquals(0f, t.padX, 1e-6f)
        assertEquals(0f, t.padY, 1e-6f)
    }

    @Test
    fun rectToSourceMapsAWholeBox() {
        val t = Letterbox.compute(1920, 1080, 640)
        val r = Letterbox.rectToSource(RectF(160f, t.padY, 480f, t.padY + 180f), t)
        assertEquals(480f, r.left, 1e-2f)
        assertEquals(0f, r.top, 1e-2f)
        assertEquals(1440f, r.right, 1e-2f)
        assertEquals(540f, r.bottom, 1e-2f)
    }

    @Test
    fun reproducesTheMeasuredContractBox() {
        // The real check. docs/model-contract.md records one exported-model row
        // on a 1919x1080 photo at 640: raw (261.764, 245.272, 641.963, 294.087)
        // un-letterboxes to roughly (784.9, 315.7, 1924.9, 462.0), which agreed
        // with the .pt reference to 1 or 2 pixels. If this drifts, the decoder's
        // boxes drift with it.
        val t = Letterbox.compute(1919, 1080, 640)
        val r = Letterbox.rectToSource(RectF(261.764f, 245.272f, 641.963f, 294.087f), t)
        assertEquals(784.9f, r.left, 0.5f)
        assertEquals(315.7f, r.top, 0.5f)
        assertEquals(1924.9f, r.right, 0.5f)
        assertEquals(462.0f, r.bottom, 0.5f)
    }

    @Test
    fun rectHelpersReportCentreAndSize() {
        val r = RectF(10f, 20f, 30f, 60f)
        assertEquals(20f, r.centerX(), 1e-6f)
        assertEquals(40f, r.centerY(), 1e-6f)
        assertEquals(20f, r.width(), 1e-6f)
        assertEquals(40f, r.height(), 1e-6f)
    }
}
