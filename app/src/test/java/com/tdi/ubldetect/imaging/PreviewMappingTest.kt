package com.tdi.ubldetect.imaging

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Review Focus 2: the preview and the model see the frame differently.
 *
 * The model gets the whole frame letterboxed into a square, padding and all.
 * The preview fills the view by scaling up and cropping whatever does not fit.
 * Those are opposite choices: `min` of the two ratios against `max` of them.
 * Conflating them puts every box on screen at a plausible but wrong offset and
 * scale, which looks like a detection problem rather than a drawing one.
 */
class PreviewMappingTest {

    private companion object {
        /** Tolerance for a coordinate that should be zero. Float arithmetic on
         *  numbers this size leaves residue far below one screen pixel. */
        const val SUB_PX = 1e-3f
    }

    @Test
    fun fillCenterScalesByTheLargerRatioAndCentres() {
        // 1080x1920 source into a 1080x2000 view: scale by height, crop width
        val m = PreviewMapping.fillCenter(1080, 1920, 1080, 2000)
        assertEquals(2000f / 1920f, m.scale, 1e-6f)
        assertTrue("width is cropped, so the x offset is negative", m.offsetX < 0f)
        // SUB_PX, not 1e-6: (2000 - 1920 * (2000f/1920f)) / 2 leaves about
        // 6e-5 of float residue. A ten-thousandth of a pixel is not an offset.
        assertEquals(0f, m.offsetY, SUB_PX)
    }

    @Test
    fun sourceCentreMapsToViewCentre() {
        val m = PreviewMapping.fillCenter(1080, 1920, 1080, 2000)
        val r = PreviewMapping.mapRect(RectF(540f, 960f, 540f, 960f), m)
        assertEquals(540f, r.left, 1e-2f)
        assertEquals(1000f, r.top, 1e-2f)
    }

    @Test
    fun matchingAspectRatioIsAPureScale() {
        val m = PreviewMapping.fillCenter(1080, 1920, 540, 960)
        assertEquals(0.5f, m.scale, 1e-6f)
        assertEquals(0f, m.offsetX, 1e-6f)
        assertEquals(0f, m.offsetY, 1e-6f)
    }

    @Test
    fun boxFromLetterboxThroughToViewLandsWhereExpected() {
        // The composition that actually runs per frame: model square -> source
        // frame -> view. The centre of the model square is the centre of the
        // frame, and must land at the centre of the preview.
        val t = Letterbox.compute(1080, 1920, 640)
        val m = PreviewMapping.fillCenter(1080, 1920, 1080, 2000)
        val inView = PreviewMapping.mapRect(
            Letterbox.rectToSource(RectF(320f, 320f, 320f, 320f), t),
            m,
        )
        assertEquals(540f, inView.left, 1f)
        assertEquals(1000f, inView.top, 1f)
    }

    @Test
    fun fillCenterCropsHeightWhenTheViewIsWiderThanTheSource() {
        // the other crop direction, which a portrait-only test would miss
        val m = PreviewMapping.fillCenter(1920, 1080, 2000, 1000)
        assertEquals(2000f / 1920f, m.scale, 1e-6f)
        assertEquals(0f, m.offsetX, SUB_PX)
        assertTrue("height is cropped, so the y offset is negative", m.offsetY < 0f)
    }

    @Test
    fun usingTheLetterboxRuleInsteadWouldMisplaceTheBox() {
        // Pins the distinction itself rather than one of its consequences. If
        // someone "simplifies" PreviewMapping to reuse Letterbox's min() rule,
        // this fails instead of the boxes quietly drifting on screen.
        val fill = PreviewMapping.fillCenter(1080, 1920, 1080, 2000)
        val letterboxRule = minOf(1080f / 1080f, 2000f / 1920f)
        assertTrue(
            "fill-centre must scale by max of the ratios, not min",
            fill.scale > letterboxRule,
        )
    }
}
