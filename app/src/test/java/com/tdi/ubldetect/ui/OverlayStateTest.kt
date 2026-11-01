package com.tdi.ubldetect.ui

import com.tdi.ubldetect.imaging.LetterboxTransform
import com.tdi.ubldetect.imaging.RectF
import com.tdi.ubldetect.model.Detection
import com.tdi.ubldetect.model.FrameResult
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Review Focus 3: what stays on screen when nothing is detected.
 *
 * The detector runs slower than the preview and drops frames while busy, so
 * the overlay is always showing a result from some earlier frame. That makes
 * "when do boxes disappear" a real decision rather than an implementation
 * detail. Leaving the last boxes up through an empty result is the failure
 * worth guarding: the person points at a bare shelf and still sees labels.
 *
 * Separated from OverlayView so it is a JVM test with no View, no looper and
 * no device.
 */
class OverlayStateTest {

    private val transform = LetterboxTransform(1f, 0f, 0f, 640)

    private fun resultWith(n: Int, timestampNs: Long = 1L) = FrameResult(
        detections = List(n) { Detection(RectF(0f, 0f, 10f, 10f), it, 0.9f) },
        frameTimestampNs = timestampNs,
        transform = transform,
        timingsMs = emptyMap(),
    )

    private fun resultAt(timestampNs: Long) = resultWith(1, timestampNs)

    @Test
    fun emptyResultClearsPreviousBoxes() {
        val s = OverlayState()
        s.submit(resultWith(2, 100))
        s.submit(resultWith(0, 200))
        assertEquals(0, s.drawn.size)
    }

    @Test
    fun nullResultWhileFrozenHoldsTheLastBoxes() {
        // Freezing is for reading labels carefully, so a late result arriving
        // after the freeze must not wipe what the person is looking at.
        val s = OverlayState()
        s.submit(resultWith(2, 100))
        s.freeze()
        s.submit(null)
        assertEquals(2, s.drawn.size)
    }

    @Test
    fun resultsAreIgnoredWhileFrozen() {
        val s = OverlayState()
        s.submit(resultWith(2, 100))
        s.freeze()
        s.submit(resultWith(5, 200))
        assertEquals(2, s.drawn.size)
    }

    @Test
    fun unfreezingLetsResultsThroughAgain() {
        val s = OverlayState()
        s.submit(resultWith(2, 100))
        s.freeze()
        s.unfreeze()
        s.submit(resultWith(5, 300))
        assertEquals(5, s.drawn.size)
    }

    @Test
    fun staleResultOlderThanTheCurrentOneIsIgnored() {
        // Frames are decoded off the camera thread, so a slow one can finish
        // after a fast one that came later. Drawing it would jump the overlay
        // backwards in time.
        val s = OverlayState()
        s.submit(resultAt(200))
        s.submit(resultAt(100))
        assertEquals(200L, s.drawnTimestampNs)
    }

    @Test
    fun nullResultWhileLiveClearsTheOverlay() {
        val s = OverlayState()
        s.submit(resultWith(3, 100))
        s.submit(null)
        assertTrue(s.drawn.isEmpty())
    }

    @Test
    fun startsEmpty() {
        assertTrue(OverlayState().drawn.isEmpty())
    }
}
