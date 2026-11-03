package com.tdi.ubldetect.ui

import com.tdi.ubldetect.infer.DelegateKind
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Which control changes cost an interpreter rebuild.
 *
 * Resolution and delegate are baked into a loaded model: LiteRT fixes the
 * input shape at export, and the delegate is chosen when the interpreter is
 * built. Changing either means tearing down a 57 MB model and loading
 * another. Confidence, the allowlist and freeze are read per frame and cost
 * nothing.
 *
 * Getting this backwards is expensive in both directions: rebuilding on a
 * slider drag would stall the camera on every pixel of movement, and not
 * rebuilding on a resolution change would silently keep running the old
 * model while the UI claims otherwise.
 */
class ControlsStateTest {

    @Test
    fun changingImgszRequiresAModelSwap() {
        assertTrue(ControlsState(imgsz = 960).requiresReload(ControlsState(imgsz = 640)))
    }

    @Test
    fun changingThresholdDoesNotRequireAModelSwap() {
        assertFalse(
            ControlsState(scoreThreshold = 0.30f)
                .requiresReload(ControlsState(scoreThreshold = 0.50f)),
        )
    }

    @Test
    fun changingDelegateRequiresAModelSwap() {
        assertTrue(
            ControlsState(delegate = DelegateKind.GPU)
                .requiresReload(ControlsState(delegate = DelegateKind.CPU_XNNPACK)),
        )
    }

    @Test
    fun togglingTheAllowlistDoesNotRequireAModelSwap() {
        assertFalse(
            ControlsState(useAllowlist = true).requiresReload(ControlsState(useAllowlist = false)),
        )
    }

    @Test
    fun freezingDoesNotRequireAModelSwap() {
        assertFalse(ControlsState(frozen = false).requiresReload(ControlsState(frozen = true)))
    }

    @Test
    fun anUnchangedStateRequiresNothing() {
        assertFalse(ControlsState().requiresReload(ControlsState()))
    }

    @Test
    fun defaultsMatchTheServerThresholds() {
        // Copied from UBL-standalone config.yaml rather than re-tuned, so that
        // any difference from the server traces to export and quantization.
        val s = ControlsState()
        assertEquals(0.30f, s.scoreThreshold, 1e-6f)
        assertEquals(40, s.instanceCap)
    }

    @Test
    fun defaultResolutionIsOneThatIsActuallyShipped() {
        // Review Focus 5: offering a resolution whose model was never exported
        // is how the app ends up showing a missing-asset message on startup.
        assertTrue(ControlsState().imgsz in ControlsState.OFFERED_SIZES)
    }

    @Test
    fun assetNameMatchesTheExporterNamingContract() {
        // tools/export_litert.py writes posm_<imgsz>.tflite; the two sides of
        // that contract are tested separately and must agree.
        assertEquals("models/posm_640.tflite", ControlsState(imgsz = 640).assetFor("posm"))
        assertEquals("models/posm_960.tflite", ControlsState(imgsz = 960).assetFor("posm"))
    }

    @Test
    fun boxesStartVisible() {
        // A detection viewer that opens showing nothing would look broken.
        assertTrue(ControlsState().showBoxes)
    }

    @Test
    fun hidingTheBoxesDoesNotReloadTheModel() {
        // It is a drawing decision. Tearing down a 57 MB interpreter to stop
        // painting rectangles would be absurd, and slow.
        val before = ControlsState()
        assertFalse(before.copy(showBoxes = false).requiresReload(before))
    }
}
