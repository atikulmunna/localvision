package com.tdi.ubldetect.ui

import com.tdi.ubldetect.infer.DelegateKind

/**
 * Everything the on-screen controls can change.
 *
 * Immutable, and compared against its predecessor through [requiresReload],
 * because two of these settings are not settings at all from the
 * interpreter's point of view: they are properties of a loaded model.
 *
 * Defaults are the server's, copied from UBL-standalone's config.yaml rather
 * than re-tuned, so that a difference between this app and the server traces
 * to the export and the quantization rather than to a knob someone moved.
 */
data class ControlsState(
    /** `posm.confidence` on the server. */
    val scoreThreshold: Float = 0.30f,
    val imgsz: Int = 640,
    val useAllowlist: Boolean = true,
    val delegate: DelegateKind = DelegateKind.GPU,
    val frozen: Boolean = false,
    /**
     * Draw the boxes and their labels.
     *
     * On by default, because the boxes are the point of a detection viewer.
     * Turning them off leaves the preview clear and the tally still counting,
     * which is what you want when looking at the shelf itself rather than at
     * what the model thinks of it.
     */
    val showBoxes: Boolean = true,
    /** `fixed_shelf` instance cap, used by the QPDS pipeline in phase 2. */
    val instanceCap: Int = 40,
) {
    /**
     * Whether moving from [previous] to this state means rebuilding the
     * interpreter.
     *
     * True only for resolution and delegate. LiteRT fixes the input shape at
     * export time, so each resolution is a separate 57 MB file, and the
     * delegate is chosen when the interpreter is constructed. Everything else
     * is read per frame and costs nothing to change.
     */
    fun requiresReload(previous: ControlsState): Boolean =
        imgsz != previous.imgsz || delegate != previous.delegate

    /** Asset path for a model family at this resolution. */
    fun assetFor(family: String): String = "models/${family}_$imgsz.tflite"

    companion object {
        /**
         * Resolutions with a shipped model.
         *
         * Offering one that was never exported is how the app greets someone
         * with a missing-asset message, so the selector is built from this
         * rather than from a free-form number.
         */
        val OFFERED_SIZES = listOf(640, 960)
    }
}
