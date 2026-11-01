package com.tdi.ubldetect.imaging

/**
 * Plain geometry types, deliberately not `android.graphics`.
 *
 * Everything that reasons about coordinates (letterboxing, the decoders, crop
 * building) uses these, so all of it is testable as JVM unit tests with no
 * device and no Android stubs. Only `OverlayView` converts to
 * `android.graphics.RectF`, at the draw call.
 */
data class PointF(val x: Float, val y: Float)

data class RectF(val left: Float, val top: Float, val right: Float, val bottom: Float) {
    fun centerX(): Float = (left + right) / 2f
    fun centerY(): Float = (top + bottom) / 2f
    fun width(): Float = right - left
    fun height(): Float = bottom - top
}

/**
 * How a source frame was fitted into the model's square input.
 *
 * @param scale factor the source was multiplied by
 * @param padX left padding inside the square, in model pixels
 * @param padY top padding inside the square, in model pixels
 * @param inputSize the square's side, 640 or 960
 */
data class LetterboxTransform(
    val scale: Float,
    val padX: Float,
    val padY: Float,
    val inputSize: Int,
)

/**
 * Fits a rectangular frame into the model's square input, and maps back.
 *
 * The model is fed an N by N image: the frame is scaled by the smaller of the
 * two ratios so it fits whole, then centred, with the remainder padded. Its
 * output boxes are therefore in that square's pixel space and mean nothing on
 * screen until they are mapped back, which is what `toSource` and
 * `rectToSource` are for.
 *
 * Geometry only. The padding *colour* is not decided here: it belongs with the
 * pixel conversion, and it matters, because the model was trained on 114 grey
 * padding (see docs/model-contract.md).
 */
object Letterbox {

    fun compute(srcWidth: Int, srcHeight: Int, inputSize: Int): LetterboxTransform {
        val scale = minOf(
            inputSize.toFloat() / srcWidth.toFloat(),
            inputSize.toFloat() / srcHeight.toFloat(),
        )
        return LetterboxTransform(
            scale = scale,
            padX = (inputSize - srcWidth * scale) / 2f,
            padY = (inputSize - srcHeight * scale) / 2f,
            inputSize = inputSize,
        )
    }

    /**
     * Model-square pixels back to source-frame pixels.
     *
     * A point inside the padding maps outside the frame, which is correct and
     * deliberate: it is the caller's job to clamp when that matters, rather
     * than this quietly returning a plausible wrong answer.
     */
    fun toSource(modelX: Float, modelY: Float, t: LetterboxTransform): PointF =
        PointF((modelX - t.padX) / t.scale, (modelY - t.padY) / t.scale)

    fun rectToSource(modelRect: RectF, t: LetterboxTransform): RectF {
        val topLeft = toSource(modelRect.left, modelRect.top, t)
        val bottomRight = toSource(modelRect.right, modelRect.bottom, t)
        return RectF(topLeft.x, topLeft.y, bottomRight.x, bottomRight.y)
    }
}
