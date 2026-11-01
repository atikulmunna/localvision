package com.tdi.ubldetect.imaging

/**
 * How the camera frame was fitted into the preview view.
 *
 * @param scale factor the source was multiplied by to fill the view
 * @param offsetX left edge of the scaled source in view pixels, negative when cropped
 * @param offsetY top edge of the scaled source in view pixels, negative when cropped
 */
data class ViewMapping(val scale: Float, val offsetX: Float, val offsetY: Float)

/**
 * Maps source-frame coordinates onto the preview the person is looking at.
 *
 * This is the opposite transform to [Letterbox], and keeping them apart is the
 * whole point of the file.
 *
 * - [Letterbox] fits the frame **inside** a square: scale by `min` of the two
 *   ratios, pad the remainder. Nothing is lost; the model sees everything.
 * - A preview fills the view: scale by `max` of the two ratios and crop
 *   whatever overflows. Nothing is padded; the edges are simply not shown.
 *
 * A detection travels through both. The model returns a box in square pixels,
 * [Letterbox.rectToSource] brings it back to frame pixels, and [mapRect] puts
 * it where the person can actually see it. Using one rule where the other
 * belongs misplaces every box by a plausible-looking amount, which reads as a
 * model problem and is not one.
 *
 * Matches CameraX `PreviewView` in FILL_CENTER, its default.
 */
object PreviewMapping {

    fun fillCenter(srcWidth: Int, srcHeight: Int, viewWidth: Int, viewHeight: Int): ViewMapping {
        val scale = maxOf(
            viewWidth.toFloat() / srcWidth.toFloat(),
            viewHeight.toFloat() / srcHeight.toFloat(),
        )
        return ViewMapping(
            scale = scale,
            offsetX = (viewWidth - srcWidth * scale) / 2f,
            offsetY = (viewHeight - srcHeight * scale) / 2f,
        )
    }

    fun mapRect(sourceRect: RectF, m: ViewMapping): RectF = RectF(
        sourceRect.left * m.scale + m.offsetX,
        sourceRect.top * m.scale + m.offsetY,
        sourceRect.right * m.scale + m.offsetX,
        sourceRect.bottom * m.scale + m.offsetY,
    )
}
