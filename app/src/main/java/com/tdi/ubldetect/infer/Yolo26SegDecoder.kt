package com.tdi.ubldetect.infer

import com.tdi.ubldetect.imaging.Letterbox
import com.tdi.ubldetect.imaging.LetterboxTransform
import com.tdi.ubldetect.imaging.RectF
import kotlin.math.ceil
import kotlin.math.exp
import kotlin.math.floor

/**
 * One segmented product: where it is, how sure, and how to rebuild its mask.
 *
 * The coefficients are kept rather than the mask itself, because a mask is
 * only needed for instances that survive the cap, and building it is the
 * expensive half.
 */
class SegInstance(
    val box: RectF,
    val boxInModel: RectF,
    val score: Float,
    val coeffs: FloatArray,
)

/**
 * A mask over part of the proto grid, values 0..1.
 *
 * Covers an instance's own box, not the whole frame, so its size scales with
 * the product rather than the picture.
 *
 * It carries [cover], the model-pixel region its cells actually span, because
 * that is not the box. The decoder works in whole proto cells, so the region
 * is the box grown out to cell boundaries and then clamped to the model
 * square. Treating the grid as if it spanned the box exactly put the mask in
 * the wrong place: off by up to a cell ordinarily, and for a box hanging off
 * the frame edge, offset and stretched by tens of percent, which blacks out
 * the wrong pixels and hands the classifier a crop unlike anything it was
 * trained on.
 */
class MaskGrid(
    private val values: FloatArray,
    val width: Int,
    val height: Int,
    /** The region of the model square these cells cover, in model pixels. */
    private val cover: RectF,
    private val transform: LetterboxTransform,
) {
    fun at(x: Int, y: Int): Float = values[y * width + x]

    /**
     * The mask value covering a source-frame pixel.
     *
     * Source to model to cell, which is the same chain the server walks when
     * it resizes the mask to full image size and then crops it. Cells outside
     * the grid are clamped onto its edge: a box can extend past the model
     * square, and those pixels lie outside the frame anyway, but the index
     * still has to stay in range.
     */
    fun atSource(sx: Float, sy: Float): Float {
        val modelX = sx * transform.scale + transform.padX
        val modelY = sy * transform.scale + transform.padY
        val cellW = cover.width() / width
        val cellH = cover.height() / height
        val mx = floor((modelX - cover.left) / cellW).toInt()
        val my = floor((modelY - cover.top) / cellH).toInt()
        return at(mx.coerceIn(0, width - 1), my.coerceIn(0, height - 1))
    }
}

/**
 * Reads the NMS-free segmentation head.
 *
 * Contract measured from the exported file (docs/model-contract.md): rows are
 * `[1, 300, 38]`, being `x1, y1, x2, y2, score, class`, then 32 mask
 * coefficients. The class is always 0 because the model has a single
 * "product" class; naming is the classifier's job in stage two. Protos are
 * `[1, 32, P, P]` with **P = N/4**, which is 160 at imgsz 640 and 240 at 960.
 */
object Yolo26SegDecoder {

    private const val STRIDE = 38
    private const val COEFF_OFFSET = 6
    const val COEFF_COUNT = 32

    /**
     * @param maxInstances cap per frame, so one crowded shelf cannot stall
     *   the loop. Rows arrive score-sorted, so this keeps the most confident.
     */
    fun decode(
        raw: FloatArray,
        shape: IntArray,
        scoreThreshold: Float,
        maxInstances: Int,
        t: LetterboxTransform,
    ): List<SegInstance> {
        val rowCount = shape[1]
        val out = ArrayList<SegInstance>(minOf(maxInstances, rowCount).coerceAtLeast(0))

        for (i in 0 until rowCount) {
            if (out.size >= maxInstances) break
            val o = i * STRIDE
            val score = raw[o + 4]
            // Score-sorted, so the first row below threshold ends the useful
            // part of the tensor.
            if (score < scoreThreshold) break

            val inModel = RectF(raw[o], raw[o + 1], raw[o + 2], raw[o + 3])
            out.add(
                SegInstance(
                    box = Letterbox.rectToSource(inModel, t),
                    boxInModel = inModel,
                    score = score,
                    coeffs = raw.copyOfRange(o + COEFF_OFFSET, o + COEFF_OFFSET + COEFF_COUNT),
                ),
            )
        }
        return out
    }

    /**
     * Builds an instance's mask, over its own box only.
     *
     * `sigmoid(coeffs . protos)` across the 32 channels. Restricted to the
     * box in proto coordinates: computing the full P by P grid per instance,
     * then cropping, would do roughly twenty times the work on a shelf of
     * twenty products and throw almost all of it away.
     *
     * The cell range is grown outwards, floor on one side and ceil on the
     * other, so the grid contains the whole box. Truncating both ends instead
     * left the box poking out of its own mask by up to a cell. The range is
     * then clamped to the grid, because boxes legitimately exceed the input
     * square: the measured rows ranged from -32 to 691 at imgsz 640.
     *
     * The returned grid records the region it covers, in model pixels, which
     * is how [MaskGrid.atSource] places it back on the frame. It is not the
     * box, and the difference is what made the mask land in the wrong place.
     */
    fun maskFor(
        inst: SegInstance,
        protos: FloatArray,
        protoShape: IntArray,
        t: LetterboxTransform,
    ): MaskGrid {
        val channels = protoShape[1]
        val gridH = protoShape[2]
        val gridW = protoShape[3]
        val plane = gridH * gridW

        // Model pixels to proto cells, read off the export rather than
        // assumed: the proto shape gives the grid and the transform gives the
        // input square, so a future export with a different stride stays
        // correct instead of silently rescaling every mask.
        val cell = gridW.toFloat() / t.inputSize

        val x0 = floor(inst.boxInModel.left * cell).toInt().coerceIn(0, gridW - 1)
        val y0 = floor(inst.boxInModel.top * cell).toInt().coerceIn(0, gridH - 1)
        val x1 = ceil(inst.boxInModel.right * cell).toInt().coerceIn(x0 + 1, gridW)
        val y1 = ceil(inst.boxInModel.bottom * cell).toInt().coerceIn(y0 + 1, gridH)

        val w = x1 - x0
        val h = y1 - y0
        val values = FloatArray(w * h)

        for (y in 0 until h) {
            val row = (y0 + y) * gridW
            for (x in 0 until w) {
                val cellIndex = row + x0 + x
                var acc = 0f
                for (c in 0 until channels) {
                    acc += inst.coeffs[c] * protos[c * plane + cellIndex]
                }
                values[y * w + x] = 1f / (1f + exp(-acc))
            }
        }
        // Back to model pixels: the cells span [x0, x1) horizontally, so
        // the region starts at the left edge of x0 and ends at the left edge
        // of x1.
        return MaskGrid(
            values, w, h,
            RectF(x0 / cell, y0 / cell, x1 / cell, y1 / cell),
            t,
        )
    }

}
