package com.tdi.ubldetect.infer

import com.tdi.ubldetect.imaging.Letterbox
import com.tdi.ubldetect.imaging.LetterboxTransform
import com.tdi.ubldetect.imaging.RectF
import kotlin.math.exp

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
 */
class MaskGrid(private val values: FloatArray, val width: Int, val height: Int) {
    fun at(x: Int, y: Int): Float = values[y * width + x]
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
     * The box is clamped to the grid because boxes legitimately exceed the
     * input square; the measured rows ranged from -32 to 691 at imgsz 640.
     */
    fun maskFor(
        inst: SegInstance,
        protos: FloatArray,
        protoShape: IntArray,
        @Suppress("UNUSED_PARAMETER") t: LetterboxTransform,
    ): MaskGrid {
        val channels = protoShape[1]
        val gridH = protoShape[2]
        val gridW = protoShape[3]
        val plane = gridH * gridW

        // Model pixels to proto cells. The grid is a quarter of the input,
        // so this ratio is derived rather than hardcoded to 4: a future
        // export with a different stride stays correct.
        val cell = gridW.toFloat() / inputSizeFor(gridW)

        val x0 = ((inst.boxInModel.left * cell).toInt()).coerceIn(0, gridW - 1)
        val y0 = ((inst.boxInModel.top * cell).toInt()).coerceIn(0, gridH - 1)
        val x1 = ((inst.boxInModel.right * cell).toInt()).coerceIn(x0 + 1, gridW)
        val y1 = ((inst.boxInModel.bottom * cell).toInt()).coerceIn(y0 + 1, gridH)

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
        return MaskGrid(values, w, h)
    }

    /** Input square that produced a proto grid of this width (P = N/4). */
    private fun inputSizeFor(gridWidth: Int): Int = gridWidth * 4
}
