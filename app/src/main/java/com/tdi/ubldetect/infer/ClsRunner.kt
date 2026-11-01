package com.tdi.ubldetect.infer

import java.nio.ByteBuffer

/**
 * Picks a class from the classifier's output, optionally restricted.
 *
 * Restricting to a slab's own SKUs is not a filter applied afterwards, it is
 * part of the decision: the best of six expected products is a different and
 * better answer than the best of fifty-six. With `allowed` null the open
 * answer comes back instead, which is what the allowlist toggle is for.
 */
object RestrictedArgmax {

    /**
     * @param scores already softmaxed by the model, so these are
     *   probabilities and comparable against a threshold directly
     * @param allowed class ids to consider, or null for all of them
     * @return the chosen class and its score, or null if nothing cleared
     *   [minScore]
     */
    fun pick(scores: FloatArray, allowed: Set<Int>?, minScore: Float): Pair<Int, Float>? {
        var bestId = -1
        var bestScore = Float.NEGATIVE_INFINITY

        if (allowed == null) {
            for (i in scores.indices) {
                if (scores[i] > bestScore) {
                    bestScore = scores[i]
                    bestId = i
                }
            }
        } else {
            for (id in allowed) {
                // A stale catalog could name a class this model does not
                // have; that must not index past the end of the output.
                if (id !in scores.indices) continue
                if (scores[id] > bestScore) {
                    bestScore = scores[id]
                    bestId = id
                }
            }
        }

        return if (bestId >= 0 && bestScore >= minScore) bestId to bestScore else null
    }
}

/**
 * Runs the classifier over the crops from one frame.
 *
 * One crop per inference, because the exported model's input is
 * `[1, 3, 224, 224]`: the batch is fixed at 1 and cannot be filled. The plan
 * assumed a batch of 8, which the export does not allow.
 *
 * Inference is injected rather than taken as a [TfliteRunner] so the
 * selection logic is testable on the JVM without a device.
 */
class ClsRunner(
    private val numClasses: Int,
    private val infer: (ByteBuffer, FloatArray) -> Unit,
) {
    constructor(runner: TfliteRunner) : this(
        numClasses = runner.outputShapes()[0].last(),
        infer = { input, out -> runner.run(input, listOf(out)) },
    )

    /** Reused across crops and frames; this runs dozens of times per frame. */
    private val scores = FloatArray(numClasses)

    /**
     * @return one entry per crop, in the same order, null where nothing
     *   cleared [minScore]. The caller zips these back onto segmentation
     *   boxes by index, so entries are never dropped.
     */
    fun classify(
        crops: List<ByteBuffer>,
        allowed: Set<Int>?,
        minScore: Float,
    ): List<Pair<Int, Float>?> = crops.map { crop ->
        infer(crop, scores)
        RestrictedArgmax.pick(scores, allowed, minScore)
    }
}
