package com.tdi.ubldetect.infer

import java.nio.ByteBuffer
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/**
 * Stage two: naming a segmented product.
 *
 * The restriction is the point. A slab carries a handful of SKUs, so asking
 * the classifier to choose among those rather than all 56 removes most of the
 * ways it can be wrong. With the allowlist off the app deliberately takes the
 * open answer, which is how you tell "not detected" from "detected as
 * something this slab does not stock".
 *
 * Note what is absent: batching. The exported model's input is
 * `[1, 3, 224, 224]`, a fixed batch of 1, so crops go through one at a time.
 * The plan assumed a batch of 8; the export does not allow it.
 */
class ClsRunnerTest {

    private val numClasses = 56

    /** Scores where 41 is the global best and 3 the best of the slab's own. */
    private fun logits(): FloatArray = FloatArray(numClasses).also {
        it[41] = 0.80f
        it[3] = 0.55f
        it[7] = 0.20f
    }

    private fun crop() = ByteBuffer.allocate(8)

    @Test
    fun restrictedArgmaxPicksTheBestAllowedClassNotTheGlobalBest() {
        assertEquals(3, RestrictedArgmax.pick(logits(), setOf(3, 7), 0.30f)!!.first)
    }

    @Test
    fun restrictedArgmaxReturnsTheScoreOfTheClassItPicked() {
        assertEquals(0.55f, RestrictedArgmax.pick(logits(), setOf(3, 7), 0.30f)!!.second, 1e-6f)
    }

    @Test
    fun restrictedArgmaxReturnsNullBelowMinScore() {
        assertNull(RestrictedArgmax.pick(logits(), setOf(7), 0.30f))
    }

    @Test
    fun nullAllowlistPicksTheGlobalBest() {
        assertEquals(41, RestrictedArgmax.pick(logits(), null, 0.30f)!!.first)
    }

    @Test
    fun anEmptyAllowlistPicksNothing() {
        // A slab whose classes the model does not have. The picker refuses
        // these, but the decoder must not pick a class anyway if one slips by.
        assertNull(RestrictedArgmax.pick(logits(), emptySet(), 0.30f))
    }

    @Test
    fun allowedClassesOutsideTheModelsRangeAreIgnored() {
        // Catalog ids come from the exported names, but a stale catalog could
        // name a class this model does not have. That must not index past the
        // end of the output.
        assertEquals(3, RestrictedArgmax.pick(logits(), setOf(3, 999), 0.30f)!!.first)
    }

    @Test
    fun classifyReturnsOneSlotPerCropIncludingNullsSoIndicesStayAligned() {
        // The caller zips these back onto segmentation boxes by index, so a
        // dropped entry would label the wrong product.
        val runner = ClsRunner(numClasses) { _, out -> logits().copyInto(out) }
        val out = runner.classify(listOf(crop(), crop(), crop()), setOf(999), 0.30f)
        assertEquals(3, out.size)
        assertEquals(listOf(null, null, null), out)
    }

    @Test
    fun classifyRunsTheModelOncePerCropBecauseTheBatchIsFixedAtOne() {
        var calls = 0
        val runner = ClsRunner(numClasses) { _, out ->
            calls++
            logits().copyInto(out)
        }
        runner.classify(List(20) { crop() }, null, 0.30f)
        assertEquals(20, calls)
    }

    @Test
    fun classifyAppliesTheAllowlistPerCrop() {
        val runner = ClsRunner(numClasses) { _, out -> logits().copyInto(out) }
        val out = runner.classify(listOf(crop(), crop()), setOf(3), 0.30f)
        assertEquals(listOf(3, 3), out.map { it?.first })
    }

    @Test
    fun classifyOfNothingIsNothing() {
        val runner = ClsRunner(numClasses) { _, _ -> error("must not run") }
        assertEquals(emptyList<Pair<Int, Float>?>(), runner.classify(emptyList(), null, 0.30f))
    }
}
