package com.tdi.ubldetect.infer

import java.util.Collections
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlin.concurrent.thread
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertSame
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Keeping loaded models alive between camera screens.
 *
 * A model depends on the mode and the resolution, never on which category or
 * slab was picked, so every category opened the same file again and waited
 * 3.5 s for it. The pool exists to notice that.
 *
 * What it must not do is hold everything at once. The four bundled models are
 * 197 MiB together, and opening a new one before releasing the old left 678
 * MiB of mapped pages resident over ten resolution swaps, which is why the
 * order below is asserted rather than assumed.
 */
class ModelPoolTest {

    private class Fake(val asset: String, val delegate: DelegateKind) : AutoCloseable {
        var closed = false
        override fun close() {
            closed = true
        }
    }

    /**
     * The lists are synchronized because the pool opens models in parallel,
     * so two `open` lambdas genuinely run on two threads here.
     */
    private class Harness {
        val opened: MutableList<Fake> = Collections.synchronizedList(mutableListOf())
        val events: MutableList<String> = Collections.synchronizedList(mutableListOf())
        var failOn: String? = null

        /** Makes a named asset take this long to open. */
        val slowOpens = mutableMapOf<String, Long>()

        /** Counted down by the first open to start, for the locking tests. */
        val openStarted = CountDownLatch(1)

        /** Holds every open until a test releases it, likewise. */
        var block: CountDownLatch? = null

        val pool = ModelPool<Fake> { asset, delegate ->
            if (asset == failOn) throw ModelAssetMissing(asset)
            events += "open $asset"
            openStarted.countDown()
            block?.await(5, TimeUnit.SECONDS)
            slowOpens[asset]?.let { Thread.sleep(it) }
            Fake(asset, delegate).also { opened += it }
        }

        init {
            pool.onClose = { events += "close ${it.asset}" }
        }

        val live: Int get() = opened.count { !it.closed }
    }

    private val gpu = DelegateKind.GPU
    private val cpu = DelegateKind.CPU_XNNPACK

    @Test
    fun asksForTheSameModelTwiceAndGetsTheSameOne() {
        // The whole point: a second category in the same mode is free.
        val h = Harness()
        val first = h.pool.acquire(listOf("posm_640"), gpu)
        val second = h.pool.acquire(listOf("posm_640"), gpu)
        assertSame(first[0], second[0])
        assertEquals(listOf("open posm_640"), h.events)
    }

    @Test
    fun runnersComeBackInTheOrderTheyWereAskedFor() {
        // QPDS indexes them positionally: [0] is the segmenter, [1] the
        // classifier. A map's iteration order is not an answer.
        val h = Harness()
        val got = h.pool.acquire(listOf("qpds_seg_640", "qpds_cls_224"), gpu)
        assertEquals("qpds_seg_640", got[0].asset)
        assertEquals("qpds_cls_224", got[1].asset)
    }

    @Test
    fun theOldModelIsClosedBeforeTheNewOneIsOpened() {
        // Not merely closed: closed first. Opening first means two 57 MB
        // models resident at once, which is what cost 678 MiB before.
        val h = Harness()
        h.pool.acquire(listOf("posm_640"), gpu)
        h.pool.acquire(listOf("posm_960"), gpu)
        assertEquals(listOf("open posm_640", "close posm_640", "open posm_960"), h.events)
    }

    @Test
    fun switchingModeKeepsNothingItDoesNotNeed() {
        val h = Harness()
        h.pool.acquire(listOf("posm_640"), gpu)
        h.pool.acquire(listOf("qpds_seg_640", "qpds_cls_224"), gpu)
        assertEquals(2, h.live)
        assertTrue(h.opened.first { it.asset == "posm_640" }.closed)
    }

    @Test
    fun aDelegateChangeIsADifferentModel() {
        // The delegate is chosen when the interpreter is built, so the cached
        // one cannot answer for the other.
        val h = Harness()
        val onGpu = h.pool.acquire(listOf("posm_640"), gpu)[0]
        val onCpu = h.pool.acquire(listOf("posm_640"), cpu)[0]
        assertTrue(onGpu.closed)
        assertEquals(cpu, onCpu.delegate)
    }

    @Test
    fun partOfTheSetIsKeptWhenOnlyTheRestChanges() {
        // The QPDS classifier has one resolution, so a 640 to 960 swap must
        // not reload it.
        val h = Harness()
        h.pool.acquire(listOf("qpds_seg_640", "qpds_cls_224"), gpu)
        h.events.clear()
        h.pool.acquire(listOf("qpds_seg_960", "qpds_cls_224"), gpu)
        assertEquals(listOf("close qpds_seg_640", "open qpds_seg_960"), h.events)
    }

    @Test
    fun releasingWhileInUseDoesNothing() {
        // Memory pressure can arrive while a frame is inside the interpreter.
        // Closing it there is the native crash this app already had twice.
        val h = Harness()
        h.pool.beginUse()
        h.pool.acquire(listOf("posm_640"), gpu)
        assertFalse(h.pool.releaseIfIdle())
        assertEquals(1, h.live)
    }

    @Test
    fun theLastHolderToLeaveDecidesWhetherItIsIdle() {
        // Leaving one camera for another runs the new screen's onCreate
        // before the old one's onDestroy. A flag would let the departing
        // screen declare the pool idle while the arriving one runs frames.
        val h = Harness()
        h.pool.beginUse()
        h.pool.acquire(listOf("posm_640"), gpu)
        h.pool.beginUse()
        h.pool.endUse()
        assertFalse("released while a holder remained", h.pool.releaseIfIdle())
        h.pool.endUse()
        assertTrue(h.pool.releaseIfIdle())
    }

    @Test
    fun releasingWhenIdleClosesEverything() {
        val h = Harness()
        h.pool.beginUse()
        h.pool.acquire(listOf("qpds_seg_640", "qpds_cls_224"), gpu)
        h.pool.endUse()
        assertTrue(h.pool.releaseIfIdle())
        assertEquals(0, h.live)
    }

    @Test
    fun whatFailedToOpenDoesNotStrandWhatDid() {
        // QPDS opens two. If the classifier is missing, the segmenter is
        // already open and nothing else holds a reference to it.
        val h = Harness()
        h.failOn = "qpds_cls_224"
        assertThrows(ModelAssetMissing::class.java) {
            h.pool.acquire(listOf("qpds_seg_640", "qpds_cls_224"), gpu)
        }
        assertEquals(0, h.live)
    }

    @Test
    fun aFailedAcquireLeavesThePoolUsable() {
        val h = Harness()
        h.failOn = "posm_960"
        assertThrows(ModelAssetMissing::class.java) {
            h.pool.acquire(listOf("posm_960"), gpu)
        }
        assertEquals(1, h.pool.acquire(listOf("posm_640"), gpu).size)
    }

    @Test
    fun warmingLeavesThePoolIdle() {
        // Warming happens from the picker, where nothing is running yet. If
        // it marked the pool in use, memory pressure could never release it.
        val h = Harness()
        h.pool.warm(listOf("posm_640"), gpu)
        assertTrue(h.pool.releaseIfIdle())
        assertEquals(0, h.live)
    }

    @Test
    fun warmingThenOpeningThatModeOpensNothingFurther() {
        val h = Harness()
        h.pool.warm(listOf("posm_640"), gpu)
        h.pool.acquire(listOf("posm_640"), gpu)
        assertEquals(listOf("open posm_640"), h.events)
    }

    @Test
    fun warmingNeverClosesWhatIsAlreadyThere() {
        // Warming runs on its own thread while a camera may still be shutting
        // down, and what it would close is exactly what that camera's last
        // frame is inside. The next acquire tidies up instead, from the
        // analysis thread, where closing is safe by construction.
        val h = Harness()
        h.pool.beginUse()
        h.pool.acquire(listOf("qpds_seg_640", "qpds_cls_224"), gpu)
        h.pool.warm(listOf("posm_640"), gpu)
        assertEquals(3, h.live)
        assertTrue(h.events.none { it.startsWith("close") })
    }

    @Test
    fun theNextAcquireClearsWhatWarmingLeftBehind() {
        val h = Harness()
        h.pool.warm(listOf("posm_640"), gpu)
        h.pool.acquire(listOf("qpds_seg_640", "qpds_cls_224"), gpu)
        assertEquals(2, h.live)
        assertTrue(h.opened.first { it.asset == "posm_640" }.closed)
    }

    @Test
    fun warmingSomethingThatIsNotBundledIsNotACrash() {
        // Nobody is waiting on a warm up, and a resolution with no shipped
        // model is a reachable state the picker must survive.
        val h = Harness()
        h.failOn = "posm_960"
        h.pool.warm(listOf("posm_960"), gpu)
        assertEquals(0, h.live)
    }

    @Test
    fun takingAHoldDoesNotWaitForARunningLoad() {
        // The regression this was written for: beginUse took the pool's
        // lock, a warm up holds that lock for its whole duration, and
        // beginUse is called from onCreate on the main thread. Tapping a
        // slab while QPDS was still warming measured the camera screen at
        // 4121 ms to appear, and what the person saw was a frozen picker.
        val h = Harness()
        h.block = CountDownLatch(1)
        val warming = thread { h.pool.warm(listOf("qpds_seg_640"), gpu) }
        assertTrue("the load never started", h.openStarted.await(5, TimeUnit.SECONDS))

        val began = System.nanoTime()
        h.pool.beginUse()
        h.pool.endUse()
        val waitedMs = (System.nanoTime() - began) / 1_000_000

        h.block!!.countDown()
        warming.join()
        assertTrue("beginUse waited ${waitedMs}ms for the load", waitedMs < 200)
    }

    @Test
    fun readingProgressDoesNotWaitForARunningLoadEither() {
        // Same lock, and the caller here is the view drawing the spinner
        // about this very load.
        val h = Harness()
        h.block = CountDownLatch(1)
        val warming = thread { h.pool.warm(listOf("qpds_seg_640", "qpds_cls_224"), gpu) }
        assertTrue(h.openStarted.await(5, TimeUnit.SECONDS))

        val began = System.nanoTime()
        val seen = h.pool.loadProgress
        val waitedMs = (System.nanoTime() - began) / 1_000_000

        h.block!!.countDown()
        warming.join()
        assertTrue("loadProgress waited ${waitedMs}ms", waitedMs < 200)
        assertEquals(2, seen.total)
        assertTrue("a load in flight is a load being waited for", seen.waiting)
    }

    @Test
    fun theTwoQpdsModelsOpenAtTheSameTime() {
        // Sequentially these cost their sum, which on the device is 3.8 s
        // plus 1.4 s. Overlapped the pair costs about what the segmenter
        // costs alone.
        val h = Harness()
        h.slowOpens["qpds_seg_640"] = 400
        h.slowOpens["qpds_cls_224"] = 400

        val began = System.nanoTime()
        h.pool.warm(listOf("qpds_seg_640", "qpds_cls_224"), gpu)
        val tookMs = (System.nanoTime() - began) / 1_000_000

        assertEquals(2, h.live)
        assertTrue("two 400ms loads took ${tookMs}ms, so they were sequential", tookMs < 700)
    }

    @Test
    fun orderSurvivesTheSecondModelFinishingFirst() {
        // QPDS indexes positionally, and parallel loads finish in whatever
        // order they finish. The slow one is asked for first on purpose.
        val h = Harness()
        h.slowOpens["qpds_seg_640"] = 300
        val got = h.pool.acquire(listOf("qpds_seg_640", "qpds_cls_224"), gpu)
        assertEquals("qpds_seg_640", got[0].asset)
        assertEquals("qpds_cls_224", got[1].asset)
    }

    @Test
    fun progressCountsOnlyWhatActuallyHadToOpen() {
        val h = Harness()
        h.pool.warm(listOf("qpds_seg_640", "qpds_cls_224"), gpu)
        assertEquals(LoadProgress(2, 2), h.pool.loadProgress)
        assertFalse(h.pool.loadProgress.waiting)

        // Asking again opens nothing, so there is nothing to wait for and
        // the card must not appear.
        h.pool.acquire(listOf("qpds_seg_640", "qpds_cls_224"), gpu)
        assertEquals(LoadProgress(0, 0), h.pool.loadProgress)
        assertFalse(h.pool.loadProgress.waiting)
    }

    @Test
    fun oneModelFailingStillHoldsTheOneThatOpened() {
        // Parallel opens mean the good one is already built by the time the
        // bad one throws. Dropping it on the floor would leak it.
        val h = Harness()
        h.failOn = "qpds_cls_224"
        h.pool.warm(listOf("qpds_seg_640", "qpds_cls_224"), gpu)
        assertEquals(1, h.live)
        assertEquals("qpds_seg_640", h.opened.single().asset)
    }
}
