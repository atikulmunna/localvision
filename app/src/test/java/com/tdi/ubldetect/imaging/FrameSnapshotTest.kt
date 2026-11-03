package com.tdi.ubldetect.imaging

import java.nio.ByteBuffer
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertSame
import org.junit.Test

/**
 * Keeping a frame still so it can be saved.
 *
 * The RGB buffer is reused for every frame, so reading it after the fact
 * returns whatever the camera has written since. A saved debug JPEG could be
 * two frames spliced together, with the drawn boxes belonging to neither, and
 * those saves are the evidence base for `validation/`: a torn frame reads as
 * a geometry bug in the pipeline.
 */
class FrameSnapshotTest {

    private fun frame(w: Int, h: Int, fill: Byte): RgbFrame {
        val buf = ByteBuffer.allocate(w * h * 3)
        for (i in 0 until w * h * 3) buf.put(i, fill)
        return RgbFrame(buf, w, h)
    }

    @Test
    fun startsEmpty() {
        assertNull(FrameSnapshot<String>().taken)
    }

    @Test
    fun takeCopiesThePixels() {
        val snap = FrameSnapshot<String>()
        snap.take(frame(4, 2, 7), "labels")
        val taken = snap.taken
        assertNotNull(taken)
        taken!!
        assertEquals(4, taken.frame.width)
        assertEquals(2, taken.frame.height)
        assertEquals(7, taken.frame.buffer.get(0).toInt())
        assertEquals("labels", taken.result)
    }

    @Test
    fun aLaterWriteToTheSourceDoesNotChangeWhatWasTaken() {
        // The whole point: the camera writes the next frame into the same
        // bytes while the snapshot is still being saved.
        val source = frame(4, 2, 7)
        val snap = FrameSnapshot<String>()
        snap.take(source, "labels")
        for (i in 0 until 4 * 2 * 3) source.buffer.put(i, 99)
        assertEquals(7, snap.taken!!.frame.buffer.get(0).toInt())
    }

    @Test
    fun theBufferIsReusedBetweenTakesOfTheSameSize() {
        // One copy per frame is cheap; one allocation per frame is 2.7 MB of
        // churn at 720p.
        val snap = FrameSnapshot<String>()
        snap.take(frame(4, 2, 1), "a")
        val first = snap.taken!!.frame.buffer
        snap.take(frame(4, 2, 2), "b")
        assertSame(first, snap.taken!!.frame.buffer)
        assertEquals(2, snap.taken!!.frame.buffer.get(0).toInt())
        assertEquals("b", snap.taken!!.result)
    }

    @Test
    fun aLargerFrameGetsABiggerBuffer() {
        val snap = FrameSnapshot<String>()
        snap.take(frame(2, 2, 1), "a")
        snap.take(frame(8, 8, 3), "b")
        assertEquals(8, snap.taken!!.frame.width)
        assertEquals(3, snap.taken!!.frame.buffer.get(8 * 8 * 3 - 1).toInt())
    }

    @Test
    fun aSmallerFrameDoesNotLeaveTheOldFramesPixelsBehindIt() {
        // The buffer is reused, so a smaller frame has to report its own size
        // rather than the buffer's, or the save reads the previous frame's
        // tail as image data.
        val snap = FrameSnapshot<String>()
        snap.take(frame(8, 8, 9), "a")
        snap.take(frame(2, 2, 4), "b")
        val taken = snap.taken!!
        assertEquals(2, taken.frame.width)
        assertEquals(2, taken.frame.height)
        assertEquals(2 * 2 * 3, taken.frame.buffer.limit())
    }
}
