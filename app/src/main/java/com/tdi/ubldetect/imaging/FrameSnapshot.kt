package com.tdi.ubldetect.imaging

import java.nio.ByteBuffer

/**
 * Holds a copy of one frame and whatever was worked out about it.
 *
 * The camera's RGB buffer is reused for every frame, so anything that reads
 * it later reads a mixture: a saved debug JPEG could be two frames spliced
 * together, with the boxes drawn on it belonging to neither. Those saves are
 * the evidence base for `validation/`, where a torn frame reads as a geometry
 * bug in the pipeline rather than as a race in the save button.
 *
 * So the pixels are copied at the moment the frame is accepted. One copy per
 * frame, which is a 2.7 MB memcpy at 720p against 400 ms of inference, and it
 * is taken on the analysis thread, the only thread that touches this.
 *
 * The copy also decides *which* frame a save writes. Not taking one, while
 * the overlay is frozen, is how Hold means what it says: the frame on screen
 * is the frame that gets saved.
 *
 * @param T whatever accompanies the frame; the holder does not look at it
 */
class FrameSnapshot<T> {

    class Taken<T>(val frame: RgbFrame, val result: T)

    /**
     * The last frame taken, or null before the first one.
     *
     * Volatile because the UI thread reads it to decide whether a save has
     * anything to write.
     */
    @Volatile
    var taken: Taken<T>? = null
        private set

    private var buffer: ByteBuffer? = null

    /** Copies [frame]'s pixels and keeps [result] with them. */
    fun take(frame: RgbFrame, result: T) {
        val needed = frame.width * frame.height * 3
        var dst = buffer
        if (dst == null || dst.capacity() < needed) {
            dst = ByteBuffer.allocate(needed)
            buffer = dst
        }
        val src = frame.buffer.duplicate()
        src.clear()
        src.limit(needed)
        dst.clear()
        dst.limit(needed)
        dst.put(src)
        dst.clear()
        dst.limit(needed)
        taken = Taken(RgbFrame(dst, frame.width, frame.height), result)
    }
}
