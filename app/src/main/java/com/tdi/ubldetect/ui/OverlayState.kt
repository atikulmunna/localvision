package com.tdi.ubldetect.ui

import com.tdi.ubldetect.model.Detection
import com.tdi.ubldetect.model.FrameResult

/**
 * Decides what the overlay is currently showing.
 *
 * Kept apart from the view that draws it so the decisions are testable
 * without a device, a looper or a Canvas. The view asks this what to paint.
 *
 * Three rules, each of them a visible behaviour rather than bookkeeping:
 *
 * 1. **An empty result clears the overlay.** The detector lags the preview,
 *    so there are always boxes on screen from an earlier frame. If finding
 *    nothing left them up, pointing the camera at a bare shelf would still
 *    show labels, which is worse than showing nothing.
 * 2. **A frozen overlay ignores everything.** Freezing exists to read labels
 *    carefully; a result still in flight when the person froze must not wipe
 *    what they are now reading.
 * 3. **An older result never replaces a newer one.** Frames are decoded off
 *    the camera thread and a slow one can finish after a later fast one.
 *    Drawing it would send the overlay backwards in time.
 */
class OverlayState {

    var drawn: List<Detection> = emptyList()
        private set

    var drawnTimestampNs: Long = Long.MIN_VALUE
        private set

    var frozen: Boolean = false
        private set

    fun submit(result: FrameResult?) {
        if (frozen) return
        if (result == null) {
            drawn = emptyList()
            return
        }
        if (result.frameTimestampNs < drawnTimestampNs) return
        drawn = result.detections
        drawnTimestampNs = result.frameTimestampNs
    }

    fun freeze() {
        frozen = true
    }

    fun unfreeze() {
        frozen = false
    }
}
