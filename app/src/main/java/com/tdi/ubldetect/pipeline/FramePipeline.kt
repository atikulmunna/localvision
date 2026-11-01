package com.tdi.ubldetect.pipeline

import com.tdi.ubldetect.imaging.RgbFrame
import com.tdi.ubldetect.model.FrameResult
import com.tdi.ubldetect.ui.ControlsState

/**
 * One frame in, labelled boxes out.
 *
 * The controls are passed per frame rather than held, because most of them
 * are meant to take effect on the next frame: confidence, the allowlist and
 * the instance cap all change what a frame produces without touching the
 * interpreter. Holding them at construction is how a slider ends up inert
 * until something unrelated rebuilds the pipeline, which is exactly what
 * happened here before this interface existed.
 *
 * Only resolution and delegate need a rebuild, and
 * [ControlsState.requiresReload] is the single place that says so.
 */
interface FramePipeline {
    fun process(
        frame: RgbFrame,
        timestampNs: Long,
        state: ControlsState,
        extraTimings: Map<String, Long> = emptyMap(),
    ): FrameResult
}
