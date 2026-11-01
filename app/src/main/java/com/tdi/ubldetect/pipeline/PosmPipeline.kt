package com.tdi.ubldetect.pipeline

import android.os.SystemClock
import com.tdi.ubldetect.catalog.CatalogEntry
import com.tdi.ubldetect.imaging.InputTensor
import com.tdi.ubldetect.imaging.RgbFrame
import com.tdi.ubldetect.infer.TfliteRunner
import com.tdi.ubldetect.infer.Yolo26DetectDecoder
import com.tdi.ubldetect.model.FrameResult
import com.tdi.ubldetect.ui.ControlsState
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * POSM: one detector pass, one set of labels.
 *
 * The simple pipeline of the two, and the reason POSM is built first: a
 * single model with meaningful class names, no second stage, no masks.
 *
 * Buffers are allocated once and reused. This runs continuously against a
 * live camera, so a per-frame allocation of several megabytes would spend
 * more time in the garbage collector than in the model.
 *
 * Confidence and the allowlist are read from the controls on every frame,
 * not held, so moving the slider takes effect on the next frame rather than
 * at the next rebuild.
 */
class PosmPipeline(
    private val runner: TfliteRunner,
    private val entry: CatalogEntry,
) : FramePipeline {
    private val inputSize = runner.inputSize
    private val outputShape = runner.outputShapes()[0]

    private val input: ByteBuffer =
        ByteBuffer.allocateDirect(inputSize * inputSize * 3 * 4).order(ByteOrder.nativeOrder())

    private val output = FloatArray(outputShape.fold(1) { a, b -> a * b })

    /**
     * @param extraTimings stages measured outside this pipeline, such as the
     *   YUV conversion, so the on-screen readout accounts for the whole frame
     *   rather than only the part this class happens to own.
     */
    override fun process(
        frame: RgbFrame,
        timestampNs: Long,
        state: ControlsState,
        extraTimings: Map<String, Long>,
    ): FrameResult {
        val t0 = SystemClock.elapsedRealtime()
        val transform = InputTensor.letterboxToNchw(frame, inputSize, input)

        val t1 = SystemClock.elapsedRealtime()
        runner.run(input, listOf(output))

        val t2 = SystemClock.elapsedRealtime()
        val detections = Yolo26DetectDecoder.decode(
            raw = output,
            shape = outputShape,
            scoreThreshold = state.scoreThreshold,
            allowed = if (state.useAllowlist) entry.classIds else null,
            t = transform,
        )
        val t3 = SystemClock.elapsedRealtime()

        return FrameResult(
            detections = detections,
            frameTimestampNs = timestampNs,
            transform = transform,
            timingsMs = extraTimings + mapOf(
                "prep" to (t1 - t0),
                "infer" to (t2 - t1),
                "decode" to (t3 - t2),
            ),
        )
    }
}
