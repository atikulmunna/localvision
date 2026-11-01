package com.tdi.ubldetect.pipeline

import android.os.SystemClock
import com.tdi.ubldetect.catalog.CatalogEntry
import com.tdi.ubldetect.imaging.CropBuilder
import com.tdi.ubldetect.imaging.InputTensor
import com.tdi.ubldetect.imaging.RgbFrame
import com.tdi.ubldetect.infer.ClsRunner
import com.tdi.ubldetect.infer.TfliteRunner
import com.tdi.ubldetect.infer.Yolo26SegDecoder
import com.tdi.ubldetect.model.Detection
import com.tdi.ubldetect.model.FrameResult
import com.tdi.ubldetect.ui.ControlsState
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * QPDS: segment the products, then name each one.
 *
 * Stage one finds products (the segmentation model has a single "product"
 * class, so it says where but never what). Stage two crops each instance,
 * masked to its own silhouette, and classifies it against the SKUs the
 * chosen slab expects.
 *
 * The cost is in stage two and grows with the shelf: the classifier's input
 * is a fixed batch of 1, so a shelf of twenty products is twenty inferences
 * on top of the segmentation. [ControlsState.instanceCap] bounds that, and
 * because the rows arrive score-sorted the cap keeps the most confident
 * instances rather than an arbitrary slice.
 *
 * Thresholds are the server's, copied from `config.yaml` rather than
 * re-tuned: `seg_conf` 0.20 and `cls_conf` 0.30.
 */
class QpdsPipeline(
    private val seg: TfliteRunner,
    private val cls: ClsRunner,
    private val entry: CatalogEntry,
) : FramePipeline {
    private val inputSize = seg.inputSize
    private val segShapes = seg.outputShapes()
    private val rowShape = segShapes[0]
    private val protoShape = segShapes[1]

    private val input: ByteBuffer =
        ByteBuffer.allocateDirect(inputSize * inputSize * 3 * 4).order(ByteOrder.nativeOrder())
    private val rows = FloatArray(rowShape.fold(1) { a, b -> a * b })
    private val protos = FloatArray(protoShape.fold(1) { a, b -> a * b })

    /**
     * One crop buffer, reused for every instance in every frame.
     *
     * A crowded shelf builds dozens of these per frame; allocating 600 KB
     * apiece would spend more time collecting garbage than classifying.
     */
    private val crop: ByteBuffer =
        ByteBuffer.allocateDirect(CLS_SIZE * CLS_SIZE * 3 * 4).order(ByteOrder.nativeOrder())

    override fun process(
        frame: RgbFrame,
        timestampNs: Long,
        state: ControlsState,
        extraTimings: Map<String, Long>,
    ): FrameResult {
        val t0 = SystemClock.elapsedRealtime()
        val transform = InputTensor.letterboxToNchw(frame, inputSize, input)

        val t1 = SystemClock.elapsedRealtime()
        seg.run(input, listOf(rows, protos))

        val t2 = SystemClock.elapsedRealtime()
        val instances = Yolo26SegDecoder.decode(
            raw = rows,
            shape = rowShape,
            scoreThreshold = SEG_CONF,
            maxInstances = state.instanceCap,
            t = transform,
        )

        val t3 = SystemClock.elapsedRealtime()
        val allowed = if (state.useAllowlist) entry.classIds else null
        val detections = ArrayList<Detection>(instances.size)

        for (inst in instances) {
            val mask = Yolo26SegDecoder.maskFor(inst, protos, protoShape, transform)
            val built = CropBuilder.build(
                frame = frame,
                box = inst.box,
                mask = mask,
                maskThreshold = MASK_THRESHOLD,
                outSize = CLS_SIZE,
                out = crop,
            )
            // Degenerate crops are skipped, as the server does, rather than
            // classified as whatever an empty image happens to resemble.
            if (!built) continue

            val picked = cls.classify(listOf(crop), allowed, state.scoreThreshold)[0] ?: continue
            detections.add(Detection(inst.box, picked.first, picked.second))
        }
        val t4 = SystemClock.elapsedRealtime()

        return FrameResult(
            detections = detections,
            frameTimestampNs = timestampNs,
            transform = transform,
            timingsMs = extraTimings + mapOf(
                "prep" to (t1 - t0),
                "seg" to (t2 - t1),
                "decode" to (t3 - t2),
                "cls x${instances.size}" to (t4 - t3),
            ),
        )
    }

    private companion object {
        /** `fixed_shelf.seg_conf` on the server. */
        const val SEG_CONF = 0.20f

        /** The classifier's input side, fixed by the export. */
        const val CLS_SIZE = 224

        /**
         * Mask cells at or above this count as product.
         *
         * The server thresholds a binary mask; this is the equivalent cut on
         * the sigmoid output. Measured separation on a real shelf was 0.948
         * inside the box against 0.374 overall, so 0.5 sits well clear of
         * both.
         */
        const val MASK_THRESHOLD = 0.5f
    }
}
