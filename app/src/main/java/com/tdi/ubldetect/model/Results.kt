package com.tdi.ubldetect.model

import com.tdi.ubldetect.imaging.LetterboxTransform
import com.tdi.ubldetect.imaging.RectF

/**
 * One detection, in source-frame coordinates.
 *
 * Carries a class id rather than a label. Naming belongs to the catalog,
 * which knows what the chosen slab or category calls that class, and the
 * decoder has no business knowing about either.
 */
data class Detection(val box: RectF, val classId: Int, val score: Float)

/**
 * Everything one processed frame produced.
 *
 * The timestamp and transform travel with the detections rather than being
 * read from "the current frame", because by the time this is drawn the camera
 * has moved on. Without them the overlay cannot tell a late result from a
 * fresh one, and cannot map boxes with the geometry they were computed under.
 *
 * @param timingsMs per stage, for the on-screen readout and the Task 17 table
 */
data class FrameResult(
    val detections: List<Detection>,
    val frameTimestampNs: Long,
    val transform: LetterboxTransform,
    val timingsMs: Map<String, Long>,
)
