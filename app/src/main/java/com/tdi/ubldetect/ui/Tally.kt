package com.tdi.ubldetect.ui

import com.tdi.ubldetect.model.Detection

/** One line of the tally: a product name and how many were found. */
data class TallyRow(val label: String, val count: Int)

/**
 * Counts a frame's detections by name.
 *
 * The boxes already say where things are. What they cannot say, off a live
 * preview that changes every 400 ms, is how many of each are up there, which
 * is the question a shelf audit actually asks.
 *
 * Grouped by **label** rather than class id, because that is what is on
 * screen: a shelf talker is detected in four pieces that all carry one name,
 * and listing it four times would be a count of model classes, not of things
 * on the shelf.
 */
object Tally {

    /**
     * @param maxRows how many lines there is room for. If more kinds were
     *   found, the last line folds the remainder into "+N more" so that the
     *   counts still add up to every detection.
     */
    fun of(
        detections: List<Detection>,
        labeller: (Int) -> String,
        maxRows: Int,
    ): List<TallyRow> {
        if (detections.isEmpty() || maxRows <= 0) return emptyList()

        val counts = HashMap<String, Int>()
        for (d in detections) {
            val label = labeller(d.classId)
            counts[label] = (counts[label] ?: 0) + 1
        }

        // Commonest first, then by name. The name tiebreak is not cosmetic:
        // detection order varies frame to frame, so without it two products
        // with one facing each would trade places several times a second.
        val sorted = counts.entries
            .sortedWith(compareByDescending<Map.Entry<String, Int>> { it.value }.thenBy { it.key })
            .map { TallyRow(it.key, it.value) }

        if (sorted.size <= maxRows) return sorted

        val shown = sorted.take(maxRows - 1)
        val rest = sorted.drop(maxRows - 1)
        return shown + TallyRow("+${rest.size} more", rest.sumOf { it.count })
    }
}
