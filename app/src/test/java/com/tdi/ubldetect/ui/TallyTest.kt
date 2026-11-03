package com.tdi.ubldetect.ui

import com.tdi.ubldetect.imaging.RectF
import com.tdi.ubldetect.model.Detection
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * What is on the shelf, and how many of each.
 *
 * The boxes already say where things are; counting them by eye off a live
 * preview is what nobody can do. The order has to be stable between frames:
 * a table that reshuffles every 400 ms cannot be read at all.
 */
class TallyTest {

    private fun det(cls: Int) = Detection(RectF(0f, 0f, 1f, 1f), cls, 0.9f)

    private val names = mapOf(1 to "dove_mask_25", 2 to "sunsilk_serum_25", 3 to "lux_soap")
    private val labeller: (Int) -> String = { names[it] ?: "class $it" }

    @Test
    fun countsEachKindOnce() {
        val rows = Tally.of(listOf(det(1), det(1), det(2)), labeller, 8)
        assertEquals(listOf(TallyRow("dove_mask_25", 2), TallyRow("sunsilk_serum_25", 1)), rows)
    }

    @Test
    fun theCommonestComesFirst() {
        val rows = Tally.of(listOf(det(2), det(1), det(1), det(1), det(2)), labeller, 8)
        assertEquals("dove_mask_25", rows[0].label)
        assertEquals(3, rows[0].count)
    }

    @Test
    fun equalCountsAreOrderedByNameSoTheTableDoesNotReshuffle() {
        // Detection order changes frame to frame. If ties followed it, two
        // products with one facing each would swap places constantly.
        val a = Tally.of(listOf(det(1), det(3)), labeller, 8)
        val b = Tally.of(listOf(det(3), det(1)), labeller, 8)
        assertEquals(a, b)
        assertEquals("dove_mask_25", a[0].label)
    }

    @Test
    fun detectionsSharingALabelAreOneRow() {
        // Several model classes can carry one display name: a shelf talker is
        // detected in four pieces, all called the same thing. The table counts
        // what a person would call one item.
        val rows = Tally.of(listOf(det(1), det(2)), { "Hair Care Shelf Talker" }, 8)
        assertEquals(listOf(TallyRow("Hair Care Shelf Talker", 2)), rows)
    }

    @Test
    fun nothingDetectedIsAnEmptyTableNotAnEmptyRow() {
        assertTrue(Tally.of(emptyList(), labeller, 8).isEmpty())
    }

    @Test
    fun beyondTheRowLimitTheRestIsSummarised() {
        // A crowded shelf would otherwise cover the preview, which is the
        // thing being looked at.
        val many = (1..10).flatMap { cls -> List(cls) { det(cls) } }
        val rows = Tally.of(many, { "class $it" }, 3)
        assertEquals(3, rows.size)
        assertEquals(TallyRow("class 10", 10), rows[0])
        assertEquals(TallyRow("class 9", 9), rows[1])
        // 1 + 2 + ... + 8 across 8 kinds that did not fit
        assertEquals(TallyRow("+8 more", 36), rows[2])
    }

    @Test
    fun exactlyTheRowLimitIsNotSummarised() {
        val rows = Tally.of(listOf(det(1), det(2), det(3)), labeller, 3)
        assertEquals(3, rows.size)
        assertTrue(rows.none { it.label.endsWith("more") })
    }

    @Test
    fun oneKindOverTheLimitStillSaysHowManyKindsWereFolded() {
        val rows = Tally.of(listOf(det(1), det(2), det(3)), labeller, 2)
        assertEquals(TallyRow("+2 more", 2), rows[1])
    }

    @Test
    fun theTotalIsPreservedHoweverTheRowsFall() {
        val many = (1..10).flatMap { cls -> List(cls) { det(cls) } }
        for (limit in 1..12) {
            assertEquals(
                "limit $limit",
                many.size,
                Tally.of(many, { "class $it" }, limit).sumOf { it.count },
            )
        }
    }
}
