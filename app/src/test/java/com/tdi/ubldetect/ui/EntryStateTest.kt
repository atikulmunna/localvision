package com.tdi.ubldetect.ui

import com.tdi.ubldetect.catalog.CatalogEntry
import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * How a slab or category presents itself before you tap it.
 *
 * The picker knows, from the catalog's reconciliation, whether an entry can
 * work at all. Three states, and the distinction is worth drawing because
 * each wants different handling:
 *
 * - **READY**: everything it expects is in the model.
 * - **PARTIAL**: some classes are missing, so it will work but find less
 *   than the standards describe. Expected, since new SKUs ship in config
 *   before a model can detect them, so it is a note rather than an alarm.
 * - **UNUSABLE**: nothing it expects is in the model. Opening a camera on
 *   this shows an empty screen that looks like a broken detector.
 *
 * Previously the picker let you tap an unusable entry and then refused,
 * which teaches you nothing until after you have tried.
 */
class EntryStateTest {

    private fun entry(
        key: String = "X",
        classIds: List<Int> = listOf(1, 2),
        missing: List<String> = emptyList(),
    ) = CatalogEntry(
        key = key,
        subtitle = "sub",
        classIdList = classIds,
        display = classIds.associate { it.toString() to "name$it" },
        missingFromModel = missing,
    )

    @Test
    fun everythingPresentIsReady() {
        assertEquals(EntryState.READY, EntryState.of(entry()))
    }

    @Test
    fun someClassesMissingIsPartial() {
        assertEquals(EntryState.PARTIAL, EntryState.of(entry(missing = listOf("a", "b"))))
    }

    @Test
    fun nothingDetectableIsUnusable() {
        // The live case: mini_meals expects one material the model lacks.
        assertEquals(
            EntryState.UNUSABLE,
            EntryState.of(entry(classIds = emptyList(), missing = listOf("knorr_soup"))),
        )
    }

    @Test
    fun nothingDetectableAndNothingMissingIsStillUnusable() {
        // An empty entry is unusable whatever the reason; the picker must not
        // open a camera that can never label.
        assertEquals(EntryState.UNUSABLE, EntryState.of(entry(classIds = emptyList())))
    }

    @Test
    fun onlyReadyAndPartialCanBeOpened() {
        assertEquals(true, EntryState.READY.canOpen)
        assertEquals(true, EntryState.PARTIAL.canOpen)
        assertEquals(false, EntryState.UNUSABLE.canOpen)
    }
}
