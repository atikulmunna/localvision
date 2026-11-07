package com.tdi.ubldetect.catalog

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The catalog is everything the device knows about UBL's standards.
 *
 * It is generated at build time by tools/gen_catalog.py, so no compliance
 * logic and no YAML reach the phone. These tests cover the reading end of
 * that contract; tools/tests/test_gen_catalog.py covers the writing end.
 */
class CatalogTest {

    private fun fixture(name: String): String =
        checkNotNull(javaClass.classLoader?.getResourceAsStream(name)) { "missing $name" }
            .bufferedReader().readText()

    @Test
    fun parsesBothModes() {
        val c = Catalog.parse(fixture("catalog_min.json"))
        assertEquals(2, c.entries(Mode.QPDS).size)
        assertEquals(1, c.entries(Mode.POSM).size)
    }

    @Test
    fun labelForClassIdOutsideTheEntryUsesTheModelsOwnName() {
        // With the allowlist off the app draws classes the chosen slab never
        // expected. Falling back to the bare index meant the screen read
        // "class 41  100%" while the model had correctly identified a Dove
        // hair mask: right answer, unreadable.
        val c = Catalog.parse(fixture("catalog_min.json"))
        assertEquals("dove_mask_25", c.labelFor(Mode.QPDS, c.entries(Mode.QPDS)[0], 41))
    }

    @Test
    fun theEntrysOwnDisplayNameWinsOverTheModelsClassName() {
        // Within the chosen slab the human-facing name is better than the
        // model's internal one.
        val c = Catalog.parse(fixture("catalog_min.json"))
        assertEquals(
            "Standard Horlicks Jar",
            c.labelFor(Mode.QPDS, c.entries(Mode.QPDS)[0], 3),
        )
    }

    @Test
    fun eachModeLooksUpItsOwnModelsClassNames() {
        // POSM class 41 and QPDS class 41 are different things entirely, so
        // the fallback has to know which model produced the detection.
        val c = Catalog.parse(fixture("catalog_min.json"))
        assertEquals("ps_gal_sticker", c.labelFor(Mode.POSM, c.entries(Mode.POSM)[0], 41))
        assertEquals("dove_mask_25", c.labelFor(Mode.QPDS, c.entries(Mode.QPDS)[0], 41))
    }

    @Test
    fun aClassNeitherTheEntryNorTheModelKnowsStillGetsALabel() {
        val c = Catalog.parse(fixture("catalog_min.json"))
        assertEquals("class 999", c.labelFor(Mode.QPDS, c.entries(Mode.QPDS)[0], 999))
    }

    @Test
    fun entryWithNoResolvableClassesIsFlaggedEmpty() {
        // Review Focus 4: the picker must be able to warn rather than start a
        // camera that can never label anything.
        val c = Catalog.parse(fixture("catalog_min.json"))
        val empty = c.entries(Mode.QPDS).first { it.key == "ALL_MISSING" }
        assertTrue(empty.isEmpty)
        assertEquals(listOf("horlicks_lite"), empty.missingFromModel)
    }

    @Test
    fun entryWithClassesIsNotFlaggedEmpty() {
        val c = Catalog.parse(fixture("catalog_min.json"))
        assertFalse(c.entries(Mode.QPDS).first { it.key == "DOUBLE_SHELF" }.isEmpty)
    }

    @Test
    fun classIdsAreExposedAsASetForFiltering() {
        // the decoders filter per detection, so this is a membership test, not a list scan
        val c = Catalog.parse(fixture("catalog_min.json"))
        val slab = c.entries(Mode.QPDS).first { it.key == "DOUBLE_SHELF" }
        assertTrue(3 in slab.classIds)
        assertFalse(41 in slab.classIds)
    }

    /**
     * Reads the real generated asset, not a fixture.
     *
     * The fixture pins the parser's behaviour; this pins the parser against
     * what tools/gen_catalog.py actually emits from the live standards. A
     * change to either side that breaks the other shows up here rather than
     * on the device.
     */
    @Test
    fun parsesTheRealGeneratedCatalog() {
        val asset = File("src/main/assets/catalog.json")
        assertTrue("expected the generated catalog at ${asset.absolutePath}", asset.isFile)

        val c = Catalog.parse(asset.readText())
        assertEquals(22, c.entries(Mode.QPDS).size)
        assertEquals(8, c.entries(Mode.POSM).size)

        // Every slab can label something now. WINTER_QPDS could not when it
        // arrived with the Nov'26 standards, because the Vaseline classes it
        // wants had not been trained yet; the 59 class QPDS-cls of 2026-11-06
        // added all three. The picker's refusal path is still exercised by
        // mini_meals, which wants a Knorr sticker POSM cannot detect.
        assertEquals(
            emptyList<String>(),
            c.entries(Mode.QPDS).filter { it.isEmpty }.map { it.key },
        )
        assertTrue(c.entries(Mode.POSM).first { it.key == "mini_meals" }.isEmpty)

        // The three classes that made the difference, at the indices the
        // classifier gives them. They were appended rather than inserted, so
        // no existing slab's class ids moved.
        val winter = c.entries(Mode.QPDS).first { it.key == "WINTER_QPDS" }
        assertEquals(setOf(56, 57, 58), winter.classIds)
        assertEquals("Vaseline Body Lotion Cocoa Glow", c.labelFor(Mode.QPDS, winter, 56))

        // Every model class has a name, so nothing on screen can read
        // "class 54" while the model knows it is a Tresemme hair mask.
        val anySlab = c.entries(Mode.QPDS).first()
        assertEquals("tresemme_mask_25", c.labelFor(Mode.QPDS, anySlab, 54))
        assertEquals("dove_mask_25", c.labelFor(Mode.QPDS, anySlab, 12))
    }
}
