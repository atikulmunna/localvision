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
    fun labelForKnownClassIdReturnsDisplayName() {
        val c = Catalog.parse(fixture("catalog_min.json"))
        assertEquals("Standard Horlicks Jar", c.labelFor(c.entries(Mode.QPDS)[0], 3))
    }

    @Test
    fun labelForClassIdOutsideTheEntryFallsBackToTheRawIndex() {
        // With the allowlist toggled off the app draws classes the chosen slab
        // never expected, and they still need something on the label.
        val c = Catalog.parse(fixture("catalog_min.json"))
        assertEquals("class 41", c.labelFor(c.entries(Mode.QPDS)[0], 41))
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
        assertEquals(21, c.entries(Mode.QPDS).size)
        assertEquals(8, c.entries(Mode.POSM).size)

        // Every slab resolves at least one class after the `:size` key fix.
        assertTrue(c.entries(Mode.QPDS).none { it.isEmpty })

        // mini_meals is genuinely unlabelable: the model has no class for its
        // one material. A real case for the picker's refusal path.
        val miniMeals = c.entries(Mode.POSM).first { it.key == "mini_meals" }
        assertTrue(miniMeals.isEmpty)
    }
}
