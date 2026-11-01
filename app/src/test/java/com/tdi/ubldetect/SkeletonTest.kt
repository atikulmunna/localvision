package com.tdi.ubldetect

import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * Proves the JVM unit test source set runs at all.
 *
 * Most of this app's logic (letterbox geometry, decoders, crop building,
 * catalog parsing) is deliberately free of Android types so it can be tested
 * here rather than on a device. That only pays off if this source set works,
 * so it gets its own test.
 */
class SkeletonTest {
    @Test
    fun jvmUnitTestsRun() {
        assertEquals(4, 2 + 2)
    }
}
