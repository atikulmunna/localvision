package com.tdi.ubldetect.infer

import android.content.pm.PackageManager
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test
import org.junit.runner.RunWith

/**
 * The interpreter, on the real device, against the real exported models.
 *
 * Shapes here are copied from docs/model-contract.md, which was measured from
 * the exported files and checked against the .pt reference. They are not
 * assumptions: the default export produces a different output entirely
 * (the raw [1, 52, 8400] head) and would need NMS.
 */
@RunWith(AndroidJUnit4::class)
class TfliteRunnerTest {

    private val ctx = InstrumentationRegistry.getInstrumentation().targetContext

    private fun fixtureInput(size: Int): ByteBuffer =
        ByteBuffer.allocateDirect(size * size * 3 * 4).order(ByteOrder.nativeOrder()).apply {
            val grey = 114f / 255f // the letterbox pad colour, a valid input everywhere
            repeat(size * size * 3) { putFloat(grey) }
            rewind()
        }

    @Test
    fun loadsPosmModelAndReportsContractShapes() {
        TfliteRunner(ctx, "models/posm_960.tflite", DelegateKind.CPU_XNNPACK).use { r ->
            assertEquals(960, r.inputSize)
            assertArrayEquals(intArrayOf(1, 3, 960, 960), r.inputShape())
            assertArrayEquals(intArrayOf(1, 300, 6), r.outputShapes()[0])
        }
    }

    @Test
    fun loadsThe640ModelToo() {
        TfliteRunner(ctx, "models/posm_640.tflite", DelegateKind.CPU_XNNPACK).use { r ->
            assertEquals(640, r.inputSize)
            assertArrayEquals(intArrayOf(1, 300, 6), r.outputShapes()[0])
        }
    }

    @Test
    fun assetExistsAnswersBeforeAnythingIsLoaded() {
        // Lets a caller decide whether a swap is possible before tearing down
        // the model it already has. Without it, swapping resolution means
        // holding two 57 MB models at once, which on ten swaps left 678 MB
        // of mapped pages resident until the system asked for them back.
        assertTrue(TfliteRunner.assetExists(ctx, "models/posm_640.tflite"))
        assertFalse(TfliteRunner.assetExists(ctx, "models/posm_1280.tflite"))
    }

    @Test
    fun missingAssetThrowsModelAssetMissingNotAGenericCrash() {
        // Review Focus 5: the person picks a resolution whose file was not
        // shipped. That has to be a message, not a stack trace.
        assertThrows(ModelAssetMissing::class.java) {
            TfliteRunner(ctx, "models/posm_1280.tflite", DelegateKind.CPU_XNNPACK)
        }
    }

    @Test
    fun gpuUnavailableFallsBackToCpuRatherThanThrowing() {
        // Review Focus 5: a device or driver that refuses the GPU delegate must
        // downgrade and say which delegate actually won, not fail the run.
        TfliteRunner(
            ctx,
            "models/posm_640.tflite",
            DelegateKind.GPU,
            gpuProbe = { false },
        ).use { r ->
            assertEquals(DelegateKind.CPU_XNNPACK, r.activeDelegate)
        }
    }

    @Test
    fun runFillsOutputsAndIsRepeatable() {
        TfliteRunner(ctx, "models/posm_640.tflite", DelegateKind.CPU_XNNPACK).use { r ->
            val a = FloatArray(1 * 300 * 6)
            val b = FloatArray(1 * 300 * 6)
            r.run(fixtureInput(640), listOf(a))
            r.run(fixtureInput(640), listOf(b))
            assertArrayEquals(a, b, 0f)
        }
    }

    @Test
    fun outputRowsAreSortedByScoreDescending() {
        // A property the decoder relies on to stop early instead of walking all
        // 300 slots. Measured in docs/model-contract.md; asserted here so a
        // future re-export cannot quietly drop it.
        TfliteRunner(ctx, "models/posm_640.tflite", DelegateKind.CPU_XNNPACK).use { r ->
            val out = FloatArray(1 * 300 * 6)
            r.run(fixtureInput(640), listOf(out))
            for (i in 1 until 300) {
                val prev = out[(i - 1) * 6 + 4]
                val cur = out[i * 6 + 4]
                assert(cur <= prev + 1e-6f) { "row $i scored $cur after $prev" }
            }
        }
    }

    @Test
    fun theAppShipsTheCameraPermissionAndNothingElse() {
        // Guards the Task 3 finding permanently. The LiteRT GPU delegate's
        // manifest has no targetSdkVersion, so the merger legacy-implies
        // READ_PHONE_STATE and storage permissions unless they are explicitly
        // removed. A build-time grep catches that once; this catches it
        // forever, including when a new dependency reintroduces it.
        // Only platform permissions are judged. AGP also auto-adds a signature
        // permission namespaced to this package
        // (com.tdi.ubldetect.DYNAMIC_RECEIVER_NOT_EXPORTED_PERMISSION) for
        // runtime receivers on SDK 33 and above. It grants nothing to anyone
        // else and is not something this app chose.
        val info = ctx.packageManager.getPackageInfo(ctx.packageName, PackageManager.GET_PERMISSIONS)
        val platform = (info.requestedPermissions ?: emptyArray())
            .filter { it.startsWith("android.permission.") }
            .sorted()
        assertEquals(listOf("android.permission.CAMERA"), platform)
    }
}
