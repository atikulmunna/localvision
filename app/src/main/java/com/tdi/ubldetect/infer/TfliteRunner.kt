package com.tdi.ubldetect.infer

import android.content.Context
import android.util.Log
import java.io.Closeable
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate

/** Which accelerator the interpreter actually ended up using. */
enum class DelegateKind { GPU, CPU_XNNPACK }

/**
 * The selected model file is not in the APK.
 *
 * Its own type because it is a reachable situation, not a bug: the resolution
 * selector offers sizes, and only the exported ones are shipped. The UI turns
 * this into a message rather than a crash.
 */
class ModelAssetMissing(val assetPath: String) :
    Exception("model asset not found: $assetPath")

/**
 * Loads one exported model and runs it.
 *
 * Three things here are deliberate rather than incidental.
 *
 * The model is memory mapped straight out of the APK rather than read into a
 * byte array. These files are around 57 MB each, so copying one onto the heap
 * to hand to the interpreter doubles its footprint for no benefit. That is
 * also why `noCompress += "tflite"` is set in the build file: a compressed
 * asset cannot be mapped.
 *
 * The GPU delegate is attempted and then given up on, rather than required. A
 * driver that refuses it should cost speed, not the session, so the runner
 * falls back to XNNPACK and records which one won in [activeDelegate] so the
 * UI can show it.
 *
 * Output buffers are allocated once and reused. This runs several times a
 * second; a 300 by 6 allocation per frame is needless churn.
 */
class TfliteRunner(
    context: Context,
    private val assetPath: String,
    preferred: DelegateKind,
    gpuProbe: () -> Boolean = { CompatibilityList().isDelegateSupportedOnThisDevice },
) : Closeable {

    val activeDelegate: DelegateKind
    private val interpreter: Interpreter
    private val gpuDelegate: GpuDelegate?

    init {
        val model = mapModel(context, assetPath)

        var delegate: GpuDelegate? = null
        var chosen = DelegateKind.CPU_XNNPACK
        var built: Interpreter? = null

        if (preferred == DelegateKind.GPU && runCatching(gpuProbe).getOrDefault(false)) {
            built = runCatching {
                val compat = CompatibilityList()
                delegate = GpuDelegate(compat.bestOptionsForThisDevice)
                Interpreter(model, Interpreter.Options().addDelegate(delegate))
            }.onFailure {
                Log.w(TAG, "GPU delegate unavailable for $assetPath, falling back to CPU", it)
                delegate?.close()
                delegate = null
            }.getOrNull()
            if (built != null) chosen = DelegateKind.GPU
        }

        if (built == null) {
            built = Interpreter(
                model,
                Interpreter.Options().setUseXNNPACK(true).setNumThreads(CPU_THREADS),
            )
        }

        interpreter = built
        gpuDelegate = delegate
        activeDelegate = chosen
    }

    fun inputShape(): IntArray = interpreter.getInputTensor(0).shape()

    /** The square side the model expects. Input is NCHW, so it is the last axis. */
    val inputSize: Int get() = inputShape().last()

    fun outputShapes(): List<IntArray> =
        List(interpreter.outputTensorCount) { interpreter.getOutputTensor(it).shape() }

    private val outBuffers: List<ByteBuffer> by lazy {
        List(interpreter.outputTensorCount) { i ->
            ByteBuffer.allocateDirect(interpreter.getOutputTensor(i).numBytes())
                .order(ByteOrder.nativeOrder())
        }
    }

    /**
     * Runs one inference, filling the caller's arrays in output-tensor order.
     *
     * The caller owns those arrays and reuses them between frames, for the
     * same reason the RGB buffer is reused.
     */
    fun run(input: ByteBuffer, outputs: List<FloatArray>) {
        input.rewind()
        val outMap = HashMap<Int, Any>(outBuffers.size)
        outBuffers.forEachIndexed { i, b ->
            b.rewind()
            outMap[i] = b
        }
        interpreter.runForMultipleInputsOutputs(arrayOf<Any>(input), outMap)
        outputs.forEachIndexed { i, dst ->
            outBuffers[i].rewind()
            outBuffers[i].asFloatBuffer().get(dst)
        }
    }

    override fun close() {
        interpreter.close()
        gpuDelegate?.close()
    }

    private companion object {
        const val TAG = "TfliteRunner"
        const val CPU_THREADS = 4

        /**
         * Maps the asset without copying it.
         *
         * A missing asset surfaces from `openFd` as an IOException, which is
         * indistinguishable at the call site from a dozen other failures, so
         * it is translated into [ModelAssetMissing] here where the cause is
         * still known.
         */
        fun mapModel(context: Context, assetPath: String): ByteBuffer {
            val afd = try {
                context.assets.openFd(assetPath)
            } catch (e: IOException) {
                throw ModelAssetMissing(assetPath)
            }
            afd.use {
                FileInputStream(it.fileDescriptor).use { stream ->
                    return stream.channel.map(
                        FileChannel.MapMode.READ_ONLY,
                        it.startOffset,
                        it.declaredLength,
                    )
                }
            }
        }
    }
}
