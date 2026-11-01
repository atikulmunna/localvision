package com.tdi.ubldetect.camera

import android.content.Context
import android.util.Log
import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.tdi.ubldetect.imaging.RgbFrame
import com.tdi.ubldetect.imaging.YuvPlanes
import com.tdi.ubldetect.imaging.YuvToRgb
import java.nio.ByteBuffer
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Feeds camera frames to a detector without letting it hold up the preview.
 *
 * The preview runs at the camera's own rate on its own surface. Analysis is a
 * separate use case with `KEEP_ONLY_LATEST`, and on top of that a single
 * flight guard drops any frame that arrives while one is still being
 * processed. Queueing instead would make latency compound: with inference
 * slower than the frame rate, a queue only grows, and the overlay ends up
 * describing a shelf the phone was pointed at seconds ago.
 *
 * Dropping frames is the point, not a limitation.
 */
class FrameSource(
    private val context: Context,
    private val owner: LifecycleOwner,
    private val previewView: PreviewView,
    /** @param convertMs cost of YUV to upright RGB, which scales with the
     *   analysis resolution and is invisible in the pipeline's own timings. */
    private val onFrame: (RgbFrame, Long, Long) -> Unit,
) {
    private val executor = Executors.newSingleThreadExecutor()
    private val busy = AtomicBoolean(false)
    private var rgb: ByteBuffer? = null
    private var provider: ProcessCameraProvider? = null

    /** Source frame size, post rotation. Null until the first frame arrives. */
    @Volatile
    var frameWidth: Int = 0
        private set

    @Volatile
    var frameHeight: Int = 0
        private set

    /**
     * Asks for a 16:9 analysis stream at [ANALYSIS_TARGET].
     *
     * Two reasons, and the second is the one that matters.
     *
     * CameraX defaults ImageAnalysis to roughly 640x480. That is a 0.3 MP,
     * 4:3 stream, heavily binned, which is a poorer picture than the sensor
     * can give. It does **not** make detections smaller in the model input,
     * since the letterbox caps the long side at the model size either way,
     * but it does mean those pixels carry less real detail than the
     * full-resolution photographs the server scores.
     *
     * The field of view is the sharper issue. The preview is 16:9 from the
     * sensor; a 4:3 analysis stream is a different crop of it. Boxes are
     * computed against the analysis frame and drawn over the preview, so a
     * mismatch between the two puts every box slightly wrong, in a way that
     * looks like a model problem. Asking both for 16:9 keeps them agreeing.
     *
     * FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER rather than an exact size,
     * because a camera that cannot provide this should give the nearest it
     * has instead of failing to bind.
     */
    private fun analysisResolution(): ResolutionSelector = ResolutionSelector.Builder()
        .setAspectRatioStrategy(AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY)
        .setResolutionStrategy(
            ResolutionStrategy(
                ANALYSIS_TARGET,
                ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER,
            ),
        )
        .build()

    fun start() {
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            val cameraProvider = future.get()
            provider = cameraProvider

            val preview = Preview.Builder().build().also {
                it.surfaceProvider = previewView.surfaceProvider
            }

            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                .setResolutionSelector(analysisResolution())
                .build()
                .also { it.setAnalyzer(executor, ::analyze) }

            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                owner,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                analysis,
            )
        }, ContextCompat.getMainExecutor(context))
    }

    private fun analyze(image: ImageProxy) {
        // Drop rather than queue. See the class comment.
        if (!busy.compareAndSet(false, true)) {
            image.close()
            return
        }
        try {
            val rotation = image.imageInfo.rotationDegrees
            val swap = rotation == 90 || rotation == 270
            val outW = if (swap) image.height else image.width
            val outH = if (swap) image.width else image.height

            val needed = outW * outH * 3
            var buffer = rgb
            if (buffer == null || buffer.capacity() < needed) {
                buffer = ByteBuffer.allocate(needed)
                rgb = buffer
            }

            val planes = YuvPlanes(
                y = image.planes[0].buffer,
                u = image.planes[1].buffer,
                v = image.planes[2].buffer,
                width = image.width,
                height = image.height,
                yRowStride = image.planes[0].rowStride,
                uvRowStride = image.planes[1].rowStride,
                uvPixelStride = image.planes[1].pixelStride,
            )

            val t0 = android.os.SystemClock.elapsedRealtime()
            val frame = YuvToRgb.convert(planes, rotation, buffer)
            val convertMs = android.os.SystemClock.elapsedRealtime() - t0
            frameWidth = frame.width
            frameHeight = frame.height
            onFrame(frame, image.imageInfo.timestamp, convertMs)
        } catch (t: Throwable) {
            Log.e(TAG, "frame dropped after an error", t)
        } finally {
            image.close()
            busy.set(false)
        }
    }

    /**
     * Runs [block] on the analysis thread, never alongside a frame.
     *
     * Swapping the model is the reason this exists. The interpreter is a
     * native object: closing it from the UI thread while this thread is
     * inside `run()` is a use-after-free, and it crashed exactly that way
     * (SIGSEGV under ImageAnalysisAbstractAnalyzer.analyzeImage). The
     * analyzer runs on a single-threaded executor, so work submitted here is
     * serialised against frame processing by construction rather than by a
     * lock someone has to remember to take.
     */
    fun post(block: () -> Unit) {
        if (!executor.isShutdown) executor.execute(block)
    }

    fun stop() {
        provider?.unbindAll()
        executor.shutdown()
    }

    private companion object {
        const val TAG = "FrameSource"

        /**
         * 1280x720, not 1920x1080.
         *
         * The YUV to RGB conversion is O(source pixels) and runs in Kotlin on
         * every frame, so the source size is paid in full each time, while
         * the letterbox into the model input is O(output pixels) and costs
         * the same whatever arrives. 1280x720 is 3x the default's pixels and
         * 1920x1080 is 6.75x, for a picture the model mostly discards on its
         * way down to 640. Measure before raising it further.
         */
        val ANALYSIS_TARGET = Size(1280, 720)
    }
}
