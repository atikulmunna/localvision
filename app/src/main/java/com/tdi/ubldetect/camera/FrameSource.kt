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
import java.util.concurrent.TimeUnit
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

    /**
     * Asks for a full HD preview.
     *
     * Without this CameraX picks a preview resolution on its own, and when
     * another use case is bound it tends to choose conservatively, which is
     * why the preview looked softer than the phone's own camera app. The
     * analysis stream stays at [ANALYSIS_TARGET]; they are separate streams
     * and only the analysis one is paid for per frame in Kotlin.
     */
    private fun previewResolution(): ResolutionSelector = ResolutionSelector.Builder()
        .setAspectRatioStrategy(AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY)
        .setResolutionStrategy(
            ResolutionStrategy(
                PREVIEW_TARGET,
                ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER,
            ),
        )
        .build()

    fun start() {
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            val cameraProvider = future.get()
            provider = cameraProvider

            val preview = Preview.Builder()
                .setResolutionSelector(previewResolution())
                .build()
                .also { it.surfaceProvider = previewView.surfaceProvider }

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

    /**
     * Stops delivering frames and releases the models on the analysis thread.
     *
     * [releaseModels] is queued rather than run by the caller, and that is
     * what makes teardown safe. `shutdown()` only refuses new work; it
     * returns while a frame is still inside the analyzer, and closing an
     * interpreter out from under running native code segfaulted the process
     * (SIGSEGV in libtensorflowlite_jni.so) every time someone left the
     * camera screen.
     *
     * Waiting was the first fix, and it was not enough. The wait was sized
     * for a frame, at roughly 400 ms, but [post] puts model loads on this
     * same thread and each one costs about 3.5 s, so QPDS spends some 7 s
     * opening two. Leaving during a swap outlasted any wait short enough to
     * keep off an ANR, and the activity then closed a runner list the
     * loading thread had not filled in yet: both new interpreters stayed
     * open for the life of the process.
     *
     * Queueing removes the race instead of outrunning it. The release sits
     * behind whatever is already on this thread, so it cannot overlap a
     * frame or a load, whether or not the wait below expires.
     *
     * The wait is now only politeness: it keeps the usual case finished by
     * the time the activity is gone. Expiring is no longer a hazard, because
     * nothing the caller does afterwards touches the models.
     */
    fun stop(releaseModels: () -> Unit = {}) {
        provider?.unbindAll()
        // A second stop() would re-queue the release; the executor refuses it
        // once shut down, and TfliteRunner.close() is idempotent regardless.
        if (!executor.isShutdown) executor.execute(releaseModels)
        executor.shutdown()
        val finished = runCatching {
            executor.awaitTermination(SHUTDOWN_WAIT_SECONDS, TimeUnit.SECONDS)
        }.getOrDefault(false)
        if (!finished) {
            Log.i(TAG, "models still releasing after ${SHUTDOWN_WAIT_SECONDS}s")
        }
    }

    private companion object {
        const val TAG = "FrameSource"

        /**
         * How long teardown waits for the analysis thread to drain.
         *
         * A frame costs roughly 400 ms, so this covers one comfortably. It
         * deliberately does not cover a model load: that would be 7 s on the
         * UI thread, well past the ANR limit, and the release no longer needs
         * the wait in order to be correct.
         */
        const val SHUTDOWN_WAIT_SECONDS = 3L

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

        /**
         * What the person sees. Higher than the analysis stream on purpose:
         * the preview is composited by the display pipeline and costs us
         * nothing per frame, while every analysis pixel is converted in
         * Kotlin on the way to the model.
         */
        val PREVIEW_TARGET = Size(1920, 1080)
    }
}
