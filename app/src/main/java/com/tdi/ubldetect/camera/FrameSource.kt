package com.tdi.ubldetect.camera

import android.content.Context
import android.util.Log
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
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
    private val onFrame: (RgbFrame, Long) -> Unit,
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

            val frame = YuvToRgb.convert(planes, rotation, buffer)
            frameWidth = frame.width
            frameHeight = frame.height
            onFrame(frame, image.imageInfo.timestamp)
        } catch (t: Throwable) {
            Log.e(TAG, "frame dropped after an error", t)
        } finally {
            image.close()
            busy.set(false)
        }
    }

    fun stop() {
        provider?.unbindAll()
        executor.shutdown()
    }

    private companion object {
        const val TAG = "FrameSource"
    }
}
