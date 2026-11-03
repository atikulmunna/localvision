package com.tdi.ubldetect.ui

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.view.Gravity
import android.view.ViewGroup
import android.widget.FrameLayout
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.tdi.ubldetect.camera.FrameSource
import com.tdi.ubldetect.catalog.Catalog
import com.tdi.ubldetect.catalog.CatalogEntry
import com.tdi.ubldetect.catalog.Mode
import com.tdi.ubldetect.imaging.FrameSnapshot
import com.tdi.ubldetect.models
import com.tdi.ubldetect.imaging.PreviewMapping
import com.tdi.ubldetect.infer.ModelAssetMissing
import com.tdi.ubldetect.infer.TfliteRunner
import com.tdi.ubldetect.model.FrameResult
import com.tdi.ubldetect.infer.ClsRunner
import com.tdi.ubldetect.pipeline.FramePipeline
import com.tdi.ubldetect.pipeline.PosmPipeline
import com.tdi.ubldetect.pipeline.QpdsPipeline

/**
 * The live screen: preview underneath, boxes and labels on top, controls at
 * the bottom.
 *
 * The preview fills the view and is deliberately not inset, so it runs edge
 * to edge behind the system bars. The overlay and the controls above it are
 * inset, which is why insets are applied per view rather than to the window.
 */
class CameraActivity : ComponentActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var overlayView: OverlayView
    private lateinit var controls: ControlsPanel
    private lateinit var entry: CatalogEntry
    private lateinit var catalog: Catalog

    private var frameSource: FrameSource? = null

    /**
     * Written on the analysis thread, read on the UI thread at teardown.
     *
     * Volatile because [onDestroy] has no other happens-before edge to the
     * loading thread once [FrameSource.stop]'s wait expires. Without it the
     * UI thread could see a stale list and close interpreters the analysis
     * thread had already replaced.
     */
    @Volatile
    private var runners: List<TfliteRunner> = emptyList()

    @Volatile
    private var pipeline: FramePipeline? = null
    private lateinit var mode: Mode

    @Volatile
    private var state = ControlsState()

    /**
     * What is currently on screen, copied, for the save button.
     *
     * A copy rather than the live frame. The camera reuses one RGB buffer, so
     * reading it from the UI thread after the fact produced a saved JPEG that
     * could be two frames spliced together with boxes belonging to neither.
     *
     * It also only updates while the overlay is accepting frames, which is
     * what makes Hold mean something: freeze to read a label, press Save, and
     * the file is the frame that was in front of you rather than whichever
     * one the analysis thread has reached.
     */
    private val onScreen = FrameSnapshot<FrameResult>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Declared before anything is acquired and given back in
        // releaseModels, so memory pressure cannot close an interpreter with
        // a frame inside it.
        models.beginUse()

        mode = Mode.valueOf(intent.getStringExtra(EXTRA_MODE) ?: Mode.POSM.name)
        val key = intent.getStringExtra(EXTRA_KEY).orEmpty()

        catalog = Catalog.parse(assets.open("catalog.json").bufferedReader().readText())
        entry = catalog.entries(mode).firstOrNull { it.key == key } ?: run {
            Toast.makeText(this, "Unknown selection: $key", Toast.LENGTH_LONG).show()
            finish()
            return
        }

        previewView = PreviewView(this).apply {
            scaleType = PreviewView.ScaleType.FILL_CENTER
        }
        overlayView = OverlayView(this)
        controls = ControlsPanel(this, state, ::onControlsChanged, ::onSave).apply {
            layoutParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
                Gravity.BOTTOM,
            )
        }

        setContentView(
            FrameLayout(this).apply {
                addView(previewView)
                addView(overlayView)
                addView(controls)
            },
        )
        // The preview stays full bleed; only the chrome above it is inset.
        overlayView.padForSystemBars(bottom = false)
        controls.padForSystemBars(top = false)

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            requestCamera.launch(Manifest.permission.CAMERA)
        }
    }

    private val requestCamera =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) {
                startCamera()
            } else {
                Toast.makeText(this, "The camera is the whole app", Toast.LENGTH_LONG).show()
                finish()
            }
        }

    /**
     * Rebuilds the model only when the change actually needs it.
     *
     * Dragging the confidence slider must not tear down a 57 MB interpreter
     * on every pixel of movement, and a resolution change must not silently
     * keep running the old one. [ControlsState.requiresReload] draws that
     * line and is unit tested.
     */
    private fun onControlsChanged(next: ControlsState) {
        val previous = state
        state = next

        if (next.frozen != previous.frozen) {
            if (next.frozen) overlayView.freeze() else overlayView.unfreeze()
        }
        overlayView.showBoxes = next.showBoxes

        if (!next.requiresReload(previous)) return

        val source = frameSource
        if (source == null) {
            applySwap(next, previous)
            return
        }
        // On the analysis thread, so the interpreter is never closed while a
        // frame is inside it. Doing this from here crashed natively.
        source.post { applySwap(next, previous) }
    }

    private fun applySwap(next: ControlsState, previous: ControlsState) {
        if (buildPipeline(next)) return
        // Keep running on what still works and tell the panel the truth,
        // rather than showing a setting the app is not honouring.
        state = previous
        buildPipeline(previous)
        runOnUiThread { controls.adopt(previous) }
    }

    /**
     * Swaps in the models for [s], releasing the old ones first.
     *
     * The order matters. Opening before closing keeps two 57 MB mapped files
     * alive through every swap; ten resolution changes measured 678 MB of
     * mapped pages still resident, released only when the system asked for
     * memory back. So the assets are checked first, the old models are
     * released, and only then are the new ones opened.
     *
     * QPDS needs two models, a segmenter and a classifier, and the
     * classifier has one resolution. If either is missing nothing is swapped.
     *
     * @return false when a model for this state is not bundled
     */
    private fun buildPipeline(s: ControlsState): Boolean {
        val startedAt = android.os.SystemClock.elapsedRealtime()
        val assets = s.assetsFor(mode)
        val missing = assets.firstOrNull { !TfliteRunner.assetExists(this, it) }
        if (missing != null) {
            // Review Focus 5: a resolution whose file was never shipped is a
            // message, not a crash and not a silent no-op. Checked before
            // anything is torn down, so the running models keep running.
            toast("Model not bundled: $missing")
            return false
        }

        pipeline = null
        runners = emptyList()

        // From the pool, which keeps what is still wanted and closes the rest
        // before opening anything new. Picking another category in this mode
        // asks for the models it already holds, so nothing is loaded at all.
        val opened: List<TfliteRunner>
        try {
            opened = models.acquire(assets, s.delegate)
        } catch (e: Exception) {
            // Not only ModelAssetMissing. `FileChannel.map` raises
            // IOException and the XNNPACK interpreter raises
            // IllegalArgumentException, and letting either escape from here
            // left the old models already closed, no pipeline, no message,
            // and a live preview that never detected anything again.
            Log.e(TAG, "could not open models for ${s.imgsz}", e)
            toast(
                when (e) {
                    is ModelAssetMissing -> "Model not bundled: ${e.assetPath}"
                    else -> "Could not load model: ${e.message ?: e.javaClass.simpleName}"
                },
            )
            return false
        }
        runners = opened

        pipeline = when (mode) {
            Mode.POSM -> PosmPipeline(opened[0], entry)
            Mode.QPDS -> QpdsPipeline(opened[0], ClsRunner(opened[1]), entry)
        }
        val loadMs = android.os.SystemClock.elapsedRealtime() - startedAt
        Log.i(
            TAG,
            "models ready in ${loadMs}ms (${mode.name} ${s.imgsz} ${opened[0].activeDelegate})",
        )
        toast("${mode.name} ${s.imgsz} on ${opened[0].activeDelegate}")
        return true
    }

    /**
     * Shows the preview first, loads the models second.
     *
     * Loading is not quick: a 57 MB model is memory mapped and the GPU
     * delegate is initialised, and QPDS does that twice. Doing it before
     * starting the camera left the screen blank for a second or two after
     * tapping a category, which reads as the app hanging.
     *
     * So the camera binds immediately and the load is posted to the analysis
     * thread. Frames that arrive before it finishes find no pipeline and are
     * dropped, which costs a few early frames and nothing else. The same
     * thread later runs inference, so the load cannot overlap a frame.
     */
    private fun startCamera() {
        val source = FrameSource(this, this, previewView) { frame, timestampNs, convertMs ->
            // Threshold is read per frame, so the slider takes effect on the
            // next frame without touching the interpreter.
            val s = state
            // Null until the models finish loading; those frames are dropped.
            val result = pipeline?.process(frame, timestampNs, s, mapOf("yuv" to convertMs))
                ?: return@FrameSource
            // Frozen means the overlay is ignoring new frames, so the
            // snapshot has to ignore them too, or Save would write a frame
            // the person never saw. Read now rather than from `s` above: Hold
            // can be pressed while this frame is in flight, and `s` is as the
            // state was 400 ms ago, which would snapshot one frame past what
            // the overlay kept.
            if (!state.frozen) onScreen.take(frame, result)
            runOnUiThread {
                if (overlayView.width == 0) return@runOnUiThread
                overlayView.submit(
                    result,
                    PreviewMapping.fillCenter(
                        frame.width,
                        frame.height,
                        overlayView.width,
                        overlayView.height,
                    ),
                ) { classId -> catalog.labelFor(mode, entry, classId) }
            }
        }
        frameSource = source
        source.start()

        source.post {
            val ok = buildPipeline(state)
            runOnUiThread {
                if (ok) overlayView.loadingMessage = null else finish()
            }
        }
    }

    /** Safe to call from the analysis thread as well as the UI thread. */
    private fun toast(text: String) {
        runOnUiThread { Toast.makeText(this, text, Toast.LENGTH_SHORT).show() }
    }

    /**
     * Writes the frame on screen, on the analysis thread.
     *
     * Posted rather than done here for the same reason the model release is:
     * it is the thread that owns the snapshot. Writing a JPEG is also not
     * work for the UI thread.
     */
    private fun onSave() {
        if (onScreen.taken == null) {
            Toast.makeText(this, "Nothing captured yet", Toast.LENGTH_SHORT).show()
            return
        }
        val source = frameSource
        if (source == null) {
            Toast.makeText(this, "Nothing captured yet", Toast.LENGTH_SHORT).show()
            return
        }
        source.post {
            val taken = onScreen.taken ?: return@post
            val file = FrameSaver(this).save(taken.frame, taken.result) {
                catalog.labelFor(mode, entry, it)
            }
            toast("Saved ${file.name}")
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        // Handed to the analysis thread rather than done here, so the close
        // queues behind any frame or model load still running. Doing it on
        // this thread raced both.
        val source = frameSource
        if (source != null) source.stop(::releaseModels) else releaseModels()
    }

    /**
     * Runs on the analysis thread, the only thread that opens these.
     *
     * Hands the models back rather than closing them. They belong to the
     * pool, which holds them for the next category and releases them when
     * the app stops being looked at.
     */
    private fun releaseModels() {
        pipeline = null
        runners = emptyList()
        models.endUse()
    }

    companion object {
        const val EXTRA_MODE = "mode"
        const val EXTRA_KEY = "key"
        private const val TAG = "CameraActivity"
    }
}
