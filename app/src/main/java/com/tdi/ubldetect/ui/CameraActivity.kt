package com.tdi.ubldetect.ui

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
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
import com.tdi.ubldetect.imaging.PreviewMapping
import com.tdi.ubldetect.imaging.RgbFrame
import com.tdi.ubldetect.infer.ModelAssetMissing
import com.tdi.ubldetect.infer.TfliteRunner
import com.tdi.ubldetect.model.FrameResult
import com.tdi.ubldetect.pipeline.PosmPipeline

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
    private var runner: TfliteRunner? = null
    private var pipeline: PosmPipeline? = null

    @Volatile
    private var state = ControlsState()

    /** Kept for the save button, which acts on what is currently on screen. */
    @Volatile
    private var lastFrame: RgbFrame? = null

    @Volatile
    private var lastResult: FrameResult? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val mode = Mode.valueOf(intent.getStringExtra(EXTRA_MODE) ?: Mode.POSM.name)
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

        if (mode == Mode.QPDS) {
            Toast.makeText(this, "QPDS arrives in a later phase", Toast.LENGTH_LONG).show()
            finish()
            return
        }

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
     * Swaps in the model for [s], one model resident at a time.
     *
     * The order matters. Opening the new model before closing the old keeps
     * two 57 MB mapped files alive through every swap; ten resolution
     * changes measured 678 MB of mapped pages still resident, released only
     * when the system asked for memory back. So the asset is checked first,
     * the old model is released, and only then is the new one opened.
     *
     * @return false when the model for this state is not bundled
     */
    private fun buildPipeline(s: ControlsState): Boolean {
        val asset = s.assetFor("posm")
        if (!TfliteRunner.assetExists(this, asset)) {
            // Review Focus 5: a resolution whose file was never shipped is a
            // message, not a crash and not a silent no-op. Checked before
            // anything is torn down, so the running model keeps running.
            toast("Model not bundled: $asset")
            return false
        }

        pipeline = null
        runner?.close()
        runner = null

        val next = try {
            TfliteRunner(this, asset, s.delegate)
        } catch (e: ModelAssetMissing) {
            toast("Model not bundled: ${e.assetPath}")
            return false
        }
        runner = next
        pipeline = PosmPipeline(next, entry, s.scoreThreshold)
        toast("${s.imgsz} on ${next.activeDelegate}")
        return true
    }

    private fun startCamera() {
        if (!buildPipeline(state)) {
            finish()
            return
        }

        val source = FrameSource(this, this, previewView) { frame, timestampNs, convertMs ->
            // Threshold is read per frame, so the slider takes effect on the
            // next frame without touching the interpreter.
            val s = state
            val result = pipeline?.process(frame, timestampNs, s.useAllowlist, mapOf("yuv" to convertMs))
            lastFrame = frame
            lastResult = result
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
                ) { classId -> catalog.labelFor(entry, classId) }
            }
        }
        frameSource = source
        source.start()
    }

    /** Safe to call from the analysis thread as well as the UI thread. */
    private fun toast(text: String) {
        runOnUiThread { Toast.makeText(this, text, Toast.LENGTH_SHORT).show() }
    }

    private fun onSave() {
        val frame = lastFrame
        if (frame == null) {
            Toast.makeText(this, "Nothing captured yet", Toast.LENGTH_SHORT).show()
            return
        }
        val file = FrameSaver(this).save(frame, lastResult) { catalog.labelFor(entry, it) }
        Toast.makeText(this, "Saved ${file.name}", Toast.LENGTH_SHORT).show()
    }

    override fun onDestroy() {
        super.onDestroy()
        frameSource?.stop()
        runner?.close()
    }

    companion object {
        const val EXTRA_MODE = "mode"
        const val EXTRA_KEY = "key"
    }
}
