package com.tdi.ubldetect.ui

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
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
import com.tdi.ubldetect.infer.DelegateKind
import com.tdi.ubldetect.infer.ModelAssetMissing
import com.tdi.ubldetect.infer.TfliteRunner
import com.tdi.ubldetect.pipeline.PosmPipeline

/**
 * The live screen: preview underneath, boxes and labels on top.
 *
 * The preview fills the view and is deliberately not inset, so it runs edge
 * to edge behind the system bars. The overlayView above it draws its own status
 * line and is inset, which is why insets are applied per view rather than to
 * the window.
 */
class CameraActivity : ComponentActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var overlayView: OverlayView
    private lateinit var entry: CatalogEntry

    private var frameSource: FrameSource? = null
    private var runner: TfliteRunner? = null
    private var pipeline: PosmPipeline? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val mode = Mode.valueOf(intent.getStringExtra(EXTRA_MODE) ?: Mode.POSM.name)
        val key = intent.getStringExtra(EXTRA_KEY).orEmpty()

        val catalog = Catalog.parse(assets.open("catalog.json").bufferedReader().readText())
        entry = catalog.entries(mode).firstOrNull { it.key == key } ?: run {
            Toast.makeText(this, "Unknown selection: $key", Toast.LENGTH_LONG).show()
            finish()
            return
        }

        previewView = PreviewView(this).apply {
            scaleType = PreviewView.ScaleType.FILL_CENTER
        }
        overlayView = OverlayView(this)

        setContentView(
            FrameLayout(this).apply {
                addView(previewView)
                addView(overlayView)
            },
        )
        // The preview stays full bleed; only the overlayView's chrome is inset.
        overlayView.padForSystemBars()

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

    private fun startCamera() {
        val asset = "models/posm_${IMGSZ}.tflite"
        val r = try {
            TfliteRunner(this, asset, DelegateKind.GPU)
        } catch (e: ModelAssetMissing) {
            // Review Focus 5: a resolution whose file was never shipped is a
            // message, not a crash.
            Toast.makeText(this, "Model not bundled: ${e.assetPath}", Toast.LENGTH_LONG).show()
            finish()
            return
        }
        runner = r
        pipeline = PosmPipeline(r, entry, SCORE_THRESHOLD)
        Toast.makeText(this, "${entry.key} on ${r.activeDelegate}", Toast.LENGTH_SHORT).show()

        val source = FrameSource(this, this, previewView) { frame, timestampNs ->
            val result = pipeline?.process(frame, timestampNs, useAllowlist = true)
            runOnUiThread {
                if (overlayView.width == 0) return@runOnUiThread
                overlayView.submit(
                    result,
                    PreviewMapping.fillCenter(frame.width, frame.height, overlayView.width, overlayView.height),
                ) { classId -> labelFor(classId) }
            }
        }
        frameSource = source
        source.start()

        overlayView.setOnClickListener {
            if (overlayView.state.frozen) overlayView.unfreeze() else overlayView.freeze()
        }
    }

    private fun labelFor(classId: Int): String {
        val catalog = Catalog(listOf(entry), listOf(entry))
        return catalog.labelFor(entry, classId)
    }

    override fun onDestroy() {
        super.onDestroy()
        frameSource?.stop()
        runner?.close()
    }

    companion object {
        const val EXTRA_MODE = "mode"
        const val EXTRA_KEY = "key"

        private const val IMGSZ = 640
        /** The server's posm.confidence, copied rather than re-tuned. */
        private const val SCORE_THRESHOLD = 0.30f
    }
}
