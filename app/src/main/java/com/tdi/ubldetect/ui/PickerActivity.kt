package com.tdi.ubldetect.ui

import android.app.Activity
import android.content.Intent
import android.graphics.Color
import android.os.Bundle
import android.view.Gravity
import android.view.ViewGroup
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import com.tdi.ubldetect.catalog.Catalog
import com.tdi.ubldetect.catalog.CatalogEntry
import com.tdi.ubldetect.catalog.Mode

/**
 * Pick a mode, then a slab or a category, then start.
 *
 * The choice is not cosmetic. It restricts which classes are allowed, which
 * is both faster to read and more accurate: a classifier choosing between the
 * SKUs a slab actually carries makes fewer mistakes than one choosing between
 * all 56.
 *
 * It is also where reconciliation surfaces. Some entries expect classes the
 * deployed model does not have, and one (mini_meals) expects nothing the
 * model can detect at all. Starting a camera on that would look like a broken
 * detector, so this screen says so instead.
 */
class PickerActivity : Activity() {

    private lateinit var catalog: Catalog
    private lateinit var container: LinearLayout

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        catalog = Catalog.parse(assets.open("catalog.json").bufferedReader().readText())

        val scroll = ScrollView(this)
        container = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(PAD, PAD, PAD, PAD)
        }
        scroll.addView(container)
        setContentView(scroll)
        scroll.padForSystemBars()

        showModes()
    }

    private fun showModes() {
        container.removeAllViews()
        container.addView(heading("What are you pointing at?"))
        container.addView(
            button("POSM", "${catalog.posmCategories.size} categories") { showEntries(Mode.POSM) },
        )
        container.addView(
            button("QPDS", "${catalog.qpdsSlabs.size} slabs, two stage") { showEntries(Mode.QPDS) },
        )
    }

    private fun showEntries(mode: Mode) {
        container.removeAllViews()
        container.addView(heading(if (mode == Mode.POSM) "Category" else "Slab"))
        container.addView(button("Back", "") { showModes() })

        for (entry in catalog.entries(mode)) {
            container.addView(button(entry.key, describe(entry)) { start(mode, entry) })
        }
    }

    private fun describe(entry: CatalogEntry): String = buildString {
        append(entry.subtitle)
        append(", ")
        append(entry.classIds.size)
        append(" detectable")
        if (entry.missingFromModel.isNotEmpty()) {
            append("  (")
            append(entry.missingFromModel.size)
            append(" not in this model)")
        }
    }

    private fun start(mode: Mode, entry: CatalogEntry) {
        if (entry.isEmpty) {
            // Nothing here can ever be labelled. Say which classes are
            // missing rather than opening a camera that will stay blank.
            container.addView(
                heading(
                    "${entry.key} cannot be shown.\n\nThe deployed model has none of its " +
                        "classes:\n" + entry.missingFromModel.joinToString("\n"),
                ),
            )
            return
        }
        startActivity(
            Intent(this, CameraActivity::class.java)
                .putExtra(CameraActivity.EXTRA_MODE, mode.name)
                .putExtra(CameraActivity.EXTRA_KEY, entry.key),
        )
    }

    private fun heading(text: String) = TextView(this).apply {
        this.text = text
        textSize = 20f
        setPadding(0, PAD, 0, PAD)
    }

    private fun button(title: String, subtitle: String, onClick: () -> Unit) =
        LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(PAD, PAD, PAD, PAD)
            addView(
                Button(context).apply {
                    text = if (subtitle.isEmpty()) title else "$title\n$subtitle"
                    gravity = Gravity.START or Gravity.CENTER_VERTICAL
                    setTextColor(Color.WHITE)
                    setOnClickListener { onClick() }
                    layoutParams = LinearLayout.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.WRAP_CONTENT,
                    )
                },
            )
        }

    private companion object {
        const val PAD = 16
    }
}
