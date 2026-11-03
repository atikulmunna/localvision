package com.tdi.ubldetect.ui

import android.app.Activity
import android.content.Intent
import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import com.tdi.ubldetect.R
import com.tdi.ubldetect.UblApp
import com.tdi.ubldetect.catalog.Catalog
import com.tdi.ubldetect.catalog.CatalogEntry
import com.tdi.ubldetect.catalog.Mode

/**
 * Pick a mode, then a slab or a category, then start.
 *
 * The choice is not cosmetic. It restricts which classes are allowed, which
 * is both quicker to read and more accurate: a classifier choosing between
 * the SKUs a slab actually carries makes fewer mistakes than one choosing
 * between all 56.
 *
 * It is also where reconciliation surfaces. Some entries expect classes the
 * deployed model does not have, and at least one (`mini_meals`) expects
 * nothing it can detect at all. Rather than letting someone open a camera
 * and work that out for themselves, each row carries its [EntryState]: ready,
 * partial, or unusable and not tappable.
 */
class PickerActivity : Activity() {

    private lateinit var catalog: Catalog
    private lateinit var container: LinearLayout

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        catalog = Catalog.parse(assets.open("catalog.json").bufferedReader().readText())

        val scroll = ScrollView(this).apply {
            setBackgroundColor(color(R.color.surface))
            isFillViewport = true
        }
        container = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(20f), dp(8f), dp(20f), dp(24f))
        }
        scroll.addView(container)
        setContentView(scroll)
        scroll.padForSystemBars()

        showModes()
    }

    private fun color(id: Int) = resources.getColor(id, theme)

    private fun showModes() {
        container.removeAllViews()
        container.addView(title("localVision"))
        container.addView(subtitle("What are you pointing at?"))
        container.addView(
            card(
                heading = "POSM",
                detail = "${catalog.posmCategories.size} categories",
                state = EntryState.READY,
            ) { showEntries(Mode.POSM) },
        )
        container.addView(
            card(
                heading = "QPDS",
                detail = "${catalog.qpdsSlabs.size} slabs, two stage",
                state = EntryState.READY,
            ) { showEntries(Mode.QPDS) },
        )
    }

    private fun showEntries(mode: Mode) {
        // The load takes about as long as choosing from this list does, so it
        // starts now rather than after the tap. Defaults, because that is
        // what CameraActivity starts from.
        val defaults = ControlsState()
        (application as UblApp).warmUp(defaults.assetsFor(mode), defaults.delegate)
        container.removeAllViews()
        container.addView(title(if (mode == Mode.POSM) "Category" else "Slab"))
        container.addView(backRow { showModes() })

        for (entry in catalog.entries(mode)) {
            val state = EntryState.of(entry)
            container.addView(
                card(
                    heading = pretty(entry.key),
                    detail = describe(entry),
                    state = state,
                    note = noteFor(entry, state),
                ) { start(mode, entry) },
            )
        }
    }

    /** `SKIN_CARE_GAL_GOLD` reads better as `Skin care gal gold`. */
    private fun pretty(key: String): String =
        key.replace('_', ' ').lowercase().replaceFirstChar { it.uppercase() }

    private fun describe(entry: CatalogEntry): String =
        "${entry.subtitle}  ${DOT}  ${entry.classIds.size} detectable"

    private fun noteFor(entry: CatalogEntry, state: EntryState): String? = when (state) {
        EntryState.READY -> null
        EntryState.PARTIAL ->
            "${entry.missingFromModel.size} not in this model yet"
        EntryState.UNUSABLE ->
            "This model has none of its classes:\n" +
                entry.missingFromModel.joinToString(", ").ifEmpty { "nothing mapped" }
    }

    private fun start(mode: Mode, entry: CatalogEntry) {
        startActivity(
            Intent(this, CameraActivity::class.java)
                .putExtra(CameraActivity.EXTRA_MODE, mode.name)
                .putExtra(CameraActivity.EXTRA_KEY, entry.key),
        )
    }

    private fun title(text: String) = TextView(this).apply {
        this.text = text
        setTextSize(TypedValue.COMPLEX_UNIT_SP, 28f)
        setTextColor(color(R.color.text_primary))
        setPadding(0, dp(24f), 0, dp(4f))
    }

    private fun subtitle(text: String) = TextView(this).apply {
        this.text = text
        setTextSize(TypedValue.COMPLEX_UNIT_SP, 15f)
        setTextColor(color(R.color.text_secondary))
        setPadding(0, 0, 0, dp(16f))
    }

    private fun backRow(onClick: () -> Unit) = TextView(this).apply {
        text = "Back"
        setTextSize(TypedValue.COMPLEX_UNIT_SP, 15f)
        setTextColor(color(R.color.accent))
        setPadding(dp(4f), dp(8f), dp(4f), dp(16f))
        isClickable = true
        setOnClickListener { onClick() }
    }

    /**
     * One row.
     *
     * A card rather than a button: stock Android buttons force upper case and
     * give one line of text, which flattens the name and its detail into the
     * same weight. Here the name leads and the detail sits quiet under it.
     */
    private fun card(
        heading: String,
        detail: String,
        state: EntryState,
        note: String? = null,
        onClick: () -> Unit,
    ): View = LinearLayout(this).apply {
        orientation = LinearLayout.VERTICAL
        val usable = state.canOpen

        background = GradientDrawable().apply {
            cornerRadius = dp(14f).toFloat()
            setColor(color(if (usable) R.color.card else R.color.card_disabled))
            setStroke(dp(1f), color(R.color.card_border))
        }
        setPadding(dp(16f), dp(14f), dp(16f), dp(14f))
        layoutParams = LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT,
        ).apply { bottomMargin = dp(10f) }

        addView(
            TextView(context).apply {
                text = heading
                setTextSize(TypedValue.COMPLEX_UNIT_SP, 17f)
                setTextColor(
                    color(if (usable) R.color.text_primary else R.color.text_disabled),
                )
            },
        )
        addView(
            TextView(context).apply {
                text = detail
                setTextSize(TypedValue.COMPLEX_UNIT_SP, 13f)
                setTextColor(color(R.color.text_secondary))
                setPadding(0, dp(3f), 0, 0)
            },
        )
        if (note != null) {
            addView(noteView(note, state))
        }

        if (usable) {
            isClickable = true
            setOnClickListener { onClick() }
        } else {
            // Not tappable at all. Letting someone open it and then refusing
            // teaches them nothing until after they have tried.
            isEnabled = false
        }
    }

    private fun noteView(note: String, state: EntryState) = TextView(this).apply {
        text = note
        setTextSize(TypedValue.COMPLEX_UNIT_SP, 12f)
        setTextColor(
            color(if (state == EntryState.PARTIAL) R.color.warn_text else R.color.text_disabled),
        )
        gravity = Gravity.START
        setPadding(dp(8f), dp(6f), dp(8f), dp(6f))
        background = GradientDrawable().apply {
            cornerRadius = dp(8f).toFloat()
            setColor(
                if (state == EntryState.PARTIAL) {
                    color(R.color.warn_surface)
                } else {
                    Color.TRANSPARENT
                },
            )
        }
        (layoutParams as? LinearLayout.LayoutParams)?.topMargin = dp(8f)
    }

    private companion object {
        const val DOT = "·"
    }
}
