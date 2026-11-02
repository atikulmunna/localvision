package com.tdi.ubldetect.ui

import com.tdi.ubldetect.catalog.CatalogEntry

/**
 * Whether a slab or category can do anything, judged before it is opened.
 *
 * The catalog already reconciles the standards against the exported model's
 * classes, so the picker can say this up front instead of letting someone
 * open a camera and work out for themselves that it will never label.
 *
 * @property canOpen false means tapping it would only disappoint
 */
enum class EntryState(val canOpen: Boolean) {
    /** Every class it expects is in the model. */
    READY(true),

    /**
     * Some classes are missing.
     *
     * Worth saying, not worth alarm: new SKUs reach the standards before a
     * model is trained to detect them, so this is the normal state of a
     * live catalog rather than a fault.
     */
    PARTIAL(true),

    /** Nothing it expects is in the model, so it can never label anything. */
    UNUSABLE(false),
    ;

    companion object {
        fun of(entry: CatalogEntry): EntryState = when {
            entry.isEmpty -> UNUSABLE
            entry.missingFromModel.isNotEmpty() -> PARTIAL
            else -> READY
        }
    }
}
