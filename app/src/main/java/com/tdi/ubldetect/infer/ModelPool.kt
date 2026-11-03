package com.tdi.ubldetect.infer

/**
 * Keeps loaded models alive between camera screens.
 *
 * Which model to run depends on the mode and the resolution. It does not
 * depend on which category or slab was picked, and nothing noticed that:
 * every category opened the same file again and sat on "Loading model" for
 * about 3.5 s, twice that for QPDS, to arrive at the interpreter it had just
 * closed. Picking a second category in the same mode now costs nothing.
 *
 * It holds a working set, not everything. The four bundled models come to
 * 197 MiB, and opening one before releasing the last left 678 MiB of mapped
 * pages resident across ten resolution swaps. So the pool closes what the new
 * request does not want **before** opening what it does, and the peak stays
 * where it is today.
 *
 * Generic over the thing being pooled so the policy can be tested without a
 * device: the real pool holds [TfliteRunner], whose constructor needs a
 * Context, an APK and a GPU.
 *
 * @param open builds a model. May throw; see [acquire].
 */
class ModelPool<T : AutoCloseable>(
    private val open: (asset: String, delegate: DelegateKind) -> T,
) {

    private class Holder<T>(val key: String, val value: T)

    private val held = mutableListOf<Holder<T>>()

    /**
     * How many screens are holding these.
     *
     * A count rather than a flag: leaving one camera for another runs the
     * new screen's onCreate before the old one's onDestroy, and a flag would
     * let the departing screen declare the pool idle while the arriving one
     * is running frames through it.
     */
    private var users = 0

    /** Called after each close. For tests and logging; not a lifecycle hook. */
    var onClose: ((T) -> Unit)? = null

    /** How many models are open right now. */
    val size: Int get() = synchronized(this) { held.size }

    /**
     * The models for [assets], reusing any already open.
     *
     * Returns them in the order asked for, because QPDS indexes them
     * positionally.
     *
     * If one cannot be opened the exception is passed on, and anything this
     * call opened on the way is closed first: the caller never got a
     * reference to it, so nothing else can.
     */
    @Synchronized
    fun acquire(assets: List<String>, delegate: DelegateKind): List<T> {
        val wanted = assets.map { keyOf(it, delegate) }
        closeExcept(wanted)

        val openedHere = mutableListOf<Holder<T>>()
        try {
            val result = assets.map { asset ->
                val key = keyOf(asset, delegate)
                val existing = held.firstOrNull { it.key == key }
                if (existing != null) {
                    existing.value
                } else {
                    val holder = Holder(key, open(asset, delegate))
                    held += holder
                    openedHere += holder
                    holder.value
                }
            }
            return result
        } catch (e: Throwable) {
            for (holder in openedHere) {
                held -= holder
                closeQuietly(holder.value)
            }
            throw e
        }
    }

    /**
     * Opens [assets] ahead of being asked for them.
     *
     * Called from the picker, where the wait would otherwise land on the
     * first tap.
     *
     * Only ever adds. It must not close anything, because it runs on its own
     * thread while a camera may still be shutting down, and the thing it
     * would close is exactly what that camera's last frame is inside. The
     * next [acquire] tidies up instead, from the analysis thread, where
     * closing is safe by construction. That costs one extra model resident at
     * most, and it makes switching mode fast as well.
     *
     * Failure is swallowed: nobody is waiting on this, and a resolution with
     * no shipped model is a state the picker has to survive.
     */
    @Synchronized
    fun warm(assets: List<String>, delegate: DelegateKind) {
        for (asset in assets) {
            val key = keyOf(asset, delegate)
            if (held.any { it.key == key }) continue
            try {
                held += Holder(key, open(asset, delegate))
            } catch (e: Exception) {
                // The camera will report it properly when something waits.
                return
            }
        }
    }

    /** Declares a holder. Paired with [endUse], which must always follow. */
    @Synchronized
    fun beginUse() {
        users++
    }

    /** Says one holder has finished. */
    @Synchronized
    fun endUse() {
        if (users > 0) users--
    }

    /**
     * Closes everything, unless something is using it.
     *
     * @return whether it released. False means a camera still holds these,
     *   and closing a native interpreter under a running frame is the crash
     *   this app has already had twice.
     */
    @Synchronized
    fun releaseIfIdle(): Boolean {
        if (users > 0) return false
        closeExcept(emptyList())
        return true
    }

    private fun closeExcept(keep: List<String>) {
        val iterator = held.iterator()
        while (iterator.hasNext()) {
            val holder = iterator.next()
            if (holder.key !in keep) {
                iterator.remove()
                closeQuietly(holder.value)
            }
        }
    }

    private fun closeQuietly(value: T) {
        runCatching { value.close() }
        onClose?.invoke(value)
    }

    /** The delegate is part of the identity: it is fixed when the model is built. */
    private fun keyOf(asset: String, delegate: DelegateKind) = "$asset@$delegate"
}
