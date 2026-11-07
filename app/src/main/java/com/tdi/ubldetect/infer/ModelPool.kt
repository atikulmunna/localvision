package com.tdi.ubldetect.infer

import java.util.concurrent.Callable
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger

/**
 * How a load is going, for a screen that has to wait for one.
 *
 * [total] counts only the models a load actually had to open, so a request
 * the pool could satisfy from what it already held reports nothing to wait
 * for.
 */
data class LoadProgress(val done: Int, val total: Int) {
    val waiting: Boolean get() = done < total
}

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
 * ## What may block, and what may not
 *
 * Opening a model takes seconds, and [acquire] and [warm] hold the lock for
 * that whole time on purpose: a second caller must wait for the model being
 * opened rather than open a second copy of it.
 *
 * Everything else has to stay off that lock. [beginUse] and [endUse] are
 * called from `onCreate` and from teardown, [releaseIfIdle] from a timer, and
 * [loadProgress] from a view drawing a spinner about this very load. When
 * those took the lock too, tapping a slab while the warm-up was still running
 * blocked the main thread inside `onCreate`: the camera screen measured
 * 4121 ms to appear, which throws away the entire point of warming, and the
 * person saw a frozen picker rather than a preview with a spinner on it.
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
     * Atomic rather than guarded by the lock, because it is touched from
     * `onCreate` on the main thread while a warm-up may hold the lock.
     *
     * A count rather than a flag: leaving one camera for another runs the
     * new screen's onCreate before the old one's onDestroy, and a flag would
     * let the departing screen declare the pool idle while the arriving one
     * is running frames through it.
     */
    private val users = AtomicInteger()

    @Volatile
    private var progress = LoadProgress(0, 0)

    /**
     * What the load running right now has finished, without taking the lock.
     *
     * Readable mid-load by design: the thread that wants it is the one
     * drawing the screen that is waiting for it.
     */
    val loadProgress: LoadProgress get() = progress

    /** Called after each close. For tests and logging; not a lifecycle hook. */
    var onClose: ((T) -> Unit)? = null

    /** How many models are open right now. */
    val size: Int get() = synchronized(this) { held.size }

    /**
     * The models for [assets], reusing any already open.
     *
     * Returns them in the order asked for, because QPDS indexes them
     * positionally, while they finish loading in whatever order they finish.
     *
     * If one cannot be opened the exception is passed on, and anything this
     * call opened on the way is closed first: the caller never got a
     * reference to it, so nothing else can.
     */
    @Synchronized
    fun acquire(assets: List<String>, delegate: DelegateKind): List<T> {
        val wanted = assets.map { keyOf(it, delegate) }
        closeExcept(wanted)

        val missing = assets.filter { a -> held.none { it.key == keyOf(a, delegate) } }
        val openedHere = mutableListOf<Holder<T>>()
        try {
            openTogether(missing, delegate) { holder ->
                held += holder
                openedHere += holder
            }
            return wanted.map { key -> held.first { it.key == key }.value }
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
        val missing = assets.filter { a -> held.none { it.key == keyOf(a, delegate) } }
        runCatching { openTogether(missing, delegate) { held += it } }
    }

    /**
     * Opens several models at once, publishing how far it has got.
     *
     * At once because QPDS needs two and was paying their sum. Measured on
     * the S23+ the segmenter costs about 3.8 s and the classifier about
     * 1.4 s, and almost all of that is the GPU driver preparing the model
     * rather than contention for one resource, so overlapping them costs
     * roughly what the segmenter alone costs.
     *
     * Each model is handed to [collect] as it finishes. A single model skips
     * the thread pool: that is every POSM load, and a pool of one thread
     * would be two allocations and a context switch to do the same work.
     */
    private fun openTogether(
        assets: List<String>,
        delegate: DelegateKind,
        collect: (Holder<T>) -> Unit,
    ) {
        progress = LoadProgress(0, assets.size)
        if (assets.isEmpty()) return

        if (assets.size == 1) {
            try {
                collect(Holder(keyOf(assets[0], delegate), open(assets[0], delegate)))
            } finally {
                progress = LoadProgress(assets.size, assets.size)
            }
            return
        }

        val finished = AtomicInteger()
        val threads = Executors.newFixedThreadPool(assets.size)
        try {
            // Submitted before any is waited on, which is the whole point:
            // waiting on the first would load the second after it.
            val pending = assets.map { asset ->
                threads.submit(
                    Callable {
                        val holder = Holder(keyOf(asset, delegate), open(asset, delegate))
                        progress = LoadProgress(finished.incrementAndGet(), assets.size)
                        holder
                    },
                )
            }
            var failure: Throwable? = null
            for (task in pending) {
                // Every task is collected even after one fails, so a model
                // that did open is held rather than leaked.
                try {
                    collect(task.get())
                } catch (e: Exception) {
                    failure = e.cause ?: e
                }
            }
            failure?.let { throw it }
        } finally {
            threads.shutdown()
            progress = LoadProgress(assets.size, assets.size)
        }
    }

    /** Declares a holder. Paired with [endUse], which must always follow. */
    fun beginUse() {
        users.incrementAndGet()
    }

    /** Says one holder has finished. */
    fun endUse() {
        users.updateAndGet { if (it > 0) it - 1 else 0 }
    }

    /**
     * Closes everything, unless something is using it.
     *
     * Not to be called from the main thread: it takes the lock, and the lock
     * can be held by a load that runs for seconds.
     *
     * @return whether it released. False means a camera still holds these,
     *   and closing a native interpreter under a running frame is the crash
     *   this app has already had twice.
     */
    @Synchronized
    fun releaseIfIdle(): Boolean {
        if (users.get() > 0) return false
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
