package tarmorn.structure

import tarmorn.Settings
import kotlin.math.min

/**
 * Metric describing support/coverage/confidence.
 */
data class Metric(
    var support: Double,
    val headSize: Int,
    val bodySize: Int,
    var lift: Double = 0.0
) : Comparable<Metric> {
    val coverage: Double = if (headSize > 0) support / headSize else 0.0

    private fun surprisalFrom(conf: Double): Double {
        return if (conf < 1.0) minOf(-Math.log(1 - conf), Settings.MAX_SURPRISAL) else Settings.MAX_SURPRISAL
    }

    val rawConfidence: Double
        get() = if (bodySize > 0) support / bodySize else 0.0

    val confidence: Double
        get() = if (bodySize > 0) support / (bodySize + Settings.UNSEEN_NEGATIVE_EXAMPLES) else 0.0

    val rawSurprisal: Double
        get() = surprisalFrom(rawConfidence)

    val smoothSurprisal: Double
        get() = surprisalFrom(confidence)

    val surprisal: Double
        get() = smoothSurprisal

    val valid: Boolean
        get() = support >= Settings.MIN_SUPP && confidence > Settings.MIN_CONF // && coverage > 0.1

    val needValidation: Boolean
        get() = support < Settings.MIN_SUPP * 2 || support > min(headSize, bodySize).toDouble()

    private fun format5(value: Double): String {
        val formatted = String.format(java.util.Locale.US, "%.5f", value)
        return formatted.trimEnd('0').trimEnd('.')
    }

    override fun toString(): String {
        return "{\"support\":${format5(support)}, \"headSize\":$headSize, \"bodySize\":$bodySize, \"confidence\":${format5(confidence)}, \"lift\":${format5(lift)}}"
    }

    fun inverse() =  Metric(support, bodySize, headSize)

    override fun compareTo(other: Metric): Int {
        // Sort by confidence descending (higher confidence first)
        return other.confidence.compareTo(this.confidence)
    }
}
