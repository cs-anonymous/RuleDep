package tarmorn.data

data class MyTriple(
    val h: Int,
    val r: Long,
    val t: Int
) {
    override fun toString(): String {
        return "${IdManager.getEntityString(h)}\t${IdManager.getRelationString(r)}\t${IdManager.getEntityString(t)}"
    }
}
