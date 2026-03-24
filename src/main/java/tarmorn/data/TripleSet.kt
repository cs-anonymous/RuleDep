package tarmorn.data

import tarmorn.Settings
import java.io.File

class TripleSet(path: String, addInverseRelations: Boolean) : Iterable<MyTriple> {
    private val triples = mutableListOf<MyTriple>()

    val r2h2tSet: Map<Long, Map<Int, Set<Int>>>
    val r2loopSet: Map<Long, Set<Int>>
    val size: Int
        get() = triples.size

    init {
        val forwardIndex = mutableMapOf<Long, MutableMap<Int, MutableSet<Int>>>()
        val loopIndex = mutableMapOf<Long, MutableSet<Int>>()

        File(path).forEachLine { rawLine ->
            val line = rawLine.trim()
            if (line.isEmpty()) return@forEachLine
            val parts = line.split('\t')
            require(parts.size >= 3) { "Invalid triple line: $line" }

            val head = normalizeEntity(parts[0])
            val relation = normalizeRelation(parts[1])
            val tail = normalizeEntity(parts[2])

            val headId = IdManager.getEntityId(head)
            val relationId = IdManager.getRelationId(relation)
            val tailId = IdManager.getEntityId(tail)

            addTriple(forwardIndex, loopIndex, headId, relationId, tailId)
            if (addInverseRelations) {
                addTriple(forwardIndex, loopIndex, tailId, IdManager.getInverseRelation(relationId), headId)
            }
        }

        r2h2tSet = forwardIndex.mapValues { (_, h2t) ->
            h2t.mapValues { (_, tails) -> tails.toSet() }
        }
        r2loopSet = loopIndex.mapValues { (_, entities) -> entities.toSet() }
    }

    private fun addTriple(
        index: MutableMap<Long, MutableMap<Int, MutableSet<Int>>>,
        loops: MutableMap<Long, MutableSet<Int>>,
        headId: Int,
        relationId: Long,
        tailId: Int
    ) {
        triples += MyTriple(headId, relationId, tailId)
        val h2t = index.computeIfAbsent(relationId) { mutableMapOf() }
        val tails = h2t.computeIfAbsent(headId) { mutableSetOf() }
        tails += tailId
        if (headId == tailId) {
            loops.computeIfAbsent(relationId) { mutableSetOf() } += headId
        }
    }

    private fun normalizeEntity(entity: String): String {
        return if (Settings.SAFE_PREFIX_MODE && entity.all(Char::isDigit)) {
            Settings.PREFIX_ENTITY + entity
        } else {
            entity
        }
    }

    private fun normalizeRelation(relation: String): String {
        return if (Settings.SAFE_PREFIX_MODE && relation.all(Char::isDigit)) {
            Settings.PREFIX_RELATION + relation
        } else {
            relation
        }
    }

    override fun iterator(): Iterator<MyTriple> = triples.iterator()
}
