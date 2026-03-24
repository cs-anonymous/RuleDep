package tarmorn.data

import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicLong

object RelationPath {
    const val MAX_RELATION_ID: Long = 1_000_000_000_000L
    const val MAX_L2RELATION_ID: Long = Long.MAX_VALUE

    private val nextPathId = AtomicLong(MAX_RELATION_ID + 1)
    private val pathToId = ConcurrentHashMap<List<Long>, Long>()
    private val idToPath = ConcurrentHashMap<Long, LongArray>()

    fun isRelationPath(relationId: Long): Boolean = relationId > MAX_RELATION_ID

    fun encode(relations: LongArray): Long {
        require(relations.isNotEmpty()) { "Relation path cannot be empty" }
        if (relations.size == 1) return relations[0]
        val key = relations.toList()
        return pathToId.computeIfAbsent(key) {
            val id = nextPathId.getAndIncrement()
            idToPath[id] = relations.copyOf()
            id
        }
    }

    fun decode(relationId: Long): LongArray {
        if (!isRelationPath(relationId)) return longArrayOf(relationId)
        return idToPath[relationId]?.copyOf()
            ?: error("Unknown relation path id: $relationId")
    }

    fun getFirstRelation(relationId: Long): Long {
        return decode(relationId).first()
    }

    fun getInverseRelation(relationId: Long): Long {
        if (!isRelationPath(relationId)) return IdManager.getInverseRelation(relationId)
        val relations = decode(relationId)
        val inverse = LongArray(relations.size)
        for (i in relations.indices) {
            inverse[i] = IdManager.getInverseRelation(relations[relations.lastIndex - i])
        }
        return encode(inverse)
    }
}
