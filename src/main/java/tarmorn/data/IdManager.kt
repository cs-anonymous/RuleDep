package tarmorn.data

import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong

object IdManager {
    private const val INVERSE_PREFIX = "INVERSE_"
    private const val X_ID = -1
    private const val Y_ID = -2
    private const val Z_ID = -3

    private val nextEntityId = AtomicInteger(1)
    private val nextRelationIndex = AtomicLong(1)

    private val entityToId = ConcurrentHashMap<String, Int>()
    private val idToEntity = ConcurrentHashMap<Int, String>()
    private val relationToId = ConcurrentHashMap<String, Long>()
    private val idToRelation = ConcurrentHashMap<Long, String>()

    fun getXId(): Int = X_ID

    fun getYId(): Int = Y_ID

    fun getZId(): Int = Z_ID

    fun getEntityId(entity: String): Int {
        return entityToId.computeIfAbsent(entity) { name ->
            val id = nextEntityId.getAndIncrement()
            idToEntity[id] = name
            id
        }
    }

    fun getEntityString(entityId: Int): String {
        return when (entityId) {
            X_ID -> "X"
            Y_ID -> "Y"
            Z_ID -> "Z"
            else -> idToEntity[entityId] ?: "E$entityId"
        }
    }

    fun getRelationId(relation: String): Long {
        if (relation.startsWith(INVERSE_PREFIX)) {
            return getInverseRelation(getRelationId(relation.removePrefix(INVERSE_PREFIX)))
        }
        return relationToId.computeIfAbsent(relation) { name ->
            val forwardId = nextRelationIndex.getAndIncrement() shl 1
            idToRelation[forwardId] = name
            forwardId
        }
    }

    fun getRelationString(relationId: Long): String {
        if (RelationPath.isRelationPath(relationId)) {
            return RelationPath.decode(relationId).joinToString("*") { getRelationString(it) }
        }
        val forwardId = if (isInverseRelation(relationId)) getInverseRelation(relationId) else relationId
        val baseName = idToRelation[forwardId] ?: "R$forwardId"
        return if (isInverseRelation(relationId)) INVERSE_PREFIX + baseName else baseName
    }

    fun isInverseRelation(relationId: Long): Boolean {
        return !RelationPath.isRelationPath(relationId) && relationId % 2L == 1L
    }

    fun getInverseRelation(relationId: Long): Long {
        require(!RelationPath.isRelationPath(relationId)) {
            "Use RelationPath.getInverseRelation for relation paths: $relationId"
        }
        return if (isInverseRelation(relationId)) relationId - 1 else relationId + 1
    }
}
