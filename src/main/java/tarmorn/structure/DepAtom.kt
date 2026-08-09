package tarmorn.structure

import tarmorn.Settings
import tarmorn.data.IdManager
import tarmorn.data.RelationPath
import kotlin.math.abs

/**
 * DepAtom - represents an atom in formulas.
 * @param T Instance type: Int for UnaryAtom, Long for BinaryAtom
 * relationId: relation or relation-path id
 * entityId: Y for binary, X for loop, 0 for existence, >0 for constant entity id
 * instances: set of entity instances covered by this atom (only for non-L1 atoms)
 * minHashSignature: computed MinHash signature for LSH
 * 
 * Note: For L1 BinaryAtom with inverse relation, instances are stored in forward order
 * but isInverseInstances can be derived from (isInverseRelation && isBinary && isL1Atom)
 */
class DepAtom(
    val relationId: Long,
    val entityId: Int
) {
    // Internally cached instance collection (only used for non-L1ofBinaryAtom) 
    // UseConcurrentHashMap.newKeySet()Ensure thread safety
    private val _instances: MutableSet<Long>? by lazy {
        if (isBinary && !isL1Atom) {
            java.util.concurrent.ConcurrentHashMap.newKeySet<Long>()
        } else {
            null
        }
    }
    
    // Mark whether the sample has been exhausted (continuous sampling yield is very low)
    @Volatile
    var samplingExhausted: Boolean = false
    
    // Mark whether sampling has been performed at least once (used to distinguish first sampling and incremental sampling)
    @Volatile
    var samplingRound: Int = 0

    val hasBeenSampled: Boolean
        get() = samplingRound > 0
    
    /**
     * Get the instance collection (unified interface)
     * - L1 atom: fromDepLearnCache acquisition
     * - NotL1 atom: Returns internally cachedinstances
     */
    val instances: Set<Long>
        get() = when {
            isL1Atom && isBinary -> tarmorn.DepLearn.r2instanceSet[relationId] ?: emptySet()
            !isL1Atom && isBinary -> _instances ?: emptySet()
            else -> emptySet()
        }
    
    // Determine whether the instance set is reverse storage: only for L1 BinaryAtom And yes inverse relation time is true
    val isInverseInstances: Boolean
        get() = isBinary && isL1Atom && IdManager.isInverseRelation(relationId)

    val isInverseRelation: Boolean
        get() = IdManager.isInverseRelation(relationId)

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is DepAtom) return false
        return relationId == other.relationId && entityId == other.entityId
    }

    fun getBinaryAtom(): DepAtom {
        require(!isBinary) { "getBinaryAtom can only be called on unary atoms" }
        return DepAtom(if (isInverseRelation) IdManager.getInverseRelation(relationId) else relationId, IdManager.getYId())
    }

    override fun hashCode(): Int {
        // IMPORTANT: Be careful not to use simple31 * relationId.hashCode() + entityId, Very easy to conflict
        return pairHash32(relationId.hashCode(), entityId)
    }

    override fun toString(): String {
        val relationStr = IdManager.getRelationString(relationId)
        return when {
            entityId == IdManager.getYId() -> relationStr
            entityId == IdManager.getXId() -> "$relationStr(X)"
            entityId == 0 -> "$relationStr(*)"
            else -> {
                val entityStr = IdManager.getEntityString(entityId)
                "$relationStr($entityStr)"
            }
        }
    }

    fun getRuleString(isVariableY: Boolean = false): String {
        if (isVariableY) {
            require(!isBinary) { "isVariableY is only allowed for unary atoms" }
        }
        // Resolve terminal argument
        fun termString(eid: Int): String = when (eid) {
            IdManager.getYId() -> "Y"
            IdManager.getXId() -> "X"
            0 -> "*"
            else -> IdManager.getEntityString(eid)
        }

        // Decode relation path
        val relations: LongArray = if (relationId <= RelationPath.MAX_RELATION_ID) longArrayOf(relationId) else RelationPath.decode(relationId)
        val n = relations.size
        val tailTerm = termString(entityId)

        // Build node list: X, A, B, ..., tailTerm
        val nodes = Array(n + 1) { "" }
        nodes[0] = if(isVariableY) "Y" else "X"
        for (i in 1 until n+1) {
            nodes[i] = ('A'.code + (i - 1)).toChar().toString()
        }
        if (tailTerm != "*") nodes[n] = tailTerm
        val parts = ArrayList<String>(n)
        for (i in 0 until n) {
            val r = relations[i]
            val inv = IdManager.isInverseRelation(r)
            val forward = if (inv) IdManager.getInverseRelation(r) else r
            val name = IdManager.getRelationString(forward)
            // swap args for inverse
            val left = if (inv) nodes[i + 1] else nodes[i]
            val right = if (inv) nodes[i] else nodes[i + 1]
            parts.add("$name($left,$right)")
        }
        return parts.joinToString(", ")
    }
    
    val isBinary: Boolean
        get() = entityId == IdManager.getYId()

    val isL1Atom: Boolean
        get() = relationId < RelationPath.MAX_RELATION_ID

    val isL2Atom: Boolean
        get() = relationId < RelationPath.MAX_L2RELATION_ID

    val isHeadAtom: Boolean
        get() = isL1Atom && entityId != 0

    val firstRelation: Long
        get() = if (isL1Atom) relationId else RelationPath.getFirstRelation(relationId)

    /**
     * Get unary instances (Set<Int>) - only for L1 atoms
     * Returns the set of head entities that satisfy this atom
     */
    fun getUnaryInstances(): Set<Int> {
        require(isL1Atom) { "getUnaryInstances only supports L1 atoms, got: $this" }
        require(!isBinary) { "getUnaryInstances does not support binary atoms, got: $this" }
        
        return when {
            // Loop atom: entities that loop on themselves
            entityId == IdManager.getXId() -> {
                tarmorn.DepLearn.ts.r2loopSet[relationId] ?: emptySet()
            }
            // Existence atom: all heads that have this relation
            entityId == 0 -> {
                tarmorn.DepLearn.r2h2tSet[relationId]?.keys ?: emptySet()
            }
            // Constant atom: heads that connect to this specific entity
            else -> {
                val inverseRelation = RelationPath.getInverseRelation(relationId)
                tarmorn.DepLearn.r2h2tSet[inverseRelation]?.get(entityId) ?: emptySet()
            }
        }
    }

    /**
     * Get binary instances (Set<Long>) - only for L1 atoms
     * Returns the set of (head, tail) pairs as Long
     */
    fun getBinaryInstances(): Set<Long> {
        require(isL1Atom) { "getBinaryInstances only supports L1 atoms, got: $this" }
        require(isBinary) { "getBinaryInstances only supports binary atoms, got: $this" }
        
        return tarmorn.DepLearn.r2instanceSet[relationId] ?: emptySet()
    }

    /**
     * Representation atom: Get all entity instances that satisfy the current atom
     * 
     * For a unary atom (unary atom, Only supportsL1) : 
     * - Returns the set of all entities that satisfy this atom
     * - For example:rel(const) Return all satisfying rel(X, const) of X collection
     * 
     * For binary atoms (binary atom) : 
     * - Need to provide entityId and isHead parameters
     * - isHead=true: givenheadEntity, returns all possibletailEntity
     * - isHead=false: giventailEntity, returns all possibleheadEntity
     * 
     * @param givenEntityId For binary atoms, the entity needs to be providedID (asheadortail) 
     * @param isHead For binary atoms,givenEntityIdwhether ashead (true) Stilltail (false) 
     * @return entities that meet the conditionsIDcollection
     */
    fun materialize(givenEntityId: Int = -1, isHead: Boolean = true): Set<Int> {
        return if (isBinary) {
            materializeBinary(givenEntityId, isHead)
        } else {
            require(isL1Atom) { "materialize only supports L1 unary atoms, got: $this" }
            getUnaryInstances()
        }
    }

    /**
     * Representation Binary Atom: Given an entity (asheadortail) , Returns all entities on the other end that can form a relationship with it
     */
    private fun materializeBinary(givenEntityId: Int, isHead: Boolean): Set<Int> {
        require(givenEntityId > 0) { "Binary atom materialize requires a valid entityId, got: $givenEntityId" }
        
        // According toisHeadDecide whether to use a forward or inverse relationship
        val actualRelation = if (isHead) relationId else RelationPath.getInverseRelation(relationId)
        
        // forL1Atom, direct query
        if (isL1Atom) {
            return tarmorn.DepLearn.r2h2tSet[actualRelation]?.get(givenEntityId) ?: emptySet()
        }
        
        // forL2+Atoms, gradually expanded along the relational path
        val relations = RelationPath.decode(actualRelation)
        var currentLayer = setOf(givenEntityId)
        
        for (relation in relations) {
            val nextLayer = mutableSetOf<Int>()
            for (entity in currentLayer) {
                val successors = tarmorn.DepLearn.r2h2tSet[relation]?.get(entity) ?: continue
                nextLayer.addAll(successors)
            }
            currentLayer = nextLayer
            if (currentLayer.isEmpty()) break
        }
        
        return currentLayer
    }

    /**
     * Check if a binary instance exists using bi-directional DFS
     * After successful verification, the instance will be added to the cache
     * @param instance The (head, tail) pair as Long
     * @return true if instance exists
     */
    fun hasBinaryInstance(instance: Long): Boolean {
        require(isBinary) { "hasBinaryInstance only supports binary atoms, got: $this" }
        
        val h = (instance shr 32).toInt()
        val t = instance.toInt()
        
        // For L1 atoms, direct lookup
        if (isL1Atom) {
            return tarmorn.DepLearn.r2h2tSet[relationId]?.get(h)?.contains(t) ?: false
        }
        
        // Check cache first
        if (_instances?.contains(instance) == true) {
            return true
        }
        
        // For longer paths, use bi-directional search
        val relations = RelationPath.decode(relationId)
        val pathLength = relations.size
        var verified = false
        
        when (pathLength) {
            2 -> {
                // r1 * r2: check if exists middle node m such that r1(h, m) && r2(m, t)
                val r1 = relations[0]
                val r2 = relations[1]
                
                val r1Tails = tarmorn.DepLearn.r2h2tSet[r1]?.get(h) ?: return false
                val r2Inv = RelationPath.getInverseRelation(r2)
                val r2Heads = tarmorn.DepLearn.r2h2tSet[r2Inv]?.get(t) ?: return false
                
                // Check intersection
                verified = r1Tails.any { it in r2Heads && it != h && it != t }
            }
            3 -> {
                // r1 * r2 * r3: bi-directional search from both ends
                val r1 = relations[0]
                val r2 = relations[1]
                val r3 = relations[2]
                
                // Get 1-hop from head
                val head1hop = tarmorn.DepLearn.r2h2tSet[r1]?.get(h) ?: return false
                
                // Get 1-hop from tail (backward)
                val r3Inv = RelationPath.getInverseRelation(r3)
                val tail1hop = tarmorn.DepLearn.r2h2tSet[r3Inv]?.get(t) ?: return false
                
                // Choose smaller set to iterate
                if (head1hop.size <= tail1hop.size) {
                    // Iterate from head side
                    for (m in head1hop) {
                        if (m == h || m == t) continue
                        // Check if r2(m, ?) intersects with tail1hop
                        val r2Tails = tarmorn.DepLearn.r2h2tSet[r2]?.get(m) ?: continue
                        if (r2Tails.any { it in tail1hop && it != h && it != t }) {
                            verified = true
                            break
                        }
                    }
                } else {
                    // Iterate from tail side
                    for (m in tail1hop) {
                        if (m == h || m == t) continue
                        // Check if r2'(m, ?) intersects with head1hop
                        val r2Inv = RelationPath.getInverseRelation(r2)
                        val r2Heads = tarmorn.DepLearn.r2h2tSet[r2Inv]?.get(m) ?: continue
                        if (r2Heads.any { it in head1hop && it != h && it != t }) {
                            verified = true
                            break
                        }
                    }
                }
            }
            else -> {
                // For paths longer than 3, general bi-directional BFS
                // Start from both ends and meet in the middle
                val midPoint = pathLength / 2
                
                // Build forward reachable set from head
                var forwardReachable = setOf(h)
                for (i in 0 until midPoint) {
                    val nextReachable = mutableSetOf<Int>()
                    for (node in forwardReachable) {
                        val tails = tarmorn.DepLearn.r2h2tSet[relations[i]]?.get(node) ?: continue
                        nextReachable.addAll(tails)
                    }
                    forwardReachable = nextReachable
                }
                
                // Build backward reachable set from tail
                var backwardReachable = setOf(t)
                for (i in pathLength - 1 downTo midPoint) {
                    val nextReachable = mutableSetOf<Int>()
                    for (node in backwardReachable) {
                        val rInv = RelationPath.getInverseRelation(relations[i])
                        val heads = tarmorn.DepLearn.r2h2tSet[rInv]?.get(node) ?: continue
                        nextReachable.addAll(heads)
                    }
                    backwardReachable = nextReachable
                }
                
                verified = forwardReachable.any { it in backwardReachable && it != h && it != t }
            }
        }
        
        // If verification is successful, add to cache
        if (verified) _instances?.add(instance)
        return verified
    }
    
    /**
     * EDISsampling - Entity-DThe results are first cached locally and finally added internally in batches.instances
     * Support multiple calls and gradually accumulate instances
     * 
     * @param maxAttempts Maximum number of sampling attempts (default1000, suitable for dynamic sampling)
     * @param maxGroundings targetgroundingquantity (default10, suitable for dynamic sampling)
     * @param maxRepetitions Maximum number of consecutive repetitions (default5) 
     * @param minNewInstanceRatio Minimum new instance ratio threshold (default0.05, That is5%) , Below this ratio, it is marked as sample exhausted.
     * @return The newly added instance collection this time
     */
    fun sampleBinaryInstancesEDIS(
        maxAttempts: Int = Settings.BEAM_SAMPLING_MAX_BODY_GROUNDING_ATTEMPTS,
        maxGroundings: Int = Settings.BEAM_SAMPLING_MAX_BODY_GROUNDINGS,
        maxRepetitions: Int = Settings.BEAM_SAMPLING_MAX_REPETITIONS
    ): Set<Long> {
        require(isBinary) { "EDIS sampling only supports binary atoms, got: $this" }
        
        // L1Atoms do not need to be sampled and can be used directlyDepLearncache
        if (isL1Atom) {
            return emptySet()
        }
        
        // If it has been marked as sample exhausted, return directly
        if (samplingExhausted) {
            return emptySet()
        }
        
        // Determine whether it is the first sampling, if so, use larger parameters
        val actualMaxAttempts = if (samplingRound==0) maxAttempts else maxAttempts / 10
        val actualMaxGroundings = if (samplingRound==0) maxGroundings else maxGroundings / 10
        
        // ensure_instancesInitialized
        val instances = _instances ?: return emptySet()
        
        // Bidirectional sampling:forward + inverse
        val forwardInstances = sampleBinaryInstancesEDISDirection(
            relationPathId = relationId,
            isForward = true,
            maxAttempts = actualMaxAttempts,
            maxGroundings = actualMaxGroundings,
            maxRepetitions = maxRepetitions
        )
        val inverseInstances = sampleBinaryInstancesEDISDirection(
            relationPathId = RelationPath.getInverseRelation(relationId),
            isForward = false,
            maxAttempts = actualMaxAttempts,
            maxGroundings = actualMaxGroundings,
            maxRepetitions = maxRepetitions
        )
        
        val newInstances = mutableSetOf<Long>()
        newInstances.addAll(forwardInstances)
        newInstances.addAll(inverseInstances)
        
        samplingRound++
        
        val totalAttempts = actualMaxAttempts * 2
        if (samplingRound >= 100 ||
            newInstances.isEmpty() ||
            newInstances.size.toDouble() / totalAttempts < 0.05) {
            samplingExhausted = true
        }
        
        instances.addAll(newInstances)
        return newInstances
    }

    /**
     * One wayEDISInternal implementation of sampling
     */
    private fun sampleBinaryInstancesEDISDirection(
        relationPathId: Long,
        isForward: Boolean,
        maxAttempts: Int,
        maxGroundings: Int,
        maxRepetitions: Int
    ): Set<Long> {
        val instances = _instances ?: return emptySet()

        val relations = RelationPath.decode(relationPathId)
        val firstRelation = relations[0]

        val startEntities = getSampledStartEntities(firstRelation, maxAttempts)
        if (startEntities.isEmpty()) return emptySet()

        val newInstances = mutableSetOf<Long>()
        var attempts = 0
        var repetitions = 0

        for (startEntity in startEntities) {
            if (attempts >= maxAttempts) break
            if (newInstances.size >= maxGroundings) break
            if (repetitions >= maxRepetitions) break

            attempts++

            val endEntity = beamCyclicPath(startEntity, relations.toList())
            if (endEntity != null && endEntity != startEntity) {
                val instance = if (isForward) {
                    packLong(startEntity, endEntity)
                } else {
                    packLong(endEntity, startEntity)
                }

                if (instance !in instances && instance !in newInstances) {
                    newInstances.add(instance)
                    repetitions = 0
                } else {
                    repetitions++
                }
            }
        }

        return newInstances
    }
    
    /**
     * Get the starting entity for sampling (uniform distribution)
     * Similar toTripleSet.getNRandomEntitiesByRelation, But forDepAtom
     */
    private fun getSampledStartEntities(firstRelation: Long, n: Int): List<Int> {
        // Get all of the relationshipheadEntity
        val allHeads = tarmorn.DepLearn.r2h2tSet[firstRelation]?.keys ?: return emptyList()
        
        // If the number of entities is less thann, Return all entities
        if (allHeads.size <= n) {
            return allHeads.toList()
        }
        
        // random samplingnentities (evenly distributed)
        return allHeads.shuffled().take(n)
    }
    
    /**
     * random walk completion path
     * fromstartEntityset off, alongrelationsThe path is randomly walked and the end entity is returned.
     * 
     * @param startEntity starting entity
     * @param relations Relationship path (decoded)
     * @return End entity, returned if the path is blockednull
     */
    private fun beamCyclicPath(startEntity: Int, relations: List<Long>): Int? {
        var currentEntity = startEntity
        val visitedEntities = mutableSetOf(startEntity)
        
        for (relation in relations) {
            // Get the entities that the current entity can reach through this relationship
            val nextEntities = tarmorn.DepLearn.r2h2tSet[relation]?.get(currentEntity)
            
            if (nextEntities == null || nextEntities.isEmpty()) {
                return null // The path is blocked
            }
            
            // Randomly select an unvisited entity (OIconstraints)
            val candidateEntities = if (Settings.OI_CONSTRAINTS_ACTIVE) {
                nextEntities.filter { it !in visitedEntities }
            } else {
                nextEntities.toList()
            }
            
            if (candidateEntities.isEmpty()) {
                return null // No optional entities
            }
            
            // Randomly select the next entity
            currentEntity = candidateEntities.random()
            
            if (Settings.OI_CONSTRAINTS_ACTIVE) {
                visitedEntities.add(currentEntity)
            }
        }
        
        return currentEntity
    }
    
    /**
     * will(head, tail)packaged asLong
     */
    private fun packLong(head: Int, tail: Int): Long {
        return (head.toLong() shl 32) or (tail.toLong() and 0xFFFFFFFFL)
    }
    
    /**
     * fromLongUnpack outhead
     */
    private fun unpackHead(instance: Long): Int {
        return (instance shr 32).toInt()
    }
    
    /**
     * fromLongUnpack outtail
     */
    private fun unpackTail(instance: Long): Int {
        return instance.toInt()
    }

    companion object {
        fun pairHash32(h: Int, t: Int): Int {
            val uH = h * -0x61c88647     // 0x9E3779B9 's complement (golden proportional constant)
            val uT = t * 0x85ebca6b.toInt()
            return uH xor Integer.rotateLeft(uT, 16)
        }
    }
}
