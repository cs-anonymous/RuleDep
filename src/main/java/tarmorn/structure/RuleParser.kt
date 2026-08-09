package tarmorn.structure

import tarmorn.data.IdManager
import tarmorn.data.RelationPath

/**
 * Rule parser, supporting multiple rule formats and abbreviations
 * 
 * Supported rule formats:
 * 1. Abbreviated format:
 *    /award/award_category/winners./award/award_honor/ceremony <= 
 *    /award/award_category/winners./award/award_honor/ceremony* /award/award_ceremony/awards_presented./award/award_honor/award_winner*INVERSE_/award/award_ceremony/awards_presented./award/award_honor/award_winner
 * 
 * 2. Bracketed format:
 *    /award/award_category/winners./award/award_honor/ceremony(X,Y) <= 
 *    /award/award_category/winners./award/award_honor/award_winner(X,A), /award/award_ceremony/awards_presented./award/award_honor/award_winner(Y,A)
 * 
 * 3. Supports single variable rules (univariate) and double variable rules (binary)
 */
object RuleParser {
    
    /** Debug mode switch */
    var DEBUG = false
    
    /** Debug output function */
    private fun debug(message: String) {
        if (DEBUG) {
            println("[DEBUG] $message")
        }
    }
    
    /** Determine whether the parameter is a variable (single letter orme_myself_i)  */
    fun isVariable(arg: String) = arg.length == 1 || arg == "me_myself_i"
    
    /** Determine whether the entity name contains special characters (brackets or commas) */
    private fun hasSpecialChars(entity: String): Boolean {
        return ('(' in entity || ')' in entity || ',' in entity) && !entity.startsWith("/m/")
    }
    
    /** Determine whether it is an entity placeholder (EBegin with numbers followed by) */
    private fun isEntityPlaceholder(arg: String): Boolean {
        return arg.matches(Regex("E\\d+"))
    }
    
    /**
     * Preprocessing rule string: Replace entities containing special characters within brackets with placeholders
     * For example:playsFor(Tom_Kelly_(footballer,born_1964),Y) -> playsFor(E123,Y)
     * 
     * Strategy: only process entities that appear as parameters (in parentheses), not relationship names
     */
    private fun preprocessRule(ruleStr: String): String {
        val result = StringBuilder()
        var i = 0
        
        while (i < ruleStr.length) {
            // Find the left bracket after the relationship name
            if (ruleStr[i] == '(') {
                result.append('(')
                i++
                
                // Now we parse each parameter within the parameter list
                val argsStart = i
                var parenDepth = 1
                val argsEnd = run {
                    var pos = i
                    while (pos < ruleStr.length && parenDepth > 0) {
                        when (ruleStr[pos]) {
                            '(' -> parenDepth++
                            ')' -> parenDepth--
                        }
                        if (parenDepth > 0) pos++
                    }
                    pos
                }
                
                // Extract the parameter part and process it
                val argsString = ruleStr.substring(argsStart, argsEnd)
                val processedArgs = preprocessArguments(argsString)
                result.append(processedArgs)
                
                i = argsEnd
            } else {
                result.append(ruleStr[i])
                i++
            }
        }
        
        return result.toString()
    }
    
    /**
     * Preprocessing parameter list: Replace entities containing special characters with placeholders
     * Split only at top-level commas, respecting nested parentheses
     */
    private fun preprocessArguments(argsString: String): String {
        // Split directly by commas. If more than one comma appears, it means that an entity name contains a comma./brackets
        // one yuan/Binary rule: merge based on variable position
        var args = argsString.split(',').map { it.trim() }.filter { it.isNotEmpty() }
        if (args.size > 2) {
            args = when {
                isVariable(args.first()) && !isVariable(args[1]) -> {
                    val mergedSecond = args.drop(1).joinToString(",").trim()
                    listOf(args.first().trim(), mergedSecond)
                }
                isVariable(args.last()) -> {
                    val mergedFirst = args.dropLast(1).joinToString(",").trim()
                    listOf(mergedFirst, args.last().trim())
                }
                else -> {
                    val mergedFirst = args.dropLast(1).joinToString(",").trim()
                    listOf(mergedFirst, args.last().trim())
                }
            }
        }
        val processedArgs = args.map { arg ->
            val trimmedArg = arg.trim()
            // If the parameter contains special characters and is not a variable, replace it with a placeholder
            if (hasSpecialChars(trimmedArg) && !isVariable(trimmedArg)) {
                // in IdManager Register this entity and getID
                val entityId = IdManager.getEntityId(trimmedArg)
                "E${entityId}"
            } else {
                trimmedArg
            }
        }
        return processedArgs.joinToString(",")
    }
    
    /**
     * Parse the rule string and returnheadandbodyofDepAtomYes
     * 
     * The unified rule format is abbreviated mode:
     * - One dollar rule:/rel(const) <= /rel1*rel2(const2)
     * - Binary rules:/rel <= /rel1*INVERSE_/rel2
     * 
     * @param ruleStr rule string
     * @return Pair<DepAtom, DepAtom?> headandbodyatomic representation of
     */
    fun parseRule(ruleStr: String): Pair<DepAtom, DepAtom?> {
        require(ruleStr.contains("<=")) { "Rule malformed: missing '<='" }
        
        debug("original rules: $ruleStr")
        
        // Preprocessing: Replace entities containing special characters
        var preprocessedRule = preprocessRule(ruleStr)
        // replace me_myself_i,Y or Y,me_myself_i or me_myself_i,X or X,me_myself_i for X,X
        preprocessedRule = preprocessedRule
            .replace("me_myself_i,Y", "X,X")
            .replace("Y,me_myself_i", "X,X")
            .replace("me_myself_i,X", "X,X")
            .replace("X,me_myself_i", "X,X")

        debug("After preprocessing: $preprocessedRule")
        
        val (headPart, bodyPart) = preprocessedRule.split("<=", limit = 2).map { it.trim() }
        
        // Yes head/body Normalize separately to ensure calling the same function
        val normHead = normalizeAtomToSimplified(headPart)
        val normBody = normalizeAtomToSimplified(bodyPart)
        debug("After normalization: $normHead <= $normBody")
        
        // parseheadandbodyforDepAtom
        val headAtom = parseSimplifiedAtom(normHead)
        debug("HeadAtomparse: relationId=${headAtom.relationId}, entityId=${headAtom.entityId}")
        
        val bodyAtom = if (normBody.isNotBlank()) {
            val atom = parseSimplifiedAtom(normBody)
            debug("BodyAtomparse: relationId=${atom.relationId}, entityId=${atom.entityId}")
            atom
        } else null
        
        return Pair(headAtom, bodyAtom)
    }
    
    /**
     * Parse atoms in simplified form
     * Format:
     * 1. relation(constant) - Unary atoms, with constant constraints
     * 2. relation(*) - Existential atoms, with intermediate variables but no constant constraints
     * 3. relation - binary atom
     * 4. rel1*rel2*rel3(constant) - Relational paths, with constant constraints
     * 5. rel1*rel2*rel3(*) - relational path, existential atom
     * 
     * @return DepAtom
     */
    private fun parseSimplifiedAtom(atomStr: String): DepAtom {
        // Check if there are parentheses
        val hasParens = '(' in atomStr && ')' in atomStr
        
        val (relationPath, constantPart) = if (hasParens) {
            val relationPart = atomStr.substringBefore('(').trim()
            val constantPart = atomStr.substringAfter('(').substringBefore(')').trim()
            Pair(relationPart, constantPart)
        } else {
            Pair(atomStr.trim(), null)
        }
        
        // Resolve relationship paths (which may contain*multiple relationships connected)
        val relations = if ('*' in relationPath) {
            relationPath.split('*').map { it.trim() }
        } else {
            listOf(relationPath)
        }
        
        // Get relationshipID
        val relationIds = relations.map { IdManager.getRelationId(it) }
        val finalRelationId = if (relationIds.size == 1) {
            relationIds[0]
        } else {
            RelationPath.encode(relationIds.toLongArray())
        }
        
        // Get entityID
        val entityId = when {
            constantPart == null -> IdManager.getYId() // No parentheses, binary rules
            constantPart == "*" -> 0 // Existential atoms, with intermediate variables but no constants
            isEntityPlaceholder(constantPart) -> constantPart.substring(1).toInt() // placeholder, removeE
            else -> IdManager.getEntityId(constantPart) // Common entity name
        }
        
        return DepAtom(finalRelationId, entityId)
    }
    

    /**
     * Convert full format rules to abbreviated format
     * 
     * Conversion rules:
     * 1. One dollar rule:rel(X,/m/const) <= body1(X,A), body2(A,/m/const2)
     *    -> rel(/m/const) <= body_path(/m/const2)
     * 2. Binary rules:rel(X,Y) <= body1(X,A), body2(Y,A)
     *    -> rel <= body_path
     */
    fun normalizeAtomToSimplified(atomPart: String): String {
        val atom = atomPart.trim()
        if (atom.isBlank()) return ""

        val atoms = parseBodyAtoms(atom)
        if (atoms.size > 1) {
            return normalizeAtomListToSimplified(atoms)
        }

        // single atom Process
        if ('(' !in atom || ')' !in atom) {
            return atom
        }

        val parenContent = atom.substringAfter('(').substringBefore(')')
        if (',' !in parenContent) {
            return atom
        }

        val relation = atom.substringBefore('(').trim()
        val args = parenContent.split(',').map { it.trim() }

        val isSelfLoop = args.size == 2 && args[0] == args[1]
        if (isSelfLoop) return "$relation(X)"

        val constant = args.firstOrNull { !isVariable(it) }
        if (constant != null && args.size == 2) {
            val varPos = args.indexOfFirst { isVariable(it) }
            return when (varPos) {
                0 -> "$relation($constant)"
                1 -> "INVERSE_$relation($constant)"
                else -> "$relation($constant)"
            }
        }

        if (args.size == 2) {
            if (args[0] == "X" && args[1] == "Y") {
                return relation
            }
            if (args[0] == "Y" && args[1] == "X") {
                return "INVERSE_$relation"
            }

            val freeVars = args.filter { it == "X" || it == "Y" }
            if (freeVars.size == 1) {
                val freeVar = freeVars[0]
                val inverse = args.indexOf(freeVar) == 1
                val rel = if (inverse) "INVERSE_$relation" else relation
                return "$rel(*)"
            }
        }

        return relation
    }

    private fun normalizeAtomListToSimplified(atoms: List<String>): String {
        val parsedAtoms = atoms.map { atom ->
            mapOf(
                "relation" to extractRelationFromAtom(atom),
                "args" to extractVariables(atom)
            )
        }

        val varCounts = mutableMapOf<String, Int>()
        val constants = mutableListOf<String>()
        parsedAtoms.forEach { atom ->
            val args = atom["args"] as List<String>
            args.forEach { arg ->
                if (isVariable(arg)) {
                    varCounts[arg] = (varCounts[arg] ?: 0) + 1
                } else {
                    constants.add(arg)
                }
            }
        }

        val freeVars = listOf("X", "Y").filter { varCounts.containsKey(it) }
        val fallbackPath = parsedAtoms.joinToString("*") { it["relation"] as String }

        return when {
            freeVars.size >= 2 -> {
                val path = buildRelationPath(parsedAtoms, freeVars[0], freeVars[1]) ?: fallbackPath
                path
            }
            freeVars.size == 1 -> {
                val freeVar = freeVars[0]
                val target = constants.firstOrNull()
                    ?: varCounts.keys.firstOrNull { it != freeVar }
                val path = buildRelationPath(parsedAtoms, freeVar, target) ?: fallbackPath
                if (target != null && !isVariable(target)) "$path($target)" else "$path(*)"
            }
            else -> fallbackPath
        }
    }

    private fun buildRelationPath(
        parsedAtoms: List<Map<String, Any>>,
        start: String?,
        end: String?
    ): String? {
        if (start == null || end == null) return null

        data class Edge(val to: String, val relation: String)

        val graph = mutableMapOf<String, MutableList<Edge>>()
        parsedAtoms.forEach { atom ->
            val relation = atom["relation"] as String
            val args = atom["args"] as List<String>
            if (args.size < 2) return@forEach
            val a0 = args[0]
            val a1 = args[1]
            graph.getOrPut(a0) { mutableListOf() }.add(Edge(a1, relation))
            graph.getOrPut(a1) { mutableListOf() }.add(Edge(a0, "INVERSE_$relation"))
        }

        val visited = mutableSetOf<String>()
        val queue: ArrayDeque<Pair<String, List<String>>> = ArrayDeque()
        queue.add(start to emptyList())
        visited.add(start)

        while (queue.isNotEmpty()) {
            val (node, path) = queue.removeFirst()
            if (node == end) return path.joinToString("*")
            val edges = graph[node].orEmpty()
            edges.forEach { edge ->
                if (edge.to !in visited) {
                    visited.add(edge.to)
                    queue.add(edge.to to (path + edge.relation))
                }
            }
        }

        return null
    }
    
    
    /**
     * Extract relation name from atom
     */
    fun extractRelationFromAtom(atom: String): String {
        return if ('(' in atom) {
            atom.substringBefore('(').trim()
        } else {
            atom.trim()
        }
    }
    
    /**
     * Extract variables from atoms (including normalization me_myself_i) 
     * Improved version: Correctly handles parentheses and commas in entity names
     */
    fun extractVariables(atom: String): List<String> {
        if ('(' !in atom || ')' !in atom) {
            return emptyList()
        }
        
        // Find the outermost bracket pair
        val firstParen = atom.indexOf('(')
        val lastParen = atom.lastIndexOf(')')
        
        if (firstParen >= lastParen || firstParen == -1 || lastParen == -1) {
            return emptyList()
        }
        
        val varPart = atom.substring(firstParen + 1, lastParen)
        
        // Use smart splitting: only at bracket level for0separated by commas
        val variables = smartSplit(varPart)
        
        // Standardize me_myself_i
        return variables
    }
    
    /**
     * Smart string splitting: only at bracket level0separated by commas
     * This will handle it correctly "Tom_Kelly_(footballer,_born_1964),Y" Such a string
     */
    private fun smartSplit(text: String): List<String> {
        val result = mutableListOf<String>()
        var current = StringBuilder()
        var parenDepth = 0
        
        for (char in text) {
            when (char) {
                '(' -> {
                    parenDepth++
                    current.append(char)
                }
                ')' -> {
                    parenDepth--
                    current.append(char)
                }
                ',' -> {
                    if (parenDepth == 0) {
                        // Split only at commas outside brackets
                        result.add(current.toString().trim())
                        current = StringBuilder()
                    } else {
                        // Commas within parentheses are retained
                        current.append(char)
                    }
                }
                else -> current.append(char)
            }
        }
        
        // add last part
        if (current.isNotEmpty()) {
            result.add(current.toString().trim())
        }
        
        return result
    }
    
    /**
     * Parse the atomic list of body parts
     * Since special characters have been replaced by preprocessing, it is safe to use smart segmentation
     */
    fun parseBodyAtoms(bodyPart: String): List<String> {
        // Use smart splitting, only at the bracket level as0separated by commas
        return smartSplit(bodyPart).filter { it.isNotBlank() }
    }
}
