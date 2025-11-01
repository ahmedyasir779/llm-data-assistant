from typing import Dict, Any, List, Tuple, Optional
from enum import Enum
import re


class QueryType(Enum):
    """Types of queries for routing"""
    AGGREGATION = "aggregation"  # sum, average, count, total
    COMPARISON = "comparison"    # compare, versus, difference
    FILTERING = "filtering"      # show, find, where, filter
    STATISTICAL = "statistical"  # correlation, distribution, trend
    EXPLORATION = "exploration"  # what, describe, explain
    VISUALIZATION = "visualization"  # chart, plot, graph, visualize


class QueryIntent(Enum):
    """User intent behind query"""
    SPECIFIC_VALUE = "specific_value"  # "What is X?"
    LIST_ITEMS = "list_items"          # "Show me all X"
    ANALYZE_TREND = "analyze_trend"    # "How has X changed?"
    COMPARE_ENTITIES = "compare"       # "Compare X and Y"
    SUMMARIZE = "summarize"            # "Summarize X"
    RECOMMEND = "recommend"            # "Suggest X"
    EXPLORATION = "exploration"        # "Tell me about X"

class QueryClassifier:
    """
    Classifies queries to determine best search and processing strategy
    """
    
    # Keywords for each query type
    QUERY_PATTERNS = {
        QueryType.AGGREGATION: [
            r'\b(total|sum|average|mean|count|max|maximum|min|minimum)\b',
            r'\bhow many\b',
            r'\bhow much\b',
        ],
        QueryType.COMPARISON: [
            r'\b(compare|versus|vs|difference|between|better|worse)\b',
            r'\bcompared to\b',
            r'\bhigher than\b',
            r'\blower than\b',
        ],
        QueryType.FILTERING: [
            r'\b(show|find|get|list|filter|where|search)\b',
            r'\bwith|having|that have\b',
            r'\bgreater than|less than|equal to\b',
        ],
        QueryType.STATISTICAL: [
            r'\b(correlation|distribution|trend|pattern|relationship)\b',
            r'\bover time\b',
            r'\bstatistics|stats\b',
        ],
        QueryType.EXPLORATION: [
            r'\b(what|which|who|when|where|why|how)\b',
            r'\b(describe|explain|tell me about)\b',
            r'\b(overview|summary|details)\b',
        ],
        QueryType.VISUALIZATION: [
            r'\b(chart|plot|graph|visualize|show chart|create graph)\b',
            r'\b(bar chart|line chart|pie chart|scatter plot)\b',
        ]
    }
    
    # Keywords for query intent
    INTENT_PATTERNS = {
        QueryIntent.SPECIFIC_VALUE: [
            r'^what is\b',
            r'^what was\b',
            r'^how much\b',
            r'\bthe (highest|lowest|best|worst|first|last)\b',
        ],
        QueryIntent.LIST_ITEMS: [
            r'\b(show me|list|give me|find all)\b',
            r'\ball .* that\b',
            r'\bevery .* with\b',
        ],
        QueryIntent.ANALYZE_TREND: [
            r'\b(trend|over time|growth|decline|change)\b',
            r'\bhow .* changed\b',
            r'\b(increase|decrease)d?\b',
        ],
        QueryIntent.COMPARE_ENTITIES: [
            r'\bcompare\b',
            r'\bversus|vs\b',
            r'\bdifference between\b',
        ],
        QueryIntent.SUMMARIZE: [
            r'\bsummarize|summary|overview\b',
            r'\btell me about\b',
            r'\bdescribe\b',
        ],
        QueryIntent.RECOMMEND: [
            r'\b(recommend|suggest|advice|best)\b',
            r'\bshould i\b',
            r'\bwhich .* is better\b',
        ]
    }
    
    def __init__(self):
        """Initialize query classifier"""
        print("🧠 Query classifier initialized")
    
    def classify_query(self, query: str) -> Dict[str, Any]:
        """
        Classify query to determine type and intent
        
        Args:
            query: User query string
            
        Returns:
            Classification results with confidence scores
        """
        query_lower = query.lower()
        
        # Detect query type
        query_type, type_confidence = self._detect_query_type(query_lower)
        
        # Detect intent
        intent, intent_confidence = self._detect_intent(query_lower)
        
        # Extract entities (numbers, column names, etc)
        entities = self._extract_entities(query_lower)
        
        # Determine search strategy
        search_strategy = self._determine_search_strategy(
            query_type, intent, entities
        )
        
        return {
            "query": query,
            "query_type": query_type,
            "type_confidence": type_confidence,
            "intent": intent,
            "intent_confidence": intent_confidence,
            "entities": entities,
            "search_strategy": search_strategy,
            "requires_aggregation": query_type == QueryType.AGGREGATION,
            "requires_filtering": query_type == QueryType.FILTERING,
            "requires_visualization": query_type == QueryType.VISUALIZATION,
        }
    
    def _detect_query_type(self, query: str) -> Tuple[QueryType, float]:
        """Detect primary query type with confidence"""
        scores = {}
        
        for qtype, patterns in self.QUERY_PATTERNS.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    score += 1
            scores[qtype] = score
        
        if not scores or max(scores.values()) == 0:
            return QueryType.EXPLORATION, 0.5
        
        best_type = max(scores.items(), key=lambda x: x[1])
        confidence = min(best_type[1] / 3, 1.0)  # Normalize to 0-1
        
        return best_type[0], confidence
    
    def _detect_intent(self, query: str) -> Tuple[QueryIntent, float]:
        """Detect user intent with confidence"""
        scores = {}
        
        for intent, patterns in self.INTENT_PATTERNS.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    score += 1
            scores[intent] = score
        
        if not scores or max(scores.values()) == 0:
            return QueryIntent.EXPLORATION, 0.5
        
        best_intent = max(scores.items(), key=lambda x: x[1])
        confidence = min(best_intent[1] / 2, 1.0)
        
        return best_intent[0], confidence
    
    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract entities from query"""
        entities = {
            "numbers": [],
            "comparisons": [],
            "columns": [],
            "keywords": []
        }
        
        # Extract numbers
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', query)
        entities["numbers"] = numbers
        
        # Extract comparison operators
        comparisons = re.findall(
            r'\b(greater than|less than|equal to|above|below|over|under)\b',
            query
        )
        entities["comparisons"] = comparisons
        
        # Extract potential column names (words after "by", "of", "in")
        columns = re.findall(r'\b(?:by|of|in|for)\s+(\w+)', query)
        entities["columns"] = columns
        
        # Extract keywords (important nouns/verbs)
        keywords = [
            word for word in query.split()
            if len(word) > 3 and word not in [
                'what', 'which', 'where', 'when', 'show', 'find',
                'that', 'this', 'with', 'from', 'have', 'been'
            ]
        ]
        entities["keywords"] = keywords[:5]  # Top 5 keywords
        
        return entities
    
    def _determine_search_strategy(
        self,
        query_type: QueryType,
        intent: QueryIntent,
        entities: Dict[str, List[str]]
    ) -> str:
        """
        Determine optimal search strategy
        
        Returns:
            Strategy name: "semantic", "keyword", "hybrid", "direct"
        """
        # Direct data access for specific values with numbers
        if (intent == QueryIntent.SPECIFIC_VALUE and 
            query_type == QueryType.AGGREGATION):
            return "direct"
        
        # Hybrid search for filtering with multiple criteria
        if (query_type == QueryType.FILTERING and 
            len(entities["comparisons"]) > 0):
            return "hybrid"
        
        # Semantic search for exploratory questions
        if query_type == QueryType.EXPLORATION:
            return "semantic"
        
        # Keyword search for specific column/value lookups
        if (len(entities["columns"]) > 0 or 
            len(entities["numbers"]) > 0):
            return "keyword"
        
        # Default to hybrid for best coverage
        return "hybrid"
    
    def explain_classification(self, classification: Dict[str, Any]) -> str:
        """
        Generate human-readable explanation of classification
        
        Args:
            classification: Result from classify_query()
            
        Returns:
            Explanation string
        """
        lines = [
            f"Query: '{classification['query']}'",
            f"",
            f"Type: {classification['query_type'].value} "
            f"(confidence: {classification['type_confidence']:.0%})",
            f"Intent: {classification['intent'].value} "
            f"(confidence: {classification['intent_confidence']:.0%})",
            f"",
            f"Entities found:",
        ]
        
        for entity_type, values in classification['entities'].items():
            if values:
                lines.append(f"  - {entity_type}: {', '.join(values)}")
        
        lines.extend([
            f"",
            f"Recommended strategy: {classification['search_strategy']}",
            f"",
            f"Processing flags:",
            f"  - Aggregation needed: {classification['requires_aggregation']}",
            f"  - Filtering needed: {classification['requires_filtering']}",
            f"  - Visualization needed: {classification['requires_visualization']}",
        ])
        
        return "\n".join(lines)


class QueryRewriter:
    """
    Rewrites and expands queries for better search results
    """
    
    # Synonyms for common terms
    SYNONYMS = {
        "price": ["cost", "price", "value", "amount"],
        "product": ["product", "item", "goods", "merchandise"],
        "revenue": ["revenue", "income", "earnings", "sales"],
        "customer": ["customer", "client", "buyer", "user"],
        "high": ["high", "top", "maximum", "best", "greatest"],
        "low": ["low", "bottom", "minimum", "worst", "least"],
    }
    
    def __init__(self):
        """Initialize query rewriter"""
        print("✍️ Query rewriter initialized")
    
    def rewrite_query(self, query: str, expand: bool = True) -> List[str]:
        """
        Rewrite query with variations
        
        Args:
            query: Original query
            expand: Whether to expand with synonyms
            
        Returns:
            List of query variations
        """
        variations = [query]
        
        if expand:
            # Add synonym expansions
            expanded = self._expand_synonyms(query)
            variations.extend(expanded[:3])  # Add top 3 variations
        
        # Add simplified version
        simplified = self._simplify_query(query)
        if simplified != query:
            variations.append(simplified)
        
        return variations
    
    def _expand_synonyms(self, query: str) -> List[str]:
        """Expand query with synonyms"""
        expanded = []
        query_lower = query.lower()
        
        for term, synonyms in self.SYNONYMS.items():
            if term in query_lower:
                for syn in synonyms:
                    if syn != term:
                        expanded_query = query_lower.replace(term, syn)
                        expanded.append(expanded_query)
        
        return expanded
    
    def _simplify_query(self, query: str) -> str:
        """Simplify query by removing common words"""
        stop_words = {
            'what', 'which', 'where', 'when', 'who', 'how',
            'show', 'me', 'the', 'a', 'an', 'is', 'are', 'was', 'were',
            'can', 'you', 'please', 'tell', 'give'
        }
        
        words = query.lower().split()
        filtered = [w for w in words if w not in stop_words]
        
        return ' '.join(filtered)
    
    def extract_keywords(self, query: str) -> List[str]:
        """
        Extract important keywords from query
        
        Args:
            query: User query
            
        Returns:
            List of keywords
        """
        # Remove stop words
        stop_words = {
            'what', 'which', 'where', 'when', 'who', 'how', 'why',
            'show', 'me', 'the', 'a', 'an', 'is', 'are', 'was', 'were',
            'can', 'you', 'please', 'tell', 'give', 'find', 'get'
        }
        
        words = query.lower().split()
        keywords = [
            w.strip('?.,!') for w in words 
            if w not in stop_words and len(w) > 2
        ]
        
        return keywords[:5]  # Return top 5