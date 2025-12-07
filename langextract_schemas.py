"""
LangExtract extraction schemas for multiple document types.

This module defines extraction schemas for various document types:
- Paul Graham essays (personal narratives, startup advice, life lessons)
- Academic papers (methodology, results, citations)
- Technical documentation (features, APIs, examples)
- General documents (topics, entities, key points)

To add a new document type:
1. Define schema defaults in SCHEMA_DEFINITIONS
2. Create a schema function (e.g., get_my_schema())
3. Register it in SCHEMAS dict
4. Add to config.py ARTICLE_CONFIGS with the schema name
"""

import textwrap
import langextract as lx
import os
import json
from pymongo import MongoClient
from typing import Dict, List, Any, Optional


# =============================================================================
# SCHEMA DEFINITIONS (Static defaults for each document type)
# =============================================================================

SCHEMA_DEFINITIONS: Dict[str, Dict[str, List[str]]] = {
    # -------------------------------------------------------------------------
    # Paul Graham Essays - Personal narratives, startup advice, life lessons
    # -------------------------------------------------------------------------
    "paul_graham": {
        "concept_categories": ["technology", "business", "startups", "programming", "art", "education", "life", "philosophy", "writing"],
        "concept_importance": ["high", "medium", "low"],
        "advice_types": ["actionable", "cautionary", "philosophical", "tactical"],
        "advice_domains": ["career", "startups", "learning", "creativity", "relationships", "decision_making"],
        "experience_periods": ["childhood", "college", "grad_school", "viaweb", "yc", "post_yc", "general"],
        "experience_sentiments": ["positive", "negative", "neutral", "mixed"],
        "entity_roles": ["founder", "colleague", "investor", "friend", "company", "institution", "product"],
        "entity_significance": ["major", "minor"],
        "time_decades": ["1960s", "1970s", "1980s", "1990s", "2000s", "2010s", "2020s", "unspecified"],
        "time_specificity": ["exact_year", "approximate", "era"]
    },
    
    # -------------------------------------------------------------------------
    # Academic Papers - Research methodology, results, citations
    # -------------------------------------------------------------------------
    "academic": {
        "section_types": ["abstract", "introduction", "related_work", "methodology", "experiments", "results", "discussion", "conclusion", "references"],
        "contribution_types": ["novel_method", "theoretical", "empirical", "benchmark", "survey", "application"],
        "concept_categories": ["algorithm", "model", "dataset", "metric", "theory", "technique", "framework"],
        "concept_importance": ["primary", "secondary", "background"],
        "entity_types": ["author", "institution", "dataset", "model", "baseline", "benchmark"],
        "entity_significance": ["major", "minor"],
        "claim_types": ["hypothesis", "finding", "limitation", "future_work"],
        "evidence_types": ["quantitative", "qualitative", "theoretical", "experimental"]
    },
    
    # -------------------------------------------------------------------------
    # Technical Documentation - APIs, features, examples
    # -------------------------------------------------------------------------
    "technical": {
        "doc_sections": ["overview", "installation", "configuration", "api_reference", "examples", "troubleshooting", "faq"],
        "content_types": ["concept", "procedure", "reference", "example", "warning", "note"],
        "complexity_levels": ["beginner", "intermediate", "advanced"],
        "entity_types": ["function", "class", "method", "parameter", "config_option", "command", "file", "module"],
        "code_languages": ["python", "javascript", "typescript", "java", "go", "rust", "shell", "other"],
        "requirement_types": ["prerequisite", "dependency", "optional", "recommended"]
    },
    
    # -------------------------------------------------------------------------
    # General Documents - Broad extraction for any document type
    # -------------------------------------------------------------------------
    "general": {
        "topic_categories": ["main_topic", "subtopic", "related_topic"],
        "content_types": ["fact", "opinion", "definition", "example", "quote", "statistic"],
        "importance_levels": ["high", "medium", "low"],
        "entity_types": ["person", "organization", "location", "product", "concept", "event", "date"],
        "sentiment": ["positive", "negative", "neutral"],
        "section_types": ["introduction", "body", "conclusion", "summary"]
    },
    
    # -------------------------------------------------------------------------
    # Financial Documents - Reports, filings, analysis
    # -------------------------------------------------------------------------
    "financial": {
        "document_types": ["10k", "10q", "earnings_call", "annual_report", "prospectus"],
        "metric_types": ["revenue", "profit", "margin", "growth", "ratio", "guidance"],
        "time_periods": ["quarterly", "annual", "ytd", "yoy", "forecast"],
        "entity_types": ["company", "executive", "competitor", "regulator", "auditor"],
        "risk_categories": ["market", "operational", "regulatory", "competitive", "financial"],
        "sentiment": ["positive", "negative", "neutral", "cautious"]
    },
    
    # -------------------------------------------------------------------------
    # Career/Self-Help Documents - Advice, lessons, guidance
    # -------------------------------------------------------------------------
    "career": {
        "advice_categories": ["career_path", "skill_development", "networking", "job_search", "leadership", "work_life_balance"],
        "advice_types": ["actionable", "mindset", "cautionary", "strategic"],
        "skill_types": ["technical", "soft_skill", "domain_knowledge", "tool"],
        "career_stages": ["student", "entry_level", "mid_career", "senior", "executive", "transition"],
        "entity_types": ["person", "company", "role", "industry", "certification", "resource"],
        "importance_levels": ["essential", "recommended", "optional"]
    },
}


# =============================================================================
# CACHE FOR DYNAMIC SCHEMA LOADING
# =============================================================================

_SCHEMA_CACHE: Dict[str, Dict[str, List[str]]] = {}


# =============================================================================
# SCHEMA DEFINITION FUNCTIONS
# =============================================================================

def get_schema_definitions(
    schema_type: str = "paul_graham",
    use_dynamic_loading: bool = True,
    db_name: Optional[str] = None,
    collection_suffix: str = "_metadata_langextract/data"
) -> Dict[str, List[str]]:
    """
    Return the definitions of attributes for a given schema type.
    
    This function provides the allowed values for each metadata attribute category
    that are used in two contexts:
    
    1. **Ingestion (use_dynamic_loading=False)**: During document processing, these
       values are embedded in the LangExtract prompt to guide the LLM on what
       attribute values to assign when extracting metadata from text chunks.
    
    2. **Query Filtering (use_dynamic_loading=True)**: At query time, the function
       loads actual distinct values from MongoDB to ensure filters match stored values.
    
    Parameters:
    schema_type (str): The type of schema ("paul_graham", "academic", "technical", "general", etc.)
    use_dynamic_loading (bool): If True, attempts to fetch values from MongoDB.
    db_name (Optional[str]): MongoDB database name for dynamic loading.
    collection_suffix (str): Suffix to identify the metadata collection.
    
    Returns:
    dict: Schema definitions with attribute categories and their allowed values.
    """
    global _SCHEMA_CACHE
    
    # Validate schema type
    if schema_type not in SCHEMA_DEFINITIONS:
        available = list(SCHEMA_DEFINITIONS.keys())
        raise ValueError(f"Unknown schema type: '{schema_type}'. Available: {available}")
    
    # Get static defaults
    defaults = SCHEMA_DEFINITIONS[schema_type].copy()
    
    # Cache key includes schema type and loading mode
    cache_key = f"{schema_type}_{'dynamic' if use_dynamic_loading else 'static'}"
    
    # Return cached if available
    if cache_key in _SCHEMA_CACHE:
        return _SCHEMA_CACHE[cache_key]
    
    # If not using dynamic loading, return static defaults
    if not use_dynamic_loading:
        _SCHEMA_CACHE[cache_key] = defaults
        return defaults
    
    # Try dynamic loading from MongoDB
    if db_name:
        try:
            uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
            client = MongoClient(uri, serverSelectionTimeoutMS=2000)
            
            if db_name not in client.list_database_names():
                client.close()
                _SCHEMA_CACHE[cache_key] = defaults
                return defaults
            
            db = client[db_name]
            
            # Find the correct collection
            collection_name = None
            for col in db.list_collection_names():
                if col.endswith(collection_suffix):
                    collection_name = col
                    break
            
            if not collection_name:
                client.close()
                _SCHEMA_CACHE[cache_key] = defaults
                return defaults
            
            collection = db[collection_name]
            
            # Query distinct values for each field
            dynamic_schema = {}
            for field in defaults.keys():
                # Try __data__.metadata.field first (LlamaIndex structure)
                values = collection.distinct(f"__data__.metadata.{field}")
                if not values:
                    # Try metadata.field (Standard structure)
                    values = collection.distinct(f"metadata.{field}")
                
                if values:
                    clean_values = sorted([v for v in values if v])
                    dynamic_schema[field] = clean_values if clean_values else defaults[field]
                else:
                    dynamic_schema[field] = defaults[field]
            
            client.close()
            print(f"✅ Loaded dynamic schema from MongoDB: {db_name}/{collection_name}")
            _SCHEMA_CACHE[cache_key] = dynamic_schema
            return dynamic_schema
            
        except Exception as e:
            print(f"Warning: Failed to fetch schema from MongoDB ({e}). Using defaults.")
    
    _SCHEMA_CACHE[cache_key] = defaults
    return defaults


# Backward compatibility alias
def get_paul_graham_schema_definitions(use_dynamic_loading: bool = True) -> Dict[str, List[str]]:
    """
    Backward-compatible function for Paul Graham schema definitions.
    
    Parameters:
    use_dynamic_loading (bool): If True, attempts to fetch values from MongoDB.
    
    Returns:
    dict: Schema definitions for Paul Graham essays.
    """
    db_name = "paul_graham_paul_graham_essay_sentence_splitter" if use_dynamic_loading else None
    return get_schema_definitions(
        schema_type="paul_graham",
        use_dynamic_loading=use_dynamic_loading,
        db_name=db_name
    )


# =============================================================================
# SCHEMA GENERATORS
# =============================================================================

def get_paul_graham_essay_schema() -> Dict[str, Any]:
    """
    Get extraction schema for Paul Graham essays.
    
    Extracts: concepts, advice, experiences, entities, time references
    """
    defs = get_schema_definitions("paul_graham", use_dynamic_loading=False)
    
    prompt = textwrap.dedent(
        f"""\
        Extract structured information from Paul Graham's essay text.
        Focus on identifying key concepts, advice, personal experiences, and entities.
        Use exact text for extractions. Do not paraphrase or create overlapping entities.
        
        For each extraction, provide attributes from these predefined sets:
        
        Concept attributes:
        - category: {defs['concept_categories']}
        - importance: {defs['concept_importance']}
        
        Advice attributes:
        - type: {defs['advice_types']}
        - domain: {defs['advice_domains']}
        
        Experience attributes:
        - period: {defs['experience_periods']}
        - sentiment: {defs['experience_sentiments']}
        
        Entity attributes (people/organizations):
        - role: {defs['entity_roles']}
        - significance: {defs['entity_significance']}
        
        Time attributes:
        - decade: {defs['time_decades']}
        - specificity: {defs['time_specificity']}
        
        Focus on extracting information that would be useful for:
        - Finding specific advice or lessons
        - Identifying personal narratives and experiences
        - Connecting concepts and ideas
        - Timeline construction
        - Thematic analysis
        """
    )
    
    examples = [
        lx.data.ExampleData(
            text="I learned to program in BASIC when I was about 13 years old, on a TRS-80 computer. It was the most exciting thing I'd ever done.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="experience",
                    extraction_text="learned to program in BASIC when I was about 13 years old",
                    attributes={"period": "childhood", "sentiment": "positive"}
                ),
                lx.data.Extraction(
                    extraction_class="concept",
                    extraction_text="program in BASIC",
                    attributes={"category": "programming", "importance": "high"}
                ),
                lx.data.Extraction(
                    extraction_class="entity",
                    extraction_text="TRS-80",
                    attributes={"role": "product", "significance": "minor"}
                ),
                lx.data.Extraction(
                    extraction_class="time",
                    extraction_text="about 13 years old",
                    attributes={"decade": "1970s", "specificity": "approximate"}
                ),
            ],
        ),
        lx.data.ExampleData(
            text="The most important thing is to work on something you're genuinely interested in. Don't force yourself to work on something because you think you should.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="advice",
                    extraction_text="work on something you're genuinely interested in",
                    attributes={"type": "actionable", "domain": "career"}
                ),
                lx.data.Extraction(
                    extraction_class="advice",
                    extraction_text="Don't force yourself to work on something because you think you should",
                    attributes={"type": "cautionary", "domain": "decision_making"}
                ),
                lx.data.Extraction(
                    extraction_class="concept",
                    extraction_text="genuinely interested",
                    attributes={"category": "life", "importance": "high"}
                ),
            ],
        ),
        lx.data.ExampleData(
            text="When we started Y Combinator in 2005, we had no idea it would grow into what it is today. We just wanted to help startups the way we wished someone had helped us.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="entity",
                    extraction_text="Y Combinator",
                    attributes={"role": "company", "significance": "major"}
                ),
                lx.data.Extraction(
                    extraction_class="time",
                    extraction_text="2005",
                    attributes={"decade": "2000s", "specificity": "exact_year"}
                ),
                lx.data.Extraction(
                    extraction_class="experience",
                    extraction_text="started Y Combinator in 2005",
                    attributes={"period": "yc", "sentiment": "positive"}
                ),
                lx.data.Extraction(
                    extraction_class="concept",
                    extraction_text="help startups",
                    attributes={"category": "startups", "importance": "high"}
                ),
            ],
        ),
        lx.data.ExampleData(
            text="The best way to get startup ideas is to become the kind of person who has them. Live in the future and build what seems interesting.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="advice",
                    extraction_text="become the kind of person who has them",
                    attributes={"type": "philosophical", "domain": "startups"}
                ),
                lx.data.Extraction(
                    extraction_class="advice",
                    extraction_text="Live in the future and build what seems interesting",
                    attributes={"type": "actionable", "domain": "creativity"}
                ),
                lx.data.Extraction(
                    extraction_class="concept",
                    extraction_text="startup ideas",
                    attributes={"category": "startups", "importance": "high"}
                ),
            ],
        ),
    ]
    
    return {"prompt": prompt, "examples": examples}


def get_simple_paul_graham_schema() -> Dict[str, Any]:
    """
    Get a simpler extraction schema for Paul Graham essays.
    
    Lighter-weight version extracting: topics, advice, entities
    """
    prompt = textwrap.dedent(
        """\
        Extract main themes, advice, and key entities from Paul Graham's essay.
        Use exact text. Do not paraphrase.
        
        Topic attributes:
        - theme: ["startups", "programming", "education", "creativity", "career", "life_lessons"]
        
        Advice attributes:
        - actionable: ["yes", "no"]
        
        Entity attributes:
        - type: ["person", "company", "technology", "concept"]
        """
    )
    
    examples = [
        lx.data.ExampleData(
            text="If you want to start a startup, the most important thing is to build something people want.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="topic",
                    extraction_text="start a startup",
                    attributes={"theme": "startups"}
                ),
                lx.data.Extraction(
                    extraction_class="advice",
                    extraction_text="build something people want",
                    attributes={"actionable": "yes"}
                ),
            ],
        ),
        lx.data.ExampleData(
            text="I met Robert Morris at Harvard, and we became good friends.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="entity",
                    extraction_text="Robert Morris",
                    attributes={"type": "person"}
                ),
                lx.data.Extraction(
                    extraction_class="entity",
                    extraction_text="Harvard",
                    attributes={"type": "company"}
                ),
            ],
        ),
    ]
    
    return {"prompt": prompt, "examples": examples}


def get_academic_paper_schema() -> Dict[str, Any]:
    """
    Get extraction schema for academic/research papers.
    
    Extracts: contributions, methodology, results, entities, claims
    """
    defs = get_schema_definitions("academic", use_dynamic_loading=False)
    
    prompt = textwrap.dedent(
        f"""\
        Extract structured information from this academic paper.
        Focus on identifying key contributions, methodology, results, and citations.
        Use exact text for extractions. Do not paraphrase.
        
        For each extraction, provide attributes from these predefined sets:
        
        Contribution attributes:
        - type: {defs['contribution_types']}
        - section: {defs['section_types']}
        
        Concept attributes:
        - category: {defs['concept_categories']}
        - importance: {defs['concept_importance']}
        
        Entity attributes:
        - type: {defs['entity_types']}
        - significance: {defs['entity_significance']}
        
        Claim attributes:
        - type: {defs['claim_types']}
        - evidence: {defs['evidence_types']}
        
        Focus on extracting:
        - Main contributions and novelty
        - Methodology and experimental setup
        - Key results and findings
        - Limitations and future work
        - Important citations and comparisons
        """
    )
    
    examples = [
        lx.data.ExampleData(
            text="We propose a novel attention mechanism that achieves state-of-the-art results on machine translation benchmarks.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="contribution",
                    extraction_text="novel attention mechanism",
                    attributes={"type": "novel_method", "section": "introduction"}
                ),
                lx.data.Extraction(
                    extraction_class="claim",
                    extraction_text="achieves state-of-the-art results",
                    attributes={"type": "finding", "evidence": "quantitative"}
                ),
                lx.data.Extraction(
                    extraction_class="concept",
                    extraction_text="attention mechanism",
                    attributes={"category": "model", "importance": "primary"}
                ),
            ],
        ),
        lx.data.ExampleData(
            text="Our experiments on the WMT 2014 dataset show a BLEU score improvement of 2.3 points over the baseline Transformer model.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="entity",
                    extraction_text="WMT 2014 dataset",
                    attributes={"type": "dataset", "significance": "major"}
                ),
                lx.data.Extraction(
                    extraction_class="entity",
                    extraction_text="Transformer model",
                    attributes={"type": "baseline", "significance": "major"}
                ),
                lx.data.Extraction(
                    extraction_class="claim",
                    extraction_text="BLEU score improvement of 2.3 points",
                    attributes={"type": "finding", "evidence": "quantitative"}
                ),
            ],
        ),
    ]
    
    return {"prompt": prompt, "examples": examples}


def get_technical_doc_schema() -> Dict[str, Any]:
    """
    Get extraction schema for technical documentation.
    
    Extracts: concepts, procedures, API references, examples
    """
    defs = get_schema_definitions("technical", use_dynamic_loading=False)
    
    prompt = textwrap.dedent(
        f"""\
        Extract structured information from this technical documentation.
        Focus on identifying key concepts, procedures, API elements, and examples.
        Use exact text for extractions. Do not paraphrase.
        
        For each extraction, provide attributes from these predefined sets:
        
        Content attributes:
        - type: {defs['content_types']}
        - section: {defs['doc_sections']}
        - complexity: {defs['complexity_levels']}
        
        Entity attributes:
        - type: {defs['entity_types']}
        - language: {defs['code_languages']}
        
        Requirement attributes:
        - type: {defs['requirement_types']}
        
        Focus on extracting:
        - Key concepts and definitions
        - Step-by-step procedures
        - Function/class/API references
        - Code examples
        - Prerequisites and dependencies
        - Warnings and important notes
        """
    )
    
    examples = [
        lx.data.ExampleData(
            text="To install the package, run `pip install llamaindex`. Python 3.8 or higher is required.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="procedure",
                    extraction_text="pip install llamaindex",
                    attributes={"type": "procedure", "section": "installation", "complexity": "beginner"}
                ),
                lx.data.Extraction(
                    extraction_class="requirement",
                    extraction_text="Python 3.8 or higher",
                    attributes={"type": "prerequisite"}
                ),
                lx.data.Extraction(
                    extraction_class="entity",
                    extraction_text="pip install",
                    attributes={"type": "command", "language": "shell"}
                ),
            ],
        ),
        lx.data.ExampleData(
            text="The `VectorStoreIndex.from_documents()` method creates an index from a list of Document objects.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="entity",
                    extraction_text="VectorStoreIndex.from_documents()",
                    attributes={"type": "method", "language": "python"}
                ),
                lx.data.Extraction(
                    extraction_class="concept",
                    extraction_text="creates an index from a list of Document objects",
                    attributes={"type": "concept", "section": "api_reference", "complexity": "intermediate"}
                ),
            ],
        ),
    ]
    
    return {"prompt": prompt, "examples": examples}


def get_general_schema() -> Dict[str, Any]:
    """
    Get a general-purpose extraction schema for any document type.
    
    Extracts: topics, facts, entities, key points
    """
    defs = get_schema_definitions("general", use_dynamic_loading=False)
    
    prompt = textwrap.dedent(
        f"""\
        Extract structured information from this document.
        Focus on identifying key topics, facts, entities, and important points.
        Use exact text for extractions. Do not paraphrase.
        
        For each extraction, provide attributes from these predefined sets:
        
        Topic attributes:
        - category: {defs['topic_categories']}
        - importance: {defs['importance_levels']}
        
        Content attributes:
        - type: {defs['content_types']}
        - section: {defs['section_types']}
        
        Entity attributes:
        - type: {defs['entity_types']}
        - sentiment: {defs['sentiment']}
        
        Focus on extracting:
        - Main topics and themes
        - Key facts and statistics
        - Important entities (people, organizations, places)
        - Notable quotes and definitions
        - Summary points
        """
    )
    
    examples = [
        lx.data.ExampleData(
            text="The company reported a 25% increase in revenue for Q3 2024, exceeding analyst expectations.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="fact",
                    extraction_text="25% increase in revenue",
                    attributes={"type": "statistic", "importance": "high"}
                ),
                lx.data.Extraction(
                    extraction_class="entity",
                    extraction_text="Q3 2024",
                    attributes={"type": "date", "sentiment": "positive"}
                ),
            ],
        ),
        lx.data.ExampleData(
            text="According to CEO Jane Smith, the new product launch will focus on sustainability and innovation.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="entity",
                    extraction_text="Jane Smith",
                    attributes={"type": "person", "sentiment": "neutral"}
                ),
                lx.data.Extraction(
                    extraction_class="topic",
                    extraction_text="sustainability and innovation",
                    attributes={"category": "main_topic", "importance": "high"}
                ),
            ],
        ),
    ]
    
    return {"prompt": prompt, "examples": examples}


def get_financial_doc_schema() -> Dict[str, Any]:
    """
    Get extraction schema for financial documents (10-K, 10-Q, earnings).
    
    Extracts: metrics, risks, guidance, entities
    """
    defs = get_schema_definitions("financial", use_dynamic_loading=False)
    
    prompt = textwrap.dedent(
        f"""\
        Extract structured information from this financial document.
        Focus on identifying key metrics, risks, guidance, and business entities.
        Use exact text for extractions. Do not paraphrase.
        
        For each extraction, provide attributes from these predefined sets:
        
        Metric attributes:
        - type: {defs['metric_types']}
        - period: {defs['time_periods']}
        
        Risk attributes:
        - category: {defs['risk_categories']}
        - sentiment: {defs['sentiment']}
        
        Entity attributes:
        - type: {defs['entity_types']}
        
        Focus on extracting:
        - Revenue, profit, and growth metrics
        - Forward-looking guidance
        - Risk factors and concerns
        - Key executives and competitors
        - Regulatory mentions
        """
    )
    
    examples = [
        lx.data.ExampleData(
            text="Net revenue increased 18% year-over-year to $8.3 billion, driven by strong growth in our cloud services segment.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="metric",
                    extraction_text="Net revenue increased 18%",
                    attributes={"type": "growth", "period": "yoy"}
                ),
                lx.data.Extraction(
                    extraction_class="metric",
                    extraction_text="$8.3 billion",
                    attributes={"type": "revenue", "period": "quarterly"}
                ),
            ],
        ),
        lx.data.ExampleData(
            text="We face significant competition from established players including Microsoft, Google, and Amazon in the cloud computing market.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="risk",
                    extraction_text="significant competition",
                    attributes={"category": "competitive", "sentiment": "cautious"}
                ),
                lx.data.Extraction(
                    extraction_class="entity",
                    extraction_text="Microsoft",
                    attributes={"type": "competitor"}
                ),
                lx.data.Extraction(
                    extraction_class="entity",
                    extraction_text="Google",
                    attributes={"type": "competitor"}
                ),
            ],
        ),
    ]
    
    return {"prompt": prompt, "examples": examples}


def get_career_advice_schema() -> Dict[str, Any]:
    """
    Get extraction schema for career/self-help documents.
    
    Extracts: advice, skills, career stages, resources
    """
    defs = get_schema_definitions("career", use_dynamic_loading=False)
    
    prompt = textwrap.dedent(
        f"""\
        Extract structured information from this career advice document.
        Focus on identifying actionable advice, skills mentioned, and career guidance.
        Use exact text for extractions. Do not paraphrase.
        
        For each extraction, provide attributes from these predefined sets:
        
        Advice attributes:
        - category: {defs['advice_categories']}
        - type: {defs['advice_types']}
        - importance: {defs['importance_levels']}
        
        Skill attributes:
        - type: {defs['skill_types']}
        - stage: {defs['career_stages']}
        
        Entity attributes:
        - type: {defs['entity_types']}
        
        Focus on extracting:
        - Actionable career advice
        - Skills to develop
        - Career path guidance
        - Resources and tools mentioned
        - Success stories and examples
        """
    )
    
    examples = [
        lx.data.ExampleData(
            text="To break into AI, start by mastering Python and taking online courses in machine learning fundamentals.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="advice",
                    extraction_text="mastering Python",
                    attributes={"category": "skill_development", "type": "actionable", "importance": "essential"}
                ),
                lx.data.Extraction(
                    extraction_class="skill",
                    extraction_text="Python",
                    attributes={"type": "technical", "stage": "entry_level"}
                ),
                lx.data.Extraction(
                    extraction_class="skill",
                    extraction_text="machine learning fundamentals",
                    attributes={"type": "domain_knowledge", "stage": "entry_level"}
                ),
            ],
        ),
        lx.data.ExampleData(
            text="Networking is crucial at any career stage. Attend industry conferences and engage actively on LinkedIn.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="advice",
                    extraction_text="Attend industry conferences",
                    attributes={"category": "networking", "type": "actionable", "importance": "recommended"}
                ),
                lx.data.Extraction(
                    extraction_class="entity",
                    extraction_text="LinkedIn",
                    attributes={"type": "resource"}
                ),
            ],
        ),
    ]
    
    return {"prompt": prompt, "examples": examples}


# =============================================================================
# SCHEMA REGISTRY
# =============================================================================

SCHEMAS: Dict[str, callable] = {
    # Paul Graham essays
    "paul_graham_detailed": get_paul_graham_essay_schema,
    "paul_graham_simple": get_simple_paul_graham_schema,
    
    # Academic papers
    "academic": get_academic_paper_schema,
    "academic_paper": get_academic_paper_schema,  # alias
    
    # Technical documentation
    "technical": get_technical_doc_schema,
    "technical_doc": get_technical_doc_schema,  # alias
    
    # General purpose
    "general": get_general_schema,
    
    # Financial documents
    "financial": get_financial_doc_schema,
    "financial_doc": get_financial_doc_schema,  # alias
    
    # Career/self-help
    "career": get_career_advice_schema,
    "career_advice": get_career_advice_schema,  # alias
}


def get_schema(schema_name: str = "paul_graham_detailed") -> Dict[str, Any]:
    """
    Get extraction schema by name.
    
    Args:
        schema_name (str): Name of the schema to retrieve.
            Available schemas:
            - "paul_graham_detailed", "paul_graham_simple" (essays)
            - "academic", "academic_paper" (research papers)
            - "technical", "technical_doc" (documentation)
            - "general" (any document)
            - "financial", "financial_doc" (financial reports)
            - "career", "career_advice" (career guides)
        
    Returns:
        dict: Schema configuration with 'prompt' and 'examples'
    
    Raises:
        ValueError: If schema_name is not found
    """
    if schema_name not in SCHEMAS:
        available = list(SCHEMAS.keys())
        raise ValueError(f"Unknown schema: '{schema_name}'. Available: {available}")
    
    return SCHEMAS[schema_name]()


def list_available_schemas() -> List[str]:
    """Return list of all available schema names."""
    return list(SCHEMAS.keys())


def get_schema_type_from_name(schema_name: str) -> str:
    """
    Get the schema type (for definitions) from a schema name.
    
    Maps schema names like "paul_graham_detailed" to definition types like "paul_graham".
    """
    schema_type_map = {
        "paul_graham_detailed": "paul_graham",
        "paul_graham_simple": "paul_graham",
        "academic": "academic",
        "academic_paper": "academic",
        "technical": "technical",
        "technical_doc": "technical",
        "general": "general",
        "financial": "financial",
        "financial_doc": "financial",
        "career": "career",
        "career_advice": "career",
    }
    return schema_type_map.get(schema_name, "general")


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    print("\n🔧 LangExtract Schema System")
    print(f"\nAvailable Schemas ({len(SCHEMAS)}):")
    for name in sorted(set(SCHEMAS.keys())):
        schema_type = get_schema_type_from_name(name)
        print(f"  - {name} (type: {schema_type})")
    
    print(f"\nSchema Definition Types ({len(SCHEMA_DEFINITIONS)}):")
    for schema_type, defs in SCHEMA_DEFINITIONS.items():
        print(f"  - {schema_type}: {len(defs)} attributes")
    
    print("\n✅ Testing schema retrieval...")
    for name in ["paul_graham_detailed", "academic", "general"]:
        schema = get_schema(name)
        print(f"  {name}: {len(schema['examples'])} examples")
