"""
Article Configuration for RAG System

This module provides centralized configuration for different articles/documents
that can be processed by the RAG system. It allows easy switching between
documents without modifying the main script.

Usage:
------
1. Add a new article configuration to ARTICLE_CONFIGS
2. Set ACTIVE_ARTICLE to the article key you want to process
3. Import and use get_active_config() in your scripts

Example:
--------
    from config import get_active_config
    
    config = get_active_config()
    article_directory = config["directory"]
    article_name = config["filename"]
    schema_name = config["schema"]
"""

from typing import Dict, Any, Optional, List

# =============================================================================
# ARTICLE CONFIGURATIONS
# =============================================================================
# Add new articles here. Each article needs:
# - directory: Folder name under ./data/
# - filename: PDF filename
# - schema: LangExtract schema to use (see langextract_schemas.py)
# - description: Human-readable description
# - sample_queries: Example queries for this document (optional)

ARTICLE_CONFIGS: Dict[str, Dict[str, Any]] = {
    # -------------------------------------------------------------------------
    # Paul Graham Essays
    # -------------------------------------------------------------------------
    "paul_graham_essay": {
        "directory": "paul_graham",
        "filename": "paul_graham_essay.pdf",
        "schema": "paul_graham_detailed",
        "description": "Paul Graham's essay about his life journey through programming, startups, and Y Combinator",
        "sample_queries": [
            "What did Paul Graham do in 1980, in 1996 and in 2019?",
            "What strategic advice is given about startups?",
            "What experiences from the 1990s are described?",
            "How did rejecting prestigious conventional paths lead to the most influential creative projects?",
        ],
    },
    "how_to_do_great_work": {
        "directory": "paul_graham",
        "filename": "How_to_do_great_work.pdf",
        "schema": "paul_graham_detailed",  # Same schema works for PG essays
        "description": "Paul Graham's essay on how to do great work",
        "sample_queries": [
            "What is the key to doing great work?",
            "How should one choose what to work on?",
            "What role does curiosity play in great work?",
        ],
    },
    
    # -------------------------------------------------------------------------
    # Academic Papers (placeholder - requires new schema)
    # -------------------------------------------------------------------------
    "attention_paper": {
        "directory": "attention",
        "filename": "attention_all.pdf",
        "schema": "general",  # TODO: Create academic_paper schema
        "description": "Attention Is All You Need - Transformer architecture paper",
        "sample_queries": [
            "What is the main contribution of the paper?",
            "How does self-attention work?",
            "What are the results on machine translation?",
        ],
    },
    
    # -------------------------------------------------------------------------
    # Technical Documents (placeholder - requires new schema)
    # -------------------------------------------------------------------------
    "metagpt": {
        "directory": "metagpt",
        "filename": "metagpt.pdf",
        "schema": "general",  # TODO: Create technical_doc schema
        "description": "MetaGPT technical documentation",
        "sample_queries": [
            "What is MetaGPT?",
            "How does the multi-agent system work?",
        ],
    },
    
    # -------------------------------------------------------------------------
    # Financial Documents (placeholder)
    # -------------------------------------------------------------------------
    "uber_10q": {
        "directory": "uber",
        "filename": "uber_10q_march_2022.pdf",
        "schema": "general",  # TODO: Create financial_doc schema
        "description": "Uber 10-Q financial report for Q1 2022",
        "sample_queries": [
            "What was Uber's revenue in Q1 2022?",
            "What are the key risk factors mentioned?",
        ],
    },
    
    # -------------------------------------------------------------------------
    # Career/Education Documents
    # -------------------------------------------------------------------------
    "andrew_ng_career": {
        "directory": "andrew",
        "filename": "eBook-How-to-Build-a-Career-in-AI.pdf",
        "schema": "general",  # Could use paul_graham schema or create career_advice schema
        "description": "Andrew Ng's guide on building a career in AI",
        "sample_queries": [
            "What skills are needed for an AI career?",
            "How should one start learning AI?",
        ],
    },
}

# =============================================================================
# ACTIVE ARTICLE SELECTION
# =============================================================================
# Change this to switch which article is being processed
# Must be a key from ARTICLE_CONFIGS

ACTIVE_ARTICLE = "paul_graham_essay"
# ACTIVE_ARTICLE = "how_to_do_great_work"
# ACTIVE_ARTICLE = "attention_paper"
# ACTIVE_ARTICLE = "metagpt"
# ACTIVE_ARTICLE = "uber_10q"
# ACTIVE_ARTICLE = "andrew_ng_career"


# =============================================================================
# RAG PIPELINE SETTINGS
# =============================================================================
# These settings can be overridden per-article if needed

DEFAULT_RAG_SETTINGS: Dict[str, Any] = {
    # Chunking settings
    "chunk_size": 256,
    "chunk_overlap": 64,
    "chunk_method": "sentence_splitter",
    
    # Metadata extraction: None, "entity", "langextract", "both"
    "metadata": "langextract",
    
    # Entity filtering
    "use_entity_filtering": True,
    
    # Fusion retrieval settings
    "similarity_top_k_fusion": 48,
    "num_queries": 1,
    "fusion_top_n": 42,
    "rerank_top_n": 32,
    "num_nodes": 1,  # For PrevNextNodePostprocessor
    
    # Debug settings
    "page_filter_verbose": True,
}

# Per-article RAG settings overrides (optional)
ARTICLE_RAG_OVERRIDES: Dict[str, Dict[str, Any]] = {
    # Example: Use different settings for academic papers
    "attention_paper": {
        "chunk_size": 512,  # Larger chunks for academic papers
        "chunk_overlap": 128,
        "metadata": "entity",  # Use entity extraction only (no PG-specific schema)
    },
    "uber_10q": {
        "chunk_size": 512,
        "chunk_overlap": 128,
        "metadata": "entity",
    },
    "metagpt": {
        "metadata": "entity",
    },
    "andrew_ng_career": {
        "metadata": "entity",
    },
}


# =============================================================================
# EMBEDDING MODEL SETTINGS
# =============================================================================

EMBEDDING_CONFIG: Dict[str, Any] = {
    "model_name": "text-embedding-3-small",
    "dimension": 1536,
    "short_name": "openai_embedding_3_small",
}


# =============================================================================
# DATABASE SETTINGS
# =============================================================================

DATABASE_CONFIG: Dict[str, str] = {
    "milvus_uri": "http://localhost:19530",
    "mongo_uri": "mongodb://localhost:27017/",
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_active_config() -> Dict[str, Any]:
    """
    Get the configuration for the currently active article.
    
    Returns:
        Dict containing: directory, filename, schema, description, sample_queries
    
    Raises:
        ValueError: If ACTIVE_ARTICLE is not found in ARTICLE_CONFIGS
    """
    if ACTIVE_ARTICLE not in ARTICLE_CONFIGS:
        available = list(ARTICLE_CONFIGS.keys())
        raise ValueError(
            f"Unknown article: '{ACTIVE_ARTICLE}'. "
            f"Available articles: {available}"
        )
    return ARTICLE_CONFIGS[ACTIVE_ARTICLE]


def get_rag_settings(article_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Get RAG pipeline settings for an article, with any article-specific overrides applied.
    
    Parameters:
        article_key: Article key to get settings for. If None, uses ACTIVE_ARTICLE.
    
    Returns:
        Dict containing all RAG settings with any overrides applied.
    """
    key = article_key or ACTIVE_ARTICLE
    
    # Start with defaults
    settings = DEFAULT_RAG_SETTINGS.copy()
    
    # Apply article-specific overrides if any
    if key in ARTICLE_RAG_OVERRIDES:
        settings.update(ARTICLE_RAG_OVERRIDES[key])
    
    return settings


def get_article_info(article_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Get complete configuration for an article including RAG settings.
    
    Parameters:
        article_key: Article key to get info for. If None, uses ACTIVE_ARTICLE.
    
    Returns:
        Dict with article config merged with RAG settings.
    """
    key = article_key or ACTIVE_ARTICLE
    
    if key not in ARTICLE_CONFIGS:
        available = list(ARTICLE_CONFIGS.keys())
        raise ValueError(f"Unknown article: '{key}'. Available: {available}")
    
    # Merge article config with RAG settings
    config = ARTICLE_CONFIGS[key].copy()
    config["rag_settings"] = get_rag_settings(key)
    config["embedding"] = EMBEDDING_CONFIG.copy()
    config["database"] = DATABASE_CONFIG.copy()
    
    return config


def list_available_articles() -> List[str]:
    """
    List all available article keys.
    
    Returns:
        List of article configuration keys.
    """
    return list(ARTICLE_CONFIGS.keys())


def print_article_summary(article_key: Optional[str] = None) -> None:
    """
    Print a summary of an article's configuration.
    
    Parameters:
        article_key: Article key to summarize. If None, uses ACTIVE_ARTICLE.
    """
    key = article_key or ACTIVE_ARTICLE
    config = get_article_info(key)
    rag = config["rag_settings"]
    
    print(f"\n{'='*60}")
    print(f"📄 Article Configuration: {key}")
    print(f"{'='*60}")
    print(f"   Directory: ./data/{config['directory']}/")
    print(f"   Filename:  {config['filename']}")
    print(f"   Schema:    {config['schema']}")
    print(f"   Description: {config['description']}")
    print(f"\n   RAG Settings:")
    print(f"   ├─ Chunk Size: {rag['chunk_size']}")
    print(f"   ├─ Chunk Overlap: {rag['chunk_overlap']}")
    print(f"   ├─ Metadata: {rag['metadata'] or 'None'}")
    print(f"   ├─ Entity Filtering: {rag['use_entity_filtering']}")
    print(f"   └─ Rerank Top N: {rag['rerank_top_n']}")
    
    if config.get("sample_queries"):
        print(f"\n   Sample Queries:")
        for i, q in enumerate(config["sample_queries"][:3], 1):
            print(f"   {i}. {q[:60]}{'...' if len(q) > 60 else ''}")
    print(f"{'='*60}\n")


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    print("\n🔧 RAG System Configuration")
    print(f"\nActive Article: {ACTIVE_ARTICLE}")
    print(f"\nAvailable Articles:")
    for key in list_available_articles():
        marker = "→ " if key == ACTIVE_ARTICLE else "  "
        desc = ARTICLE_CONFIGS[key]["description"][:50]
        print(f"  {marker}{key}: {desc}...")
    
    print_article_summary()
