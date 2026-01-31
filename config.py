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
from queries import get_query_for_article

# =============================================================================
# ACTIVE ARTICLE SELECTION
# =============================================================================
# Change this to switch which article is being processed
# Must be a key from ARTICLE_CONFIGS

ACTIVE_ARTICLE = "RAG_Anything"
# ACTIVE_ARTICLE = "paul_graham_essay"
# ACTIVE_ARTICLE = "Laser_coprop_RA"
# ACTIVE_ARTICLE = "Noise_in_DRA"
# ACTIVE_ARTICLE = "ASE_noise_pump_depletion"
# ACTIVE_ARTICLE = "NF_Analysis_DFRA"
# ACTIVE_ARTICLE = "Pump_depletion_FRA"
# ACTIVE_ARTICLE = "How_to_do_great_work"
# ACTIVE_ARTICLE = "attention_all"
# ACTIVE_ARTICLE = "metagpt"
# ACTIVE_ARTICLE = "uber_10q_march_2022"
# ACTIVE_ARTICLE = "eBook-How-to-Build-a-Career-in-AI"

# Get the active query for the selected article
QUERY = get_query_for_article(ACTIVE_ARTICLE)


# =============================================================================
# ARTICLE CONFIGURATIONS
# =============================================================================
# Add new articles here. Each article needs:
# - directory: Folder name under ./data/
# - filename: PDF filename
# - schema: LangExtract schema to use (see extraction_schemas.py)
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
    },
    "How_to_do_great_work": {
        "directory": "paul_graham",
        "filename": "How_to_do_great_work.pdf",
        "schema": "paul_graham_detailed",  # Same schema works for PG essays
        "description": "Paul Graham's essay on how to do great work",
    },
    
    # -------------------------------------------------------------------------
    # Academic Papers (uses academic schema)
    # -------------------------------------------------------------------------
    "attention_all": {
        "directory": "attention",
        "filename": "attention_all.pdf",
        "schema": "academic",  # Uses academic paper schema
        "description": "Attention Is All You Need - Transformer architecture paper",
    },
        "RAG_Anything": {
        "directory": "Rag_anything",
        "filename": "RAG_Anything.pdf",
        "schema": "academic",
        "description": "RAG Anything - A multimodal RAG system paper",
    },
        "Laser_coprop_RA": {
        "directory": "DRA",
        "filename": "Laser_coprop_RA.pdf",
        "schema": "academic",
        "description": "Furukawa team proposed a new inner-grating multimode (iGM) laser for DRA",
    },
    "Noise_in_DRA": {
        "directory": "DRA",
        "filename": "Noise_in_DRA.pdf",
        "schema": "academic",
        "description": "Investigation of noise characteristics in distributed Raman amplifiers",
    },
    "ASE_noise_pump_depletion": {
        "directory": "DRA",
        "filename": "ASE_noise_pump_depletion.pdf",
        "schema": "academic",
        "description": "Analysis of ASE noise in distributed Raman amplifiers with pump depletion",
    },
    "Pump_depletion_FRA": {
        "directory": "DRA",
        "filename": "Pump_depletion_FRA.pdf",
        "schema": "academic",
        "description": "Analysis of pump depletion in fiber Raman amplifiers",
    },
    "NF_Analysis_DFRA": {
        "directory": "DRA",
        "filename": "NF_Analysis_DFRA.pdf",
        "schema": "academic",
        "description": "Analysis of noise figure in distributed fiber Raman amplifiers",
    },
    
    # -------------------------------------------------------------------------
    # Technical Documents (uses technical schema)
    # -------------------------------------------------------------------------
    "metagpt": {
        "directory": "metagpt",
        "filename": "metagpt.pdf",
        "schema": "technical",  # Uses technical documentation schema
        "description": "MetaGPT technical documentation",
    },
    
    # -------------------------------------------------------------------------
    # Financial Documents (uses financial schema)
    # -------------------------------------------------------------------------
    "uber_10q_march_2022": {
        "directory": "uber",
        "filename": "uber_10q_march_2022.pdf",
        "schema": "financial",  # Uses financial document schema
        "description": "Uber 10-Q financial report for Q1 2022",
    },
    
    # -------------------------------------------------------------------------
    # Career/Education Documents (uses career schema)
    # -------------------------------------------------------------------------
    "eBook-How-to-Build-a-Career-in-AI": {
        "directory": "andrew",
        "filename": "eBook-How-to-Build-a-Career-in-AI.pdf",
        "schema": "career",  # Uses career advice schema
        "description": "Andrew Ng's guide on building a career in AI",
    },
}


# =============================================================================
# RAG PIPELINE SETTINGS
# =============================================================================
# These settings can be overridden per-article if needed

DEFAULT_RAG_SETTINGS: Dict[str, Any] = {
    # Chunking settings
    "chunk_size": 512,
    "chunk_overlap": 128,
    "chunk_method": "sentence_splitter",
    
    # Metadata extraction: None, "entity", "langextract", "both"
    "metadata": "None",
    
    # Entity filtering
    "use_entity_filtering": False,
    
    # Fusion retrieval settings
    "similarity_top_k_fusion": 35,
    "num_queries": 1,  # number of generated queries for fusion (1 = original query only)
    "fusion_top_n": 25,
    "rerank_top_n": 15,
    "num_nodes": 0,  # For PrevNextNodePostprocessor
    
    # Debug settings
    "page_filter_verbose": True,
}

# Per-article RAG settings overrides (optional)
ARTICLE_RAG_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "paul_graham_essay": {
        "chunk_size": 256,
        "chunk_overlap": 64,
        "metadata": "both",  # Metadata extraction: "entity", "langextract", "both"
        "use_entity_filtering": True,
        "similarity_top_k_fusion": 48,
        "fusion_top_n": 35, # 38
        "rerank_top_n": 25, # 28
        "num_nodes": 2,  # For PrevNextNodePostprocessor
    },
    # Example: Use different settings for academic papers
    "attention_all": {
        "metadata": "entity",  # Use entity extraction only (no PG-specific schema)
        "use_entity_filtering": True,
    },
    "uber_10q_march_2022": {
        "metadata": "entity",
        "use_entity_filtering": True,
    },
    "metagpt": {
        "metadata": "entity",
        "use_entity_filtering": True,
    },
    "eBook-How-to-Build-a-Career-in-AI": {
        "metadata": "entity",
        "use_entity_filtering": True,
    },
    "RAG_Anything": {
        "metadata": "both",
        "use_entity_filtering": True,
        "chunk_size": 256,
        "chunk_overlap": 64,
        "num_nodes": 1,  # For PrevNextNodePostprocessor
    },
    "Laser_coprop_RA": {
        "metadata": "both",
        "use_entity_filtering": True,
        "chunk_size": 256,
        "chunk_overlap": 64,
        "num_nodes": 1,  # For PrevNextNodePostprocessor
    },
    "Noise_in_DRA": {
        "metadata": "both",
        "use_entity_filtering": True,
        "chunk_size": 256,
        "chunk_overlap": 64,
        "num_nodes": 1,  # For PrevNextNodePostprocessor
    },
    "ASE_noise_pump_depletion": {
        "metadata": "both",
        "use_entity_filtering": True,
        "chunk_size": 256,
        "chunk_overlap": 64,
        "num_nodes": 1,  # For PrevNextNodePostprocessor
    },
    "Pump_depletion_FRA": {
        "metadata": "both",
        "use_entity_filtering": True,
        "chunk_size": 256,
        "chunk_overlap": 64,
        "num_nodes": 1,  # For PrevNextNodePostprocessor
    },
    "NF_Analysis_DFRA": {
        "metadata": "both",
        "use_entity_filtering": True,
        "chunk_size": 256,
        "chunk_overlap": 64,
        "num_nodes": 1,  # For PrevNextNodePostprocessor
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

import os

# Get database connection settings from environment variables with local defaults
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

DATABASE_CONFIG: Dict[str, str] = {
    "milvus_uri": f"http://{MILVUS_HOST}:{MILVUS_PORT}",
    "mongo_uri": MONGO_URI,
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_active_config() -> Dict[str, Any]:
    """
    Get the name, directory, LangExtract schema, and other details for the currently ACTIVE article.
    
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
    Get RAG pipeline settings such as chunk size, chunk overlap, metadata extraction method, 
    fusion retrieval and reranking top N, Prev/Next Postprocessor node number, etc. with any 
    article-specific overrides applied.
    
    Parameters:
        article_key: Optional article key (a key from ARTICLE_CONFIGS that denotes the article) 
        to get settings for. If None, uses ACTIVE_ARTICLE.
    
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
        article_key: Optional article key (a key from ARTICLE_CONFIGS that denotes the article) 
        to get settings for. If None, uses ACTIVE_ARTICLE.
    
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
    metadata = rag['metadata']
    
    print(f"\n{'='*60}")
    print(f"📄 Article Configuration: {key}")
    print(f"{'='*60}")
    print(f"   Directory: ./data/{config['directory']}/")
    print(f"   Filename:  {config['filename']}")
    print(f"   Schema:    {config['schema']}")
    print(f"   Description: {config['description']}")
    print(f"   Active Query: \"{QUERY}\"")
    print(f"\n📊 RAG Settings:")
    print(f"   Metadata Extraction: {metadata if metadata else 'None (Basic)'}")
    if metadata in ["langextract", "both"]:
        print(f"   LangExtract Schema: {config['schema']}")
    print(f"   Chunk Size: {rag['chunk_size']}")
    print(f"   Chunk Overlap: {rag['chunk_overlap']}")
    if metadata in ["entity", "langextract", "both"]:
        print(f"   Entity Filtering: {'✓ Enabled (dynamic per query)' if rag['use_entity_filtering'] else '✗ Disabled'}")
    print(f"\n   Fusion Tree & Reranker:")
    print(f"   ├─ Similarity Top K: {rag['similarity_top_k_fusion']}")
    print(f"   ├─ Number of Queries: {rag['num_queries']}")
    print(f"   ├─ Fusion Top N: {rag['fusion_top_n']}")
    print(f"   ├─ Rerank Top N: {rag['rerank_top_n']}")
    print(f"   └─ Prev/Next Nodes: {rag['num_nodes']}")
    print(f"\n{'='*60}\n")


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
