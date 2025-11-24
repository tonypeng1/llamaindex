"""
Quick Start Examples - LangExtract Integration

This file demonstrates how to quickly switch between different metadata extraction options.
Copy the relevant section to langextract_simple.py for testing.
"""

# =============================================================================
# EXAMPLE 1: No Metadata Extraction (Fastest, FREE)
# =============================================================================
# Use this for: Quick testing, simple documents

metadata = None
schema_name = "paul_graham_detailed"  # Not used

# Expected output:
# - Basic chunks with page numbers only
# - Processing time: ~10 seconds for 30 pages
# - Cost: FREE


# =============================================================================
# EXAMPLE 2: EntityExtractor Only (Fast, FREE)
# =============================================================================
# Use this for: Named entity recognition, standard use case

metadata = "entity"
schema_name = "paul_graham_detailed"  # Not used

# Expected output:
# - Named entities: PERSON, ORGANIZATION, LOCATION, etc.
# - Processing time: ~30 seconds for 30 pages
# - Cost: FREE
# - Metadata example:
#   {
#       'PER': ['Paul Graham', 'Jessica Livingston'],
#       'ORG': ['Y Combinator', 'Viaweb'],
#       'LOC': ['Silicon Valley']
#   }


# =============================================================================
# EXAMPLE 3: LangExtract Only (Slow, PAID)
# =============================================================================
# Use this for: Rich semantic metadata, complex queries

metadata = "langextract"
schema_name = "paul_graham_detailed"  # Options: "paul_graham_detailed" or "paul_graham_simple"

# Expected output:
# - Concepts, advice, experiences, entities, time references
# - Processing time: ~15 minutes for 30 pages
# - Cost: ~$2 for 30 pages (GPT-4)
# - Metadata example:
#   {
#       'langextract_concepts': ['startup ecosystem', 'programming'],
#       'concept_categories': ['business', 'technology'],
#       'langextract_advice': ['focus on product-market fit'],
#       'advice_types': ['strategic'],
#       'langextract_entities': ['Y Combinator'],
#       'entity_roles': ['organization']
#   }


# =============================================================================
# EXAMPLE 4: Both EntityExtractor + LangExtract (Slowest, PAID)
# =============================================================================
# Use this for: Maximum metadata richness

metadata = "both"
schema_name = "paul_graham_detailed"

# Expected output:
# - All EntityExtractor metadata + All LangExtract metadata
# - Processing time: ~16 minutes for 30 pages
# - Cost: ~$2 for 30 pages (GPT-4)
# - Metadata example (combined):
#   {
#       # EntityExtractor
#       'PER': ['Paul Graham'],
#       'ORG': ['Y Combinator'],
#       # LangExtract
#       'langextract_concepts': ['startup ecosystem'],
#       'langextract_advice': ['focus on users'],
#       'langextract_entities': ['Y Combinator'],
#       'entity_roles': ['organization']
#   }


# =============================================================================
# EXAMPLE 5: LangExtract with Simple Schema (Faster, Cheaper)
# =============================================================================
# Use this for: Basic semantic understanding with lower cost

metadata = "langextract"
schema_name = "paul_graham_simple"  # Simpler schema = less metadata = lower cost

# Expected output:
# - Basic concepts, advice, entities
# - Processing time: ~10 minutes for 30 pages (faster than detailed)
# - Cost: ~$1 for 30 pages (cheaper than detailed)


# =============================================================================
# Testing Workflow
# =============================================================================

# Step 1: Start with None or EntityExtractor for quick testing
metadata = None  # or "entity"

# Step 2: Once your pipeline works, test LangExtract on a small sample
# Use a short document (2-3 pages) first!
metadata = "langextract"
schema_name = "paul_graham_simple"

# Step 3: If results are good, scale up to full document
metadata = "langextract"
schema_name = "paul_graham_detailed"

# Step 4: For production, choose based on requirements and budget
# - Budget-conscious: "entity"
# - Rich metadata needed: "langextract"
# - Maximum quality: "both"


# =============================================================================
# Environment Setup
# =============================================================================

# Required for ALL options:
# export ANTHROPIC_API_KEY=your_anthropic_key

# Required ONLY for "langextract" or "both":
# export OPENAI_API_KEY=your_openai_key


# =============================================================================
# Query Examples by Metadata Type
# =============================================================================

# For None or EntityExtractor:
query_examples_basic = [
    "What is on page 5?",
    "List all people mentioned",
    "What companies are in the document?",
]

# For LangExtract or Both:
query_examples_semantic = [
    "What strategic advice is given about startups?",
    "What experiences from the 1990s are described?",
    "What programming concepts are discussed?",
    "What advice relates to management?",
    "How did early career experiences shape later decisions?",
]


# =============================================================================
# Monitoring Costs (for LangExtract)
# =============================================================================

# Approximate costs with GPT-4o:
# - Input: ~$0.0025 per 1K tokens
# - Output: ~$0.01 per 1K tokens
# 
# For chunk_size=256:
# - ~300 tokens per chunk
# - ~100 tokens output per chunk
# - ~$0.0015 per chunk
# - ~$0.225 per 150 chunks (30 pages)
# 
# Note: Actual costs vary based on:
# - Schema complexity (detailed vs simple)
# - Text density
# - Number of extractions

# Set up OpenAI usage alerts:
# 1. Go to https://platform.openai.com/account/billing/limits
# 2. Set monthly budget limit
# 3. Enable email notifications


# =============================================================================
# Debugging Tips
# =============================================================================

# 1. Check metadata output:
# After running, the script will print sample metadata for the first 3 nodes

# 2. Verify extractions:
# Look for these keys in metadata:
# - EntityExtractor: 'PER', 'ORG', 'LOC', etc.
# - LangExtract: 'langextract_concepts', 'langextract_advice', etc.

# 3. Test with small documents first:
# Process 2-3 pages to verify before scaling to full document

# 4. Check API keys:
# The script will warn if OPENAI_API_KEY is missing when using LangExtract

# 5. Monitor progress:
# Watch for "Processing X/Y nodes..." messages during extraction


# =============================================================================
# Performance Optimization
# =============================================================================

# 1. Adjust chunk size:
chunk_size = 256      # Standard
chunk_size = 512      # Larger chunks = fewer API calls = lower cost (but less granular)
chunk_size = 128      # Smaller chunks = more API calls = higher cost (but more granular)

# 2. Choose schema wisely:
schema_name = "paul_graham_simple"    # Faster, cheaper
schema_name = "paul_graham_detailed"  # Slower, more expensive, richer metadata

# 3. Use caching:
# The system caches to MongoDB/Milvus - subsequent runs use cached data
# To force re-extraction, delete the database collection

# 4. Batch processing:
# Process documents in batches during off-peak hours to avoid rate limits


# =============================================================================
# Common Issues and Solutions
# =============================================================================

# Issue: "OPENAI_API_KEY not found"
# Solution: Set environment variable before running:
#   export OPENAI_API_KEY=your_key_here

# Issue: EntityExtractor device error (MPS not available)
# Solution: Change device to "cpu" in get_nodes_from_document_sentence_splitter_entity_extractor()

# Issue: Rate limit errors from OpenAI
# Solution: Add delays, reduce batch size, or upgrade API tier

# Issue: High costs
# Solution: Use "paul_graham_simple" schema or switch to "entity" option

# Issue: Missing metadata in query results
# Solution: Verify metadata was extracted (check print_sample_metadata output)
