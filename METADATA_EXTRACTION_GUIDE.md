# Metadata Extraction Guide

This guide explains how to use the different metadata extraction options in the LlamaIndex RAG implementation.

## Overview

The system supports four metadata extraction methods:

1. **None (Basic)** - Fast, free, minimal metadata
2. **EntityExtractor** - Fast, free, entity recognition
3. **LangExtract** - Slow, paid, rich semantic metadata
4. **Both** - Combination of EntityExtractor + LangExtract

## Configuration

Edit the following variables in `langextract_simple.py`:

```python
# Choose your metadata extraction method
metadata = "entity"  # Options: None, "entity", "langextract", "both"

# LangExtract schema (only for "langextract" or "both")
schema_name = "paul_graham_detailed"  # or "paul_graham_simple"
```

## Option 1: None (Basic)

**When to use:** Quick testing, simple documents, no need for metadata

**Configuration:**
```python
metadata = None
```

**Output:**
- Basic chunking with SentenceSplitter
- Only page numbers and file metadata
- Fastest processing

**Cost:** FREE  
**Speed:** ⚡⚡⚡ Very Fast

---

## Option 2: EntityExtractor

**When to use:** Need entity recognition, want free solution

**Configuration:**
```python
metadata = "entity"
```

**Output:**
- Named entities: PERSON, ORGANIZATION, LOCATION, etc.
- Uses local HuggingFace model
- Metadata fields: `entities`, `entity_labels`

**Example metadata:**
```python
{
    'source': '1',
    'file_path': '/path/to/document.pdf',
    'file_name': 'document.pdf',
    'file_type': 'application/pdf',
    'file_size': 123456,
    'page_label': '1',
    'PER': ['John Doe', 'Jane Smith'],
    'ORG': ['OpenAI', 'Google'],
    'LOC': ['San Francisco', 'New York']
}
```

**Cost:** FREE (local model)  
**Speed:** ⚡⚡ Fast  
**Requirements:** None (model downloads automatically)

---

## Option 3: LangExtract

**When to use:** Need deep semantic understanding, complex queries, rich metadata

**Configuration:**
```python
metadata = "langextract"
schema_name = "paul_graham_detailed"  # or "paul_graham_simple"
```

**Output:**
Rich structured metadata including:
- **Concepts**: programming, philosophy, business, creativity
- **Advice**: strategic, tactical, practical recommendations
- **Experiences**: early career, success stories, challenges
- **Entities**: people, organizations, products with roles
- **Time references**: years, decades, periods

**Example metadata:**
```python
{
    'source': '1',
    'langextract_concepts': ['artificial intelligence', 'startup culture'],
    'concept_categories': ['technology', 'business'],
    'concept_importance': ['high', 'medium'],
    'langextract_advice': ['focus on product-market fit', 'hire slowly'],
    'advice_types': ['strategic', 'tactical'],
    'advice_domains': ['business', 'management'],
    'langextract_entities': ['Y Combinator', 'Paul Graham'],
    'entity_roles': ['organization', 'founder'],
    'langextract_experiences': ['launching a startup', 'learning to code'],
    'experience_periods': ['early_career', 'success'],
    'time_references': ['1995', '2000s'],
    'time_decades': ['1990s', '2000s']
}
```

**Cost:** PAID (OpenAI API usage, ~$0.01-0.05 per page with GPT-4)  
**Speed:** ⚡ Slow (API calls, ~5-10 seconds per chunk)  
**Requirements:** 
- `OPENAI_API_KEY` environment variable
- OpenAI account with credits

---

## Option 4: Both (EntityExtractor + LangExtract)

**When to use:** Maximum metadata richness, comprehensive analysis

**Configuration:**
```python
metadata = "both"
schema_name = "paul_graham_detailed"
```

**Output:**
Combines all metadata from both extractors:
- EntityExtractor entities (PER, ORG, LOC)
- LangExtract semantic metadata (concepts, advice, etc.)

**Example metadata:**
```python
{
    'source': '1',
    # EntityExtractor metadata
    'PER': ['Paul Graham', 'Jessica Livingston'],
    'ORG': ['Y Combinator', 'Viaweb'],
    'LOC': ['Silicon Valley', 'Cambridge'],
    # LangExtract metadata
    'langextract_concepts': ['startup ecosystem', 'venture capital'],
    'concept_categories': ['business', 'technology'],
    'langextract_advice': ['build something people want'],
    'advice_types': ['strategic'],
    'langextract_entities': ['Y Combinator', 'Paul Graham'],
    'entity_roles': ['organization', 'founder'],
    # ... and more
}
```

**Cost:** PAID (OpenAI API usage)  
**Speed:** ⚡ Slowest (both extractors run sequentially)  
**Requirements:** 
- `OPENAI_API_KEY` environment variable
- Sufficient disk space for EntityExtractor model

---

## Available LangExtract Schemas

### 1. paul_graham_detailed

Rich, comprehensive schema with many extraction classes:
- Concepts (with categories and importance levels)
- Advice (with types and domains)
- Experiences (with periods and sentiments)
- Entities (with roles)
- Time references (with decades)

**Best for:** Deep analysis, research, complex queries

### 2. paul_graham_simple

Simplified schema with basic extractions:
- Key concepts
- Main advice
- Important entities

**Best for:** Basic semantic understanding, cost-conscious projects

---

## Setup Instructions

### 1. Install Dependencies

```bash
pip install llama-index langextract openai python-dotenv
```

### 2. Set Environment Variables

Create a `.env` file:

```bash
# Required for all options
ANTHROPIC_API_KEY=your_anthropic_key_here

# Required only for "langextract" or "both" options
OPENAI_API_KEY=your_openai_key_here

# MongoDB and Milvus (required for all options)
# Usually default values work
```

### 3. Run the Script

```bash
python langextract_simple.py
```

---

## Performance Comparison

Processing a 30-page document (chunk_size=256):

| Option | Time | Cost | Chunks | Metadata Richness |
|--------|------|------|--------|-------------------|
| None | ~10s | FREE | ~150 | ⭐ Basic |
| EntityExtractor | ~30s | FREE | ~150 | ⭐⭐ Good |
| LangExtract | ~15min | ~$2 | ~150 | ⭐⭐⭐⭐⭐ Excellent |
| Both | ~16min | ~$2 | ~150 | ⭐⭐⭐⭐⭐ Maximum |

*Times and costs are approximate and depend on document complexity*

---

## Troubleshooting

### EntityExtractor Issues

**Problem:** Model fails to download  
**Solution:** Check internet connection, try manual download:
```python
from transformers import AutoModelForTokenClassification
model = AutoModelForTokenClassification.from_pretrained(
    "lxyuan/span-marker-bert-base-multilingual-cased-multinerd"
)
```

**Problem:** Device error (MPS not available)  
**Solution:** Change device to CPU in code:
```python
entity_extractor = EntityExtractor(
    device="cpu",  # Change from "mps"
    # ... other parameters
)
```

### LangExtract Issues

**Problem:** "OPENAI_API_KEY not found"  
**Solution:** Set environment variable:
```bash
export OPENAI_API_KEY=your_key_here
```

**Problem:** API rate limits  
**Solution:** Add delay between chunks or use batching

**Problem:** High costs  
**Solution:** 
- Use "paul_graham_simple" schema
- Process fewer chunks
- Use smaller chunk_size
- Switch to "entity" option

---

## Best Practices

1. **Start with EntityExtractor** for testing and development
2. **Use LangExtract** only when you need rich semantic metadata
3. **Monitor costs** when using LangExtract (set up billing alerts)
4. **Cache results** - the system stores to MongoDB/Milvus for reuse
5. **Choose appropriate schema** - detailed for research, simple for production
6. **Test with small documents first** before processing large corpora

---

## Query Examples by Metadata Type

### Basic Queries (None option)
```python
"What is on page 5?"
"Summarize the entire document"
```

### Entity-Based Queries (EntityExtractor)
```python
"What companies are mentioned in the document?"
"List all people mentioned on pages 10-15"
"Where did the author work?"
```

### Semantic Queries (LangExtract)
```python
"What strategic advice is given about startups?"
"What experiences from the 1990s are described?"
"What programming concepts are discussed?"
"What advice relates to management?"
```

### Complex Queries (Both)
```python
"What did Paul Graham (entity) advise about startup culture (concept)?"
"How do the entities relate to the advice given in the 2000s?"
"What experiences led to the strategic advice about product development?"
```

---

## Integration with LangExtract

The implementation uses the `langextract_integration.py` module which provides:

- `extract_metadata_from_text()` - Extract from single text chunk
- `enrich_nodes_with_langextract()` - Batch enrich all nodes
- `print_sample_metadata()` - Debug metadata output
- `flatten_extraction_result()` - Convert to Milvus-friendly format

See `langextract_integration.py` for implementation details.

---

## Schema Customization

To create your own schema, edit `langextract_schemas.py`:

```python
def get_schema(schema_name: str = "my_custom_schema"):
    schemas = {
        "my_custom_schema": {
            "prompt": "Extract key themes and topics from this text",
            "examples": [
                # Your examples here
            ]
        }
    }
    return schemas.get(schema_name)
```

Then use:
```python
metadata = "langextract"
schema_name = "my_custom_schema"
```

---

## Support

For issues or questions:
1. Check the main documentation
2. Review `langextract_integration.py` implementation
3. Consult LangExtract documentation: https://github.com/google/langextract
4. Check LlamaIndex docs: https://docs.llamaindex.ai/
