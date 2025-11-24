"""
Metadata Extraction Comparison Demo

This script demonstrates the different metadata extraction options side-by-side.
Run this to see what metadata each option produces.

Note: This is for demonstration only. Use langextract_simple.py for actual processing.
"""

import os
from pathlib import Path

# Mock metadata examples showing what each option produces

print("=" * 100)
print("METADATA EXTRACTION OPTIONS COMPARISON")
print("=" * 100)
print()

# =============================================================================
# OPTION 1: None (Basic)
# =============================================================================
print("┌" + "─" * 98 + "┐")
print("│ OPTION 1: None (Basic) - Fastest, FREE                                                          │")
print("└" + "─" * 98 + "┘")
print()
print("Configuration:")
print("  metadata = None")
print()
print("Processing:")
print("  ✓ Document loaded")
print("  ✓ Split into chunks with SentenceSplitter")
print("  ✓ Basic metadata added (page numbers, file info)")
print()
print("Sample Metadata:")
metadata_none = {
    'source': '1',
    'file_path': '/Users/tony3/Documents/llamaindex/data/paul_graham/paul_graham_essay.pdf',
    'file_name': 'paul_graham_essay.pdf',
    'file_type': 'application/pdf',
    'file_size': 75042,
    'page_label': '1'
}
for key, value in metadata_none.items():
    print(f"  {key}: {value}")
print()
print("Metrics:")
print("  ⏱️  Speed: ⚡⚡⚡ Very Fast (~10 seconds for 30 pages)")
print("  💰 Cost: FREE")
print("  📊 Metadata Fields: 6 basic fields")
print()
print()

# =============================================================================
# OPTION 2: EntityExtractor
# =============================================================================
print("┌" + "─" * 98 + "┐")
print("│ OPTION 2: EntityExtractor - Fast, FREE                                                          │")
print("└" + "─" * 98 + "┘")
print()
print("Configuration:")
print("  metadata = 'entity'")
print()
print("Processing:")
print("  ✓ Document loaded")
print("  ✓ Split into chunks with SentenceSplitter")
print("  ✓ EntityExtractor applied (local HuggingFace model)")
print("  ✓ Named entities extracted: PERSON, ORGANIZATION, LOCATION, etc.")
print()
print("Sample Metadata:")
metadata_entity = {
    'source': '1',
    'file_path': '/Users/tony3/Documents/llamaindex/data/paul_graham/paul_graham_essay.pdf',
    'file_name': 'paul_graham_essay.pdf',
    'file_type': 'application/pdf',
    'file_size': 75042,
    'page_label': '1',
    'PER': ['Paul Graham', 'Robert Morris', 'Trevor Blackwell'],
    'ORG': ['Y Combinator', 'Viaweb', 'Yahoo', 'MIT'],
    'LOC': ['Silicon Valley', 'Cambridge', 'San Francisco']
}
for key, value in metadata_entity.items():
    if isinstance(value, list):
        print(f"  {key}: {', '.join(value)}")
    else:
        print(f"  {key}: {value}")
print()
print("Metrics:")
print("  ⏱️  Speed: ⚡⚡ Fast (~30 seconds for 30 pages)")
print("  💰 Cost: FREE (local model)")
print("  📊 Metadata Fields: 6 basic + entity fields (variable)")
print("  🔧 Model: span-marker-bert-base-multilingual-cased-multinerd")
print()
print()

# =============================================================================
# OPTION 3: LangExtract
# =============================================================================
print("┌" + "─" * 98 + "┐")
print("│ OPTION 3: LangExtract - Slow, PAID, Rich Metadata                                               │")
print("└" + "─" * 98 + "┘")
print()
print("Configuration:")
print("  metadata = 'langextract'")
print("  schema_name = 'paul_graham_detailed'")
print()
print("Processing:")
print("  ✓ Document loaded")
print("  ✓ Split into chunks with SentenceSplitter")
print("  ✓ LangExtract applied (OpenAI GPT-4 API calls)")
print("  ✓ Rich semantic metadata extracted:")
print("    • Concepts (with categories and importance)")
print("    • Advice (with types and domains)")
print("    • Experiences (with periods and sentiments)")
print("    • Entities (with roles)")
print("    • Time references (with decades)")
print()
print("Sample Metadata:")
metadata_langextract = {
    'source': '1',
    'file_path': '/Users/tony3/Documents/llamaindex/data/paul_graham/paul_graham_essay.pdf',
    'file_name': 'paul_graham_essay.pdf',
    'file_type': 'application/pdf',
    'file_size': 75042,
    'page_label': '1',
    'langextract_concepts': ['startup ecosystem', 'programming languages', 'venture capital', 'essay writing'],
    'concept_categories': ['business', 'technology', 'writing'],
    'concept_importance': ['high', 'medium'],
    'langextract_advice': ['focus on building something people want', 'write about what you care about'],
    'advice_types': ['strategic', 'practical'],
    'advice_domains': ['business', 'writing'],
    'langextract_entities': ['Y Combinator', 'Lisp', 'Arc'],
    'entity_roles': ['organization', 'technology', 'product'],
    'entity_names': ['Y Combinator', 'Lisp', 'Arc'],
    'langextract_experiences': ['founding a startup', 'learning programming', 'writing essays'],
    'experience_periods': ['early_career', 'success', 'reflection'],
    'experience_sentiments': ['challenging', 'rewarding'],
    'time_references': ['1995', '2000s', '2010s'],
    'time_decades': ['1990s', '2000s', '2010s']
}
for key, value in metadata_langextract.items():
    if isinstance(value, list):
        print(f"  {key}: {', '.join(value)}")
    else:
        print(f"  {key}: {value}")
print()
print("Metrics:")
print("  ⏱️  Speed: ⚡ Slow (~15 minutes for 30 pages)")
print("  💰 Cost: ~$2 for 30 pages with GPT-4o")
print("  📊 Metadata Fields: 6 basic + ~15 semantic fields")
print("  🔧 Model: OpenAI GPT-4o via LangExtract")
print("  🔑 Requires: OPENAI_API_KEY environment variable")
print()
print()

# =============================================================================
# OPTION 4: Both (EntityExtractor + LangExtract)
# =============================================================================
print("┌" + "─" * 98 + "┐")
print("│ OPTION 4: Both - Slowest, PAID, Maximum Metadata                                                │")
print("└" + "─" * 98 + "┘")
print()
print("Configuration:")
print("  metadata = 'both'")
print("  schema_name = 'paul_graham_detailed'")
print()
print("Processing:")
print("  ✓ Document loaded")
print("  ✓ Split into chunks with SentenceSplitter")
print("  ✓ EntityExtractor applied (local model)")
print("  ✓ LangExtract applied (OpenAI API)")
print("  ✓ Both metadata sets combined")
print()
print("Sample Metadata (Combined):")
metadata_both = {
    'source': '1',
    'file_path': '/Users/tony3/Documents/llamaindex/data/paul_graham/paul_graham_essay.pdf',
    'file_name': 'paul_graham_essay.pdf',
    'file_type': 'application/pdf',
    'file_size': 75042,
    'page_label': '1',
    # EntityExtractor metadata
    'PER': ['Paul Graham', 'Robert Morris', 'Trevor Blackwell'],
    'ORG': ['Y Combinator', 'Viaweb', 'Yahoo', 'MIT'],
    'LOC': ['Silicon Valley', 'Cambridge', 'San Francisco'],
    # LangExtract metadata
    'langextract_concepts': ['startup ecosystem', 'programming languages', 'venture capital'],
    'concept_categories': ['business', 'technology'],
    'concept_importance': ['high', 'medium'],
    'langextract_advice': ['focus on building something people want'],
    'advice_types': ['strategic'],
    'advice_domains': ['business'],
    'langextract_entities': ['Y Combinator', 'Lisp'],
    'entity_roles': ['organization', 'technology'],
    'langextract_experiences': ['founding a startup'],
    'experience_periods': ['early_career', 'success'],
    'time_references': ['1995', '2000s'],
    'time_decades': ['1990s', '2000s']
}
for key, value in metadata_both.items():
    if isinstance(value, list):
        print(f"  {key}: {', '.join(value)}")
    else:
        print(f"  {key}: {value}")
print()
print("Metrics:")
print("  ⏱️  Speed: ⚡ Slowest (~16 minutes for 30 pages)")
print("  💰 Cost: ~$2 for 30 pages (GPT-4o)")
print("  📊 Metadata Fields: 6 basic + entity fields + semantic fields (~20+ total)")
print("  🔧 Models: EntityExtractor (local) + OpenAI GPT-4o (API)")
print("  🔑 Requires: OPENAI_API_KEY environment variable")
print()
print()

# =============================================================================
# COMPARISON TABLE
# =============================================================================
print("=" * 100)
print("COMPARISON TABLE")
print("=" * 100)
print()
print("┌────────────────┬──────────────┬──────────────┬──────────────────┬─────────────────────────────┐")
print("│ Option         │ Speed        │ Cost         │ Metadata Fields  │ Best Use Case               │")
print("├────────────────┼──────────────┼──────────────┼──────────────────┼─────────────────────────────┤")
print("│ None           │ ⚡⚡⚡ Fast  │ FREE         │ 6 basic          │ Quick testing               │")
print("│ EntityExtract  │ ⚡⚡ Fast    │ FREE         │ ~10 total        │ Entity recognition          │")
print("│ LangExtract    │ ⚡ Slow      │ ~$2/30pg     │ ~20 total        │ Semantic understanding      │")
print("│ Both           │ ⚡ Slowest   │ ~$2/30pg     │ ~25+ total       │ Maximum metadata richness   │")
print("└────────────────┴──────────────┴──────────────┴──────────────────┴─────────────────────────────┘")
print()

# =============================================================================
# QUERY EXAMPLES
# =============================================================================
print("=" * 100)
print("QUERY EXAMPLES BY METADATA TYPE")
print("=" * 100)
print()

print("Queries that work with ALL options:")
print("  • What is on page 5?")
print("  • Summarize the entire document")
print("  • What are the contents from pages 1 to 3?")
print()

print("Queries enhanced by EntityExtractor:")
print("  • Who are the main people mentioned?")
print("  • What companies did the author work with?")
print("  • Where did these events take place?")
print("  • List all organizations mentioned on pages 10-15")
print()

print("Queries requiring LangExtract:")
print("  • What strategic advice is given about startups?")
print("  • What programming concepts are discussed?")
print("  • What experiences from the 1990s are described?")
print("  • How did early career experiences influence later decisions?")
print("  • What advice relates to product development?")
print("  • What challenges did the author face during the success period?")
print()

print("Advanced queries leveraging Both:")
print("  • How do the people (entities) relate to the startup advice (semantic)?")
print("  • What did Paul Graham (entity) advise about programming concepts (semantic)?")
print("  • Connect the organizations (entities) to the business experiences (semantic)")
print()

# =============================================================================
# RECOMMENDATIONS
# =============================================================================
print("=" * 100)
print("RECOMMENDATIONS")
print("=" * 100)
print()

print("Development Workflow:")
print("  1. Start with None or EntityExtractor for quick iteration")
print("  2. Test with a small document (2-3 pages) using LangExtract")
print("  3. Scale to full document once satisfied with results")
print("  4. Choose production option based on requirements and budget")
print()

print("Choose EntityExtractor if:")
print("  ✓ You need entity recognition only")
print("  ✓ Budget is limited (FREE)")
print("  ✓ Processing speed matters")
print("  ✓ Simple entity-based queries are sufficient")
print()

print("Choose LangExtract if:")
print("  ✓ You need rich semantic understanding")
print("  ✓ Complex conceptual queries are important")
print("  ✓ Budget allows for API costs")
print("  ✓ Metadata quality > processing speed")
print()

print("Choose Both if:")
print("  ✓ You need maximum metadata richness")
print("  ✓ Both entity and semantic queries are required")
print("  ✓ Budget is not a constraint")
print("  ✓ Highest quality results are priority")
print()

print("=" * 100)
print("For more details, see:")
print("  • METADATA_EXTRACTION_GUIDE.md - Comprehensive guide")
print("  • EXAMPLES_METADATA.py - Configuration examples")
print("  • langextract_simple.py - Main implementation")
print("=" * 100)
