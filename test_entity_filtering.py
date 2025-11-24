"""
Test script for entity-based filtering feature.

This script demonstrates how entity filtering improves retrieval precision
for entity-focused queries.
"""

from utils import extract_entities_from_query, create_entity_metadata_filters

def test_entity_extraction():
    """Test entity extraction from various queries."""
    
    print("=" * 80)
    print("ENTITY EXTRACTION TESTS")
    print("=" * 80)
    
    test_queries = [
        "What did Paul Graham advise about Y Combinator?",
        "Where did Jessica Livingston work before starting YC?",
        "What happened at Viaweb and Yahoo?",
        "Describe the author's experiences in Silicon Valley and Cambridge",
        "What advice is given about startups?",  # No entities
        "Tell me about MIT and Harvard",
        "What did the essay say about Florence, Italy?",
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        entities = extract_entities_from_query(query)
        
        if any(entities.values()):
            print("✓ Entities detected:")
            for entity_type, entity_list in entities.items():
                if entity_list:
                    print(f"  - {entity_type}: {entity_list}")
        else:
            print("✗ No entities detected (will use standard retrieval)")
        print("-" * 80)

def test_filter_creation():
    """Test metadata filter creation for different metadata options."""
    
    print("\n" + "=" * 80)
    print("METADATA FILTER CREATION TESTS")
    print("=" * 80)
    
    query = "What did Paul Graham advise about Y Combinator?"
    entities = extract_entities_from_query(query)
    
    print(f"\n📝 Query: {query}")
    print(f"✓ Extracted entities: {entities}")
    
    # Test EntityExtractor format
    print("\n🔧 Testing EntityExtractor format (metadata='entity'):")
    filters_entity = create_entity_metadata_filters(entities, metadata_option="entity")
    if filters_entity:
        print(f"   Created MetadataFilters with {len(filters_entity.filters)} conditions (OR)")
        for i, f in enumerate(filters_entity.filters, 1):
            print(f"   {i}. key='{f.key}', value='{f.value}', operator='{f.operator}'")
    
    # Test LangExtract format
    print("\n🔧 Testing LangExtract format (metadata='langextract'):")
    filters_langextract = create_entity_metadata_filters(entities, metadata_option="langextract")
    if filters_langextract:
        print(f"   Created MetadataFilters with {len(filters_langextract.filters)} conditions (OR)")
        for i, f in enumerate(filters_langextract.filters, 1):
            print(f"   {i}. key='{f.key}', value='{f.value}', operator='{f.operator}'")
    
    # Test Both format
    print("\n🔧 Testing Both format (metadata='both'):")
    filters_both = create_entity_metadata_filters(entities, metadata_option="both")
    if filters_both:
        print(f"   Created MetadataFilters with {len(filters_both.filters)} conditions (OR)")
        for i, f in enumerate(filters_both.filters, 1):
            print(f"   {i}. key='{f.key}', value='{f.value}', operator='{f.operator}'")

def test_query_comparison():
    """Compare entity-focused vs non-entity queries."""
    
    print("\n" + "=" * 80)
    print("ENTITY-FOCUSED vs NON-ENTITY QUERIES")
    print("=" * 80)
    
    comparisons = [
        ("ENTITY-FOCUSED", "What did Paul Graham advise about Y Combinator?"),
        ("NON-ENTITY", "What advice is given about startups?"),
        ("ENTITY-FOCUSED", "Where did Jessica Livingston work before YC?"),
        ("NON-ENTITY", "What are good practices for entrepreneurs?"),
        ("ENTITY-FOCUSED", "Describe experiences in Silicon Valley"),
        ("NON-ENTITY", "Summarize the document"),
    ]
    
    for query_type, query in comparisons:
        print(f"\n{query_type} Query:")
        print(f"  📝 \"{query}\"")
        
        entities = extract_entities_from_query(query)
        if any(entities.values()):
            print(f"  ✓ Entity filtering will be applied")
            for entity_type, entity_list in entities.items():
                if entity_list:
                    print(f"     - {entity_type}: {entity_list}")
        else:
            print(f"  ✗ No entities - standard retrieval")
        print("-" * 80)

def main():
    """Run all tests."""
    print("\n🧪 ENTITY-BASED FILTERING TEST SUITE\n")
    
    test_entity_extraction()
    test_filter_creation()
    test_query_comparison()
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS COMPLETE")
    print("=" * 80)
    print("\n💡 Next Steps:")
    print("   1. Run langextract_simple.py with entity-focused queries")
    print("   2. Compare results with use_entity_filtering=True vs False")
    print("   3. Add more entities to known lists in utils.py as needed")
    print("\n📖 See ENTITY_FILTERING_GUIDE.md for detailed documentation\n")

if __name__ == "__main__":
    main()
