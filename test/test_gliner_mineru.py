"""
Test GLiNER integration with MinerU pipeline.

This test suite validates:
1. Entity label selection for different schema types
2. GLiNER extraction on academic content
3. GLiNER extraction on technical content
4. GLiNER extraction on general content
"""

import sys
sys.path.insert(0, '/Users/tony3/Documents/llamaindex')

from gliner_extractor import GLiNERExtractor, print_extraction_summary
from langextract_schemas import get_gliner_entity_labels
from llama_index.core.schema import Document
from llama_index.core.node_parser import SentenceSplitter


def test_entity_label_selection():
    """Test domain-specific entity label selection."""
    print("\n" + "="*80)
    print("TEST 1: Entity Label Selection by Schema Type")
    print("="*80)
    
    test_schemas = [
        ("academic", "Academic Papers"),
        ("technical", "Technical Documentation"),
        ("financial", "Financial Documents"),
        ("paul_graham_detailed", "Paul Graham Essays"),
        ("career", "Career Advice"),
        ("general", "General Documents")
    ]
    
    for schema_name, description in test_schemas:
        labels = get_gliner_entity_labels(schema_name=schema_name)
        print(f"\n📚 {description} ({schema_name}):")
        print(f"   Entity types: {len(labels)}")
        print(f"   Examples: {labels[:5]}...")
    
    print(f"\n{'='*80}\n")


def test_gliner_extraction_academic():
    """Test GLiNER on academic content."""
    print("="*80)
    print("TEST 2: Academic Entity Extraction")
    print("="*80)
    
    # Get academic entity labels
    labels = get_gliner_entity_labels(schema_name="academic")
    print(f"\n🎯 Using {len(labels)} academic entity types")
    
    # Initialize extractor
    extractor = GLiNERExtractor(
        model_name="urchade/gliner_medium-v2.1",
        entity_labels=labels,
        threshold=0.5,
        device="mps"
    )
    
    # Test text with academic entities
    text = """
    The Transformer model was introduced by Vaswani et al. at Google Brain.
    We evaluated it on the WMT 2014 dataset and achieved a BLEU score of 28.4.
    The experiments were conducted at Stanford University using 8 NVIDIA P100 GPUs.
    Our method outperformed the baseline by 15% on the benchmark.
    """
    
    # Process through pipeline
    doc = Document(text=text)
    parser = SentenceSplitter(chunk_size=512)
    nodes = parser.get_nodes_from_documents([doc])
    nodes_with_entities = extractor(nodes)
    
    # Display results
    print_extraction_summary(nodes_with_entities, max_nodes=5)


def test_gliner_extraction_technical():
    """Test GLiNER on technical content."""
    print("="*80)
    print("TEST 3: Technical Entity Extraction")
    print("="*80)
    
    # Get technical entity labels
    labels = get_gliner_entity_labels(schema_name="technical")
    print(f"\n🎯 Using {len(labels)} technical entity types")
    
    # Initialize extractor
    extractor = GLiNERExtractor(
        model_name="urchade/gliner_medium-v2.1",
        entity_labels=labels,
        threshold=0.5,
        device="mps"
    )
    
    # Test text with technical entities
    text = """
    The VectorStoreIndex.from_documents() method accepts a documents parameter.
    You can configure the chunk_size in the settings.yaml file.
    Use the pip install command to add the llama-index package.
    The API endpoint is available at /api/v1/query.
    """
    
    # Process through pipeline
    doc = Document(text=text)
    parser = SentenceSplitter(chunk_size=512)
    nodes = parser.get_nodes_from_documents([doc])
    nodes_with_entities = extractor(nodes)
    
    # Display results
    print_extraction_summary(nodes_with_entities, max_nodes=5)


def test_gliner_extraction_general():
    """Test GLiNER on general content."""
    print("="*80)
    print("TEST 4: General Entity Extraction")
    print("="*80)
    
    # Get general entity labels
    labels = get_gliner_entity_labels(schema_name="general")
    print(f"\n🎯 Using {len(labels)} general entity types")
    
    # Initialize extractor
    extractor = GLiNERExtractor(
        model_name="urchade/gliner_medium-v2.1",
        entity_labels=labels,
        threshold=0.5,
        device="mps"
    )
    
    # Test text with general entities
    text = """
    Paul Graham founded Y Combinator in 2005 in Silicon Valley.
    The startup accelerator has funded companies like Airbnb and Dropbox.
    Jessica Livingston was a co-founder and organized the first batch.
    The Demo Day event takes place in San Francisco every year.
    """
    
    # Process through pipeline
    doc = Document(text=text)
    parser = SentenceSplitter(chunk_size=512)
    nodes = parser.get_nodes_from_documents([doc])
    nodes_with_entities = extractor(nodes)
    
    # Display results
    print_extraction_summary(nodes_with_entities, max_nodes=5)


def test_gliner_extraction_career():
    """Test GLiNER on career/self-help content."""
    print("="*80)
    print("TEST 5: Career Entity Extraction")
    print("="*80)
    
    # Get career entity labels
    labels = get_gliner_entity_labels(schema_name="career")
    print(f"\n🎯 Using {len(labels)} career entity types")
    
    # Initialize extractor
    extractor = GLiNERExtractor(
        model_name="urchade/gliner_medium-v2.1",
        entity_labels=labels,
        threshold=0.5,
        device="mps"
    )
    
    # Test text with career entities
    text = """
    To become a Senior Software Engineer at Google, you should master Python and distributed systems.
    I spoke with my mentor, Jane Smith, who recommended taking the AWS Solutions Architect certification.
    The tech industry is constantly evolving, so continuous learning on platforms like Coursera is essential.
    """
    
    # Process through pipeline
    doc = Document(text=text)
    parser = SentenceSplitter(chunk_size=512)
    nodes = parser.get_nodes_from_documents([doc])
    nodes_with_entities = extractor(nodes)
    
    # Display results
    print_extraction_summary(nodes_with_entities, max_nodes=5)


def test_comparison_threshold():
    """Test different confidence thresholds."""
    print("="*80)
    print("TEST 5: Threshold Comparison")
    print("="*80)
    
    text = """
    The Transformer architecture uses multi-head attention mechanisms.
    It was developed at Google by the research team.
    """
    
    labels = get_gliner_entity_labels(schema_name="academic")
    
    for threshold in [0.3, 0.5, 0.7]:
        print(f"\n📊 Testing with threshold={threshold}")
        
        extractor = GLiNERExtractor(
            model_name="urchade/gliner_medium-v2.1",
            entity_labels=labels,
            threshold=threshold,
            device="mps"
        )
        
        doc = Document(text=text)
        parser = SentenceSplitter(chunk_size=512)
        nodes = parser.get_nodes_from_documents([doc])
        nodes_with_entities = extractor(nodes)
        
        # Count entities
        entity_count = 0
        for node in nodes_with_entities:
            for key, values in node.metadata.items():
                if key.isupper() and values:
                    entity_count += len(values)
        
        print(f"   Entities found: {entity_count}")


def main():
    """Run all tests."""
    print("\n🧪 GLiNER MinerU Integration Test Suite")
    print(f"{'='*80}\n")
    
    try:
        test_entity_label_selection()
        test_gliner_extraction_academic()
        test_gliner_extraction_technical()
        test_gliner_extraction_general()
        test_gliner_extraction_career()
        test_comparison_threshold()
        
        print("\n" + "="*80)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\n💡 Next Steps:")
        print("   1. Run minerU.py with entity metadata extraction enabled")
        print("   2. Compare entity extraction quality with previous span-marker results")
        print("   3. Monitor inference speed and memory usage")
        print("   4. Fine-tune threshold if needed (0.3-0.7 range)\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
