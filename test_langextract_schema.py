"""
Test LangExtract with Paul Graham essay schema.

This script:
1. Loads a sample from paul_graham_essay.pdf
2. Applies the extraction schema
3. Displays extracted structured metadata
4. Shows how it would enhance Milvus search
"""

import os
import sys
from pathlib import Path

import langextract as lx
from dotenv import load_dotenv
from llama_index.readers.file import PyMuPDFReader

from langextract_schemas import get_schema


# Load environment variables from .env file
load_dotenv()


def save_html_visualization(result, output_path="extraction_visualization.html"):
    """
    Generate and save an interactive HTML visualization of extractions.
    
    This creates a beautiful HTML page with:
    - Highlighted extractions in the original text
    - Color-coded by extraction class
    - Tooltips showing attributes
    - Interactive navigation
    """
    try:
        html_content = lx.visualize(result)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path
    except Exception as e:
        print(f"Warning: Could not generate HTML visualization: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_sample_text(pdf_path, num_pages=2):
    """Load first few pages from PDF as sample text."""
    loader = PyMuPDFReader()
    docs = loader.load(file_path=Path(pdf_path))
    
    # Get text from first num_pages
    sample_text = "\n\n".join([doc.text for doc in docs[:num_pages]])
    return sample_text, docs[:num_pages]


def extract_with_openai(text, schema_config):
    """
    Extract structured metadata using LangExtract with OpenAI GPT-4.
    
    Note: LangExtract natively supports OpenAI models.
    Using GPT-4 for high-quality extraction.
    """
    
    try:
        result = lx.extract(
            text_or_documents=text,
            prompt_description=schema_config["prompt"],
            examples=schema_config["examples"],
            model_id="gpt-4o",  # Using GPT-4 Omni for extraction
        )
        return result
    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback
        traceback.print_exc()
        return None


def display_extraction_results(result):
    """Display extracted metadata in a readable format."""
    if result is None:
        print("No extraction results to display.")
        return
    
    print("\n" + "=" * 80)
    print("EXTRACTION RESULTS")
    print("=" * 80)
    
    print(f"\nDocument ID: {result.document_id}")
    print(f"\nOriginal Text Length: {len(result.text)} characters")
    print(f"\nText Preview:\n{result.text[:200]}...\n")
    
    if not result.extractions:
        print("No extractions found.")
        return
    
    print(f"\nTotal Extractions: {len(result.extractions)}")
    print("\n" + "-" * 80)
    
    # Group by extraction class
    by_class = {}
    for extraction in result.extractions:
        class_name = extraction.extraction_class
        if class_name not in by_class:
            by_class[class_name] = []
        by_class[class_name].append(extraction)
    
    # Display by category
    for class_name, extractions in by_class.items():
        print(f"\n📌 {class_name.upper()} ({len(extractions)} items)")
        print("-" * 80)
        
        for i, extraction in enumerate(extractions, 1):
            print(f"\n  {i}. Text: \"{extraction.extraction_text}\"")
            if extraction.attributes:
                print(f"     Attributes: {extraction.attributes}")
            # Note: source_index may not be available in all LangExtract versions
            if hasattr(extraction, 'source_index') and extraction.source_index is not None:
                print(f"     Source Index: {extraction.source_index}")


def show_milvus_integration_example(result):
    """Show how extraction results would be stored in Milvus."""
    if result is None or not result.extractions:
        return
    
    print("\n" + "=" * 80)
    print("MILVUS INTEGRATION EXAMPLE")
    print("=" * 80)
    
    print("\nExtracted metadata would be added to your Milvus collection as fields:")
    print("\nExample node metadata structure:")
    
    # Build metadata structure
    metadata = {
        "original_text": result.text[:100] + "...",
        "source": "paul_graham_essay.pdf",
        "page": 1,
    }
    
    # Add extracted metadata
    concepts = []
    advice_items = []
    entities = []
    experiences = []
    time_refs = []
    
    for extraction in result.extractions:
        if extraction.extraction_class == "concept":
            concepts.append({
                "text": extraction.extraction_text,
                "category": extraction.attributes.get("category", "unknown"),
                "importance": extraction.attributes.get("importance", "unknown"),
            })
        elif extraction.extraction_class == "advice":
            advice_items.append({
                "text": extraction.extraction_text,
                "type": extraction.attributes.get("type", "unknown"),
                "domain": extraction.attributes.get("domain", "unknown"),
            })
        elif extraction.extraction_class == "entity":
            entities.append({
                "text": extraction.extraction_text,
                "role": extraction.attributes.get("role", "unknown"),
            })
        elif extraction.extraction_class == "experience":
            experiences.append({
                "text": extraction.extraction_text,
                "period": extraction.attributes.get("period", "unknown"),
            })
        elif extraction.extraction_class == "time":
            time_refs.append({
                "text": extraction.extraction_text,
                "decade": extraction.attributes.get("decade", "unknown"),
            })
    
    if concepts:
        metadata["concepts"] = concepts
        metadata["concept_categories"] = list(set([c["category"] for c in concepts]))
    
    if advice_items:
        metadata["advice"] = advice_items
        metadata["advice_domains"] = list(set([a["domain"] for a in advice_items]))
    
    if entities:
        metadata["entities"] = entities
        metadata["entity_names"] = [e["text"] for e in entities]
    
    if experiences:
        metadata["experiences"] = experiences
        metadata["time_periods"] = list(set([e["period"] for e in experiences]))
    
    if time_refs:
        metadata["time_references"] = time_refs
    
    print("\n```python")
    import json
    print(json.dumps(metadata, indent=2))
    print("```")
    
    print("\n" + "=" * 80)
    print("EXAMPLE HYBRID QUERIES")
    print("=" * 80)
    
    print("\nWith this metadata, you can now do queries like:")
    print("\n1. Semantic + Category Filter:")
    print('   "Find advice about startups" + filter(advice_domains contains "startups")')
    
    print("\n2. Entity-based Search:")
    print('   "What did Paul say about Y Combinator?" + filter(entities contains "Y Combinator")')
    
    print("\n3. Time Period Filter:")
    print('   "Experiences from college" + filter(time_periods contains "college")')
    
    print("\n4. Concept Importance:")
    print('   "High importance programming concepts" + filter(concept_categories="programming", importance="high")')


def main():
    """Run the extraction test."""
    print("=" * 80)
    print("LangExtract Schema Test - Paul Graham Essay")
    print("=" * 80)
    
    # Check for API key - try to get it from environment
    openai_key = os.environ.get('OPENAI_API_KEY')
    
    if not openai_key:
        print("\n⚠️  OPENAI_API_KEY not found in environment")
        print("\nPlease set your OpenAI API key:")
        print("  export OPENAI_API_KEY='your-key-here'")
        print("\nOr run with:")
        print("  OPENAI_API_KEY='your-key' uv run python test_langextract_schema.py")
        print("\nNote: LangExtract uses OpenAI GPT-4 for extraction by default.")
        print("      You can also use Gemini or other supported models.\n")
        return 1
    
    print(f"\n✅ API key found (length: {len(openai_key)})")
    
    # Load sample text
    pdf_path = "data/paul_graham/paul_graham_essay.pdf"
    print(f"\nLoading sample from: {pdf_path}")
    
    try:
        sample_text, docs = load_sample_text(pdf_path, num_pages=1)
        print(f"✅ Loaded {len(docs)} page(s)")
        print(f"   Text length: {len(sample_text)} characters")
        print(f"\n   First 200 chars: {sample_text[:200]}...")
    except Exception as e:
        print(f"❌ Failed to load PDF: {e}")
        return 1
    
    # Get extraction schema
    print("\n" + "-" * 80)
    print("Loading extraction schema...")
    schema_config = get_schema("paul_graham_detailed")
    print(f"✅ Schema loaded: paul_graham_detailed")
    print(f"   Extraction classes: concept, advice, experience, entity, time")
    print(f"   Example count: {len(schema_config['examples'])}")
    
    # Initialize LLM note
    print("\n" + "-" * 80)
    print("Note: Using OpenAI GPT-4 for extraction")
    print("(LangExtract has built-in support for OpenAI models)")
    
    # Run extraction
    print("\n" + "-" * 80)
    print("Running LangExtract...")
    print("(This may take 30-60 seconds...)")
    
    try:
        result = extract_with_openai(
            sample_text[:2000],  # Use first 2000 chars for faster testing
            schema_config,
        )
        
        if result:
            print("✅ Extraction completed")
            display_extraction_results(result)
            show_milvus_integration_example(result)
            
            # Generate HTML visualization
            print("\n" + "=" * 80)
            print("GENERATING INTERACTIVE HTML VISUALIZATION")
            print("=" * 80)
            
            html_path = save_html_visualization(result)
            if html_path:
                abs_path = os.path.abspath(html_path)
                print(f"\n✅ HTML visualization saved to: {html_path}")
                print(f"   Full path: {abs_path}")
                print(f"\n   Open in browser with:")
                print(f"   open {html_path}")
                print(f"\n   Or on macOS:")
                print(f"   open -a 'Google Chrome' {html_path}")
            else:
                print("\n⚠️  HTML visualization could not be generated")
        else:
            print("❌ Extraction returned no results")
            return 1
            
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        print("\nNote: This might be due to:")
        print("  1. API key configuration")
        print("  2. LangExtract model backend setup")
        print("  3. Network connectivity")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "=" * 80)
    print("✅ Schema test completed successfully!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review the extracted metadata above")
    print("2. Open extraction_visualization.html in your browser to see highlights")
    print("3. Adjust schema if needed (langextract_schemas.py)")
    print("4. Integrate with your existing pipeline (bm25_simple.py)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
