"""
Test script to verify LangExtract installation and basic functionality.
This script tests the installation without requiring API keys.
"""

import sys

def test_langextract_import():
    """Test that langextract can be imported."""
    try:
        import langextract as lx
        print("✅ langextract imported successfully")
        print(f"   Version: {lx.__version__ if hasattr(lx, '__version__') else 'unknown'}")
        return True
    except ImportError as e:
        print(f"❌ Failed to import langextract: {e}")
        return False

def test_google_genai_import():
    """Test that google-genai can be imported."""
    try:
        from google import genai
        print("✅ google-genai imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import google-genai: {e}")
        return False

def test_langextract_data_classes():
    """Test that core LangExtract data classes are available."""
    try:
        import langextract as lx
        
        # Test ExampleData class
        example = lx.data.ExampleData(
            text="Test document about AI and machine learning.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="topic",
                    extraction_text="AI and machine learning",
                    attributes={"category": "technology"}
                )
            ]
        )
        print("✅ LangExtract data classes (ExampleData, Extraction) work correctly")
        print(f"   Example text: '{example.text}'")
        print(f"   Example extraction: {example.extractions[0].extraction_class}")
        return True
    except Exception as e:
        print(f"❌ Failed to create LangExtract data objects: {e}")
        return False

def main():
    """Run all installation tests."""
    print("=" * 60)
    print("LangExtract Installation Test")
    print("=" * 60)
    print()
    
    tests = [
        ("Import langextract", test_langextract_import),
        ("Import google-genai", test_google_genai_import),
        ("Test LangExtract data classes", test_langextract_data_classes),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
        print("-" * 60)
        result = test_func()
        results.append(result)
        print()
    
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! LangExtract is ready to use.")
        print("\nNext steps:")
        print("1. Set up API key (optional, only if using Gemini):")
        print("   export GEMINI_API_KEY='your-api-key-here'")
        print("2. You can use LangExtract with Claude (your existing LLM)")
        print("3. Ready to create extraction schemas and integrate with your pipeline")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
