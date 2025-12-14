"""
Test script to evaluate different LlamaParse settings for equation recognition.

This script tests whether enabling specific parsing options improves
equation detection and LaTeX formatting in the RAG-Anything paper.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from llama_parse import LlamaParse

LLAMA_CLOUD_API_KEY = os.environ['LLAMA_CLOUD_API_KEY']
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')

article_dir = "Rag_anything"
article_name = "RAG_Anything.pdf"
article_link = f"./data/{article_dir}/{article_name}"


def test_parsing_config(config_name: str, parser: LlamaParse, output_suffix: str):
    """Parse document and save results for analysis."""
    print(f"\n{'='*60}")
    print(f"Testing: {config_name}")
    print(f"{'='*60}")
    
    output_path = Path(f"./data/{article_dir}/{article_name.replace('.pdf', f'_{output_suffix}.json')}")
    
    # Parse the document
    print(f"Parsing {article_link}...")
    json_objs = parser.get_json_result(article_link)
    
    # Save to cache
    with open(output_path, "w") as f:
        json.dump(json_objs, f, indent=2)
    print(f"Saved to {output_path}")
    
    # Analyze equation detection on page 4 (where equation 2 is)
    print(f"\n--- Analyzing Page 4 (Equation 2 location) ---")
    
    pages = json_objs[0].get('pages', [])
    page_4 = None
    for page in pages:
        if page.get('page') == 4:
            page_4 = page
            break
    
    if page_4:
        items = page_4.get('items', [])
        
        # Count item types
        type_counts = {}
        equation_items = []
        items_with_2 = []
        
        for item in items:
            item_type = item.get('type', 'unknown')
            type_counts[item_type] = type_counts.get(item_type, 0) + 1
            
            md = item.get('md', '')
            value = item.get('value', '')
            
            # Look for equation-type items
            if 'equation' in item_type.lower() or '$$' in md or item_type == 'formula':
                equation_items.append(item)
            
            # Look for items containing "(2)" which is the equation number
            if '(2)' in md or '(2)' in value:
                items_with_2.append(item)
            
            # Look for the actual equation content
            if 'R(d' in md or 'Vj' in md or 'Ej' in md:
                print(f"\n  Found potential equation content:")
                print(f"    Type: {item_type}")
                print(f"    MD: {md[:200]}...")
        
        print(f"\n  Item type counts: {type_counts}")
        print(f"  Equation-type items found: {len(equation_items)}")
        print(f"  Items containing '(2)': {len(items_with_2)}")
        
        if equation_items:
            print("\n  Equation items:")
            for eq in equation_items:
                print(f"    - Type: {eq.get('type')}, MD: {eq.get('md', '')[:100]}")
        
        if items_with_2:
            print("\n  Items with '(2)':")
            for item in items_with_2:
                print(f"    - Type: {item.get('type')}, MD: {item.get('md', '')[:100]}")
        
        # Check the markdown field for LaTeX delimiters
        md_content = page_4.get('md', '')
        has_latex_inline = '$' in md_content and '$$' not in md_content.replace('$$', '')
        has_latex_display = '$$' in md_content
        print(f"\n  LaTeX inline ($...$) detected: {has_latex_inline}")
        print(f"  LaTeX display ($$...$$) detected: {has_latex_display}")
        
        # Search for equation (2) pattern in full markdown
        if '(2)' in md_content:
            # Find context around (2)
            idx = md_content.find('(2)')
            start = max(0, idx - 100)
            end = min(len(md_content), idx + 100)
            print(f"\n  Context around '(2)' in markdown:")
            print(f"    ...{md_content[start:end]}...")
    else:
        print("  Page 4 not found!")
    
    return json_objs


# Test 1: Default settings (baseline)
print("\n" + "="*70)
print("TEST 1: Default settings (baseline)")
print("="*70)

parser_default = LlamaParse(
    api_key=LLAMA_CLOUD_API_KEY,
    verbose=True,
)
# Skip if cache already exists
default_cache = Path(f"./data/{article_dir}/{article_name.replace('.pdf', '_test_default.json')}")
if not default_cache.exists():
    test_parsing_config("Default Settings", parser_default, "test_default")
else:
    print(f"Skipping - cache exists at {default_cache}")


# Test 2: With parsing instruction for equations
print("\n" + "="*70)
print("TEST 2: With equation-focused parsing instruction")
print("="*70)

parsing_instruction_eq = """
This is an academic paper containing mathematical equations.
Please ensure:
1. All mathematical equations are wrapped in LaTeX delimiters ($ for inline, $$ for display equations)
2. Equation numbers like (1), (2), (3) that appear in the right margin should be preserved and associated with their equations
3. Mathematical symbols and subscripts should use proper LaTeX notation (e.g., V_j instead of Vj)
4. Identify equation items with type "equation" or "formula" when possible
"""

parser_with_instruction = LlamaParse(
    api_key=LLAMA_CLOUD_API_KEY,
    parsing_instruction=parsing_instruction_eq,
    verbose=True,
)
instruction_cache = Path(f"./data/{article_dir}/{article_name.replace('.pdf', '_test_instruction.json')}")
if not instruction_cache.exists():
    test_parsing_config("With Equation Instruction", parser_with_instruction, "test_instruction")
else:
    print(f"Skipping - cache exists at {instruction_cache}")


# Test 2b: With MORE AGGRESSIVE parsing instruction for subscripts
print("\n" + "="*70)
print("TEST 2b: With aggressive subscript-aware parsing instruction")
print("="*70)

parsing_instruction_aggressive = """
This is an academic paper containing mathematical equations with complex subscripts and superscripts.

CRITICAL INSTRUCTIONS FOR EQUATION PARSING:
1. Mathematical equations MUST be wrapped in LaTeX delimiters: use $$ for display equations (standalone), $ for inline math
2. Equation numbers (1), (2), (3) etc. in the right margin MUST be preserved using \\tag{n} notation

SUBSCRIPT/SUPERSCRIPT HANDLING - VERY IMPORTANT:
3. In PDFs, subscripts and superscripts often appear on SEPARATE LINES below or above the main text due to PDF rendering
4. When you see a single letter (like 'j', 'i', 'k') on a line by itself near an equation, it is likely a SUBSCRIPT that belongs to the variable above it
5. For example, if you see:
   "dchunk"
   "j"
   This should be interpreted as d_j^{chunk} or d_j^{\\text{chunk}} (the j is a subscript of d)

6. Similarly for:
   "(Vj , Ej ) = R(dchunk),"
   "j"
   The trailing 'j' is the subscript for 'd', making it: (V_j, E_j) = R(d_j^{\\text{chunk}})

7. Use proper LaTeX:
   - Subscripts: V_j, E_j, d_j
   - Superscripts: d^{chunk} or d^{\\text{chunk}}
   - Combined: d_j^{\\text{chunk}} means d with subscript j and superscript "chunk"
   - Calligraphic letters: \\mathcal{V}, \\mathcal{E} for fancy V and E

8. Look at the VISUAL CONTEXT in the PDF to correctly associate disconnected subscripts with their base variables
"""

parser_aggressive = LlamaParse(
    api_key=LLAMA_CLOUD_API_KEY,
    parsing_instruction=parsing_instruction_aggressive,
    verbose=True,
)
aggressive_cache = Path(f"./data/{article_dir}/{article_name.replace('.pdf', '_test_aggressive.json')}")
if not aggressive_cache.exists():
    test_parsing_config("Aggressive Subscript Instruction", parser_aggressive, "test_aggressive")
else:
    print(f"Skipping - cache exists at {aggressive_cache}")


# Test 3: With GPT-4o mode enabled
print("\n" + "="*70)
print("TEST 3: With GPT-4o mode for better equation recognition")
print("="*70)

if OPENAI_API_KEY:
    parser_gpt4o = LlamaParse(
        api_key=LLAMA_CLOUD_API_KEY,
        gpt4o_mode=True,
        gpt4o_api_key=OPENAI_API_KEY,
        parsing_instruction=parsing_instruction_eq,
        verbose=True,
    )
    gpt4o_cache = Path(f"./data/{article_dir}/{article_name.replace('.pdf', '_test_gpt4o.json')}")
    if not gpt4o_cache.exists():
        test_parsing_config("GPT-4o Mode", parser_gpt4o, "test_gpt4o")
    else:
        print(f"Skipping - cache exists at {gpt4o_cache}")
else:
    print("Skipping GPT-4o test - OPENAI_API_KEY not set")


# Test 4: Check if there are other parsing modes
print("\n" + "="*70)
print("TEST 4: Accurate mode with premium features")
print("="*70)

parser_accurate = LlamaParse(
    api_key=LLAMA_CLOUD_API_KEY,
    parsing_instruction=parsing_instruction_eq,
    premium_mode=True,  # Enable premium parsing
    verbose=True,
)
accurate_cache = Path(f"./data/{article_dir}/{article_name.replace('.pdf', '_test_premium.json')}")
if not accurate_cache.exists():
    try:
        test_parsing_config("Premium Mode", parser_accurate, "test_premium")
    except Exception as e:
        print(f"Premium mode failed (may require subscription): {e}")
else:
    print(f"Skipping - cache exists at {accurate_cache}")


# Final summary - Compare all cached results
print("\n" + "="*70)
print("SUMMARY: Comparing all parsing results")
print("="*70)

cache_files = [
    ("Original (llamaparse.json)", f"./data/{article_dir}/{article_name.replace('.pdf', '_llamaparse.json')}"),
    ("Test Default", f"./data/{article_dir}/{article_name.replace('.pdf', '_test_default.json')}"),
    ("Test Instruction", f"./data/{article_dir}/{article_name.replace('.pdf', '_test_instruction.json')}"),
    ("Test GPT-4o", f"./data/{article_dir}/{article_name.replace('.pdf', '_test_gpt4o.json')}"),
    ("Test Premium", f"./data/{article_dir}/{article_name.replace('.pdf', '_test_premium.json')}"),
]

for name, path in cache_files:
    cache_path = Path(path)
    if cache_path.exists():
        with open(cache_path, "r") as f:
            data = json.load(f)
        
        # Find page 4 and check for equation (2)
        pages = data[0].get('pages', [])
        for page in pages:
            if page.get('page') == 4:
                md = page.get('md', '')
                
                # Check various patterns
                has_eq_2_explicit = '(2)' in md
                has_latex = '$$' in md or ('$' in md and md.count('$') >= 2)
                has_r_dchunk = 'R(d' in md or 'R(dchunk)' in md
                
                # Count equation-type items
                items = page.get('items', [])
                eq_items = [i for i in items if 'equation' in i.get('type', '').lower() or 'formula' in i.get('type', '').lower()]
                
                print(f"\n{name}:")
                print(f"  - Has '(2)' in markdown: {has_eq_2_explicit}")
                print(f"  - Has LaTeX delimiters: {has_latex}")
                print(f"  - Has equation content R(d...): {has_r_dchunk}")
                print(f"  - Equation-type items on page 4: {len(eq_items)}")
                
                # Show equation (2) context if found
                if has_eq_2_explicit:
                    idx = md.find('(2)')
                    start = max(0, idx - 80)
                    end = min(len(md), idx + 50)
                    context = md[start:end].replace('\n', ' ')
                    print(f"  - Context: ...{context}...")
                break
    else:
        print(f"\n{name}: File not found")

print("\n" + "="*70)
print("Testing complete!")
print("="*70)
