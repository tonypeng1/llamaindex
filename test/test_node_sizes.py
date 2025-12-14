"""
Quick test script to analyze node sizes from cached LlamaParse data
without making any API calls.
"""
import json
from pathlib import Path
from llama_index.core.node_parser import MarkdownElementNodeParser

# Path to cached JSON from LlamaParse
DATA_DIR = Path("./data/Rag_anything")
json_file = DATA_DIR / "RAG_Anything.json"

if not json_file.exists():
    print(f"❌ Cached JSON not found at {json_file}")
    print("   You need to run the full script at least once first.")
    exit(1)

print(f"📂 Loading cached JSON from {json_file}...")
with open(json_file, "r") as f:
    json_list = json.load(f)

print(f"   Found {len(json_list)} pages in JSON")

# Parse the JSON to get nodes (no API calls)
print("\n🔧 Parsing JSON to create nodes...")
node_parser = MarkdownElementNodeParser(num_workers=8)

# Create minimal document objects from JSON
from llama_index.core.schema import Document
documents = [Document(text=page.get("md", ""), metadata={"page": i}) for i, page in enumerate(json_list)]

base_nodes, objects = node_parser.get_nodes_from_documents(documents)

print(f"   Created {len(base_nodes)} base nodes")
print(f"   Created {len(objects)} object nodes (tables/figures)")

# Analyze sizes
print("\n🔍 Analyzing node sizes...")

all_nodes = base_nodes + objects
node_sizes = [(i, len(node.text), type(node).__name__, node.text[:100].replace('\n', ' ')) 
              for i, node in enumerate(all_nodes)]
node_sizes.sort(key=lambda x: x[1], reverse=True)

print(f"\n📊 Top 10 largest nodes by text length:")
print("-" * 80)
for idx, size, node_type, preview in node_sizes[:10]:
    tokens_approx = size // 4  # rough estimate
    status = "⚠️ OVERSIZED" if size > 8000 else "✅ OK"
    print(f"Node {idx:3d}: {size:>8,} chars (~{tokens_approx:>6,} tokens) - {node_type:15s} {status}")
    print(f"          Preview: {preview[:60]}...")
    print()

# Count problematic nodes
oversized = [n for n in node_sizes if n[1] > 8000]
print(f"\n📈 Summary:")
print(f"   Total nodes: {len(all_nodes)}")
print(f"   Nodes > 8000 chars (need splitting): {len(oversized)}")
print(f"   Nodes OK: {len(all_nodes) - len(oversized)}")

if oversized:
    print(f"\n⚠️  {len(oversized)} nodes need to be split before embedding!")
    print("   The current splitting logic should catch these.")
    
    # Check if largest node would cause 35K token error
    largest_chars = oversized[0][1]
    largest_tokens = largest_chars // 4
    print(f"\n   Largest node: {largest_chars:,} chars ≈ {largest_tokens:,} tokens")
    if largest_tokens > 8192:
        print(f"   ❌ This exceeds the 8,192 token embedding limit!")
    else:
        print(f"   🤔 This should be under the limit... the issue might be elsewhere")
