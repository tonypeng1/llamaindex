
import os
import json
from pathlib import Path
from llama_index.core.schema import Document, TextNode
from llama_index.core.node_parser import SentenceSplitter

# Mocking the functions from llamaparse.py to see what happens
def html_table_to_markdown(html_content):
    return html_content # Simplified

def load_document_mineru(content_list_path):
    with open(content_list_path, 'r') as f:
        content_list = json.load(f)
    
    pages = {}
    for item in content_list:
        p_idx = item.get('page_idx', 0)
        if p_idx not in pages:
            pages[p_idx] = []
        pages[p_idx].append(item)
    
    documents = []
    for p_idx in sorted(pages.keys()):
        page_items = pages[p_idx]
        page_content = []
        for item in page_items:
            item_type = item.get('type')
            if item_type == 'table':
                table_parts = []
                if item.get('table_caption'):
                    table_parts.append(" ".join(item['table_caption']))
                if item.get('table_body'):
                    table_parts.append(item['table_body'])
                if item.get('table_footnote'):
                    table_parts.append(" ".join(item['table_footnote']))
                if table_parts:
                    page_content.append("\n".join(table_parts))
            elif item_type == 'image':
                if item.get('image_caption'):
                    page_content.append(f"Figure: {' '.join(item['image_caption'])}")
            elif item_type == 'list':
                if item.get('list_items'):
                    page_content.append("\n".join(item['list_items']))
            elif item_type == 'code':
                code_parts = []
                if item.get('code_caption'):
                    code_parts.append(" ".join(item['code_caption']))
                if item.get('code_body'):
                    code_parts.append(f"```\n{item['code_body']}\n```")
                if code_parts:
                    page_content.append("\n".join(code_parts))
            elif item.get('text'):
                page_content.append(item.get('text'))
        
        page_text = "\n\n".join(page_content)
        metadata = {"page": p_idx + 1, "parser": "mineru"}
        documents.append(Document(text=page_text, metadata=metadata))
    return documents

article_dictory = "Rag_anything"
article_name = "RAG_Anything.pdf"
mineru_base_dir = f"./data/{article_dictory}/mineru_output/{article_name.replace('.pdf', '')}/vlm"
content_list_path = os.path.join(mineru_base_dir, f"{article_name.replace('.pdf', '')}_content_list.json")

if not os.path.exists(content_list_path):
    print(f"❌ Content list not found at {content_list_path}")
    exit(1)

docs = load_document_mineru(content_list_path)
print(f"Loaded {len(docs)} documents (pages)")

# The code in llamaparse.py uses chunk_size=1024 for MinerU
node_parser = SentenceSplitter(chunk_size=2000, chunk_overlap=200)
base_nodes = node_parser.get_nodes_from_documents(docs)
print(f"Created {len(base_nodes)} base nodes using SentenceSplitter(2000)")

# Check if any node is oversized and would be split again in create_and_save...
split_nodes = []
text_splitter = SentenceSplitter(chunk_size=2000, chunk_overlap=100)
for node in base_nodes:
    if len(node.text) > 8000:
        split_nodes.extend(text_splitter.split_nodes([node]))
    else:
        split_nodes.append(node)

print(f"Total text nodes after potential second split (8000 chars): {len(split_nodes)}")

# Count images
with open(content_list_path, 'r') as f:
    content_list = json.load(f)
image_items = [item for item in content_list if item.get('type') == 'image']
print(f"Found {len(image_items)} image items in content_list")

# Analyze each page
for i, doc in enumerate(docs):
    page_num = doc.metadata['page']
    nodes_for_page = [n for n in split_nodes if n.metadata.get('page') == page_num]
    print(f"Page {page_num}: {len(doc.text)} chars, {len(nodes_for_page)} nodes")
    if len(nodes_for_page) > 0:
        # Check if the text in the node matches the doc text
        node_text_combined = "\n".join([n.text for n in nodes_for_page])
        # SentenceSplitter might have some overlap, so we just check if the end of doc is in the last node
        last_node_text = nodes_for_page[-1].text
        doc_end = doc.text[-100:]
        if doc_end.strip() in last_node_text:
            print(f"  ✅ Page {page_num} text seems fully covered.")
        else:
            print(f"  ❌ Page {page_num} text might be truncated!")
            print(f"     Doc end: {doc_end.strip()[:50]}...")
            print(f"     Node end: {last_node_text[-50:].strip()}...")

