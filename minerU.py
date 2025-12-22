import os
from dotenv import load_dotenv
load_dotenv()

from copy import deepcopy
import asyncio
import anthropic
import base64
from io import BytesIO
import json
import subprocess
import nest_asyncio
nest_asyncio.apply()

from pathlib import Path
from PIL import Image
import time
import re
from typing import List, Dict, Optional

from llama_index.core import (
                        Document,
                        PromptTemplate,
                        Settings,
                        VectorStoreIndex,
                        )
from llama_index.core.node_parser import (
                                    MarkdownElementNodeParser,
                                    SentenceSplitter
                                    )
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.schema import ImageDocument, TextNode
from llama_index.core.tools import QueryEngineTool
from llama_index.core.vector_stores import MetadataFilters
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.anthropic import Anthropic
# from llama_index.llms.mistralai import MistralAI
from llama_index.llms.openai import OpenAI
# TEMPORARILY DISABLED: Package version conflict with llama-index-llms-anthropic>=0.10
# from llama_index.multi_modal_llms.anthropic import AnthropicMultiModal
from llama_index.postprocessor.colbert_rerank import ColbertRerank
from llama_index.core.question_gen import LLMQuestionGenerator
# from llama_index.question_gen.guidance import GuidanceQuestionGenerator
# from llama_index.core.prompts.guidance_utils import convert_to_handlebars
from llama_index.retrievers.bm25 import BM25Retriever
from guidance.models import OpenAI as GuidanceOpenAI

from db_operation import (
                check_if_milvus_database_collection_exist,
                check_if_mongo_database_namespace_exist
                )

from config import get_article_info, DATABASE_CONFIG, EMBEDDING_CONFIG
import rag_factory
from utils import (
                change_default_engine_prompt_to_in_detail,
                display_prompt_dict,
                get_article_link,
                get_database_and_llamaparse_collection_name,
                get_fusion_tree_keyphrase_sort_detail_tool_simple,
                get_fusion_tree_keyphrase_filter_sort_detail_engine,
                get_fusion_tree_page_filter_sort_detail_engine,
                get_summary_storage_context,
                get_summary_tree_detail_tool,
                get_text_nodes_from_query_keyphrase,
                get_llamaparse_vector_store_docstore_and_storage_context,
                stitch_prev_next_relationships,
                )                
from llama_index.core.response_synthesizers.factory import get_response_synthesizer
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType

ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

def build_section_index_mineru(content_list: List[Dict]) -> Dict[str, Dict]:
    """
    Builds a section index from MinerU content list.
    """
    # Patterns for section detection
    # 1. Numbered sections: "1. Introduction", "2.1 Methodology", "A.1 Appendix"
    md_num_pattern = re.compile(r'^(\d+(?:\.\d+)*|[A-Z](?:\.\d+)*)\s+(.+)$')
    # 2. Plain numbered headings like '5. CONCLUSION' or '2.1. Methodology'
    plain_num_pattern = re.compile(r'^(\d+(?:\.\d+)*|[A-Z](?:\.\d+)*)(?:\s{1,}|\.)+(.+)$')
    
    section_headings = []
    seen_titles = set()
    
    for item in content_list:
        if item.get('type') == 'text':
            text = item.get('text', '').strip()
            if not text:
                continue
                
            # MinerU uses text_level to indicate headings. 
            # If present, we trust it. If not, we use regex but with strict heuristics.
            is_heading_marked = item.get('text_level') is not None
            
            # Check for numbered patterns
            m = md_num_pattern.match(text) or plain_num_pattern.match(text)
            
            section_num = ""
            section_title = ""
            
            if is_heading_marked:
                if m:
                    section_num = m.group(1)
                    section_title = m.group(2).strip()
                else:
                    section_num = ""
                    section_title = text
            elif m:
                # Not marked as heading, but matches regex. Use strict heuristics.
                section_num = m.group(1)
                section_title = m.group(2).strip()
                
                # HEURISTIC: Filter out paragraphs misidentified as headings
                # 1. Titles are rarely very long
                if len(text) > 100 or text.count(' ') > 10:
                    continue
                # 2. If section_num is a single letter, it's likely a sentence if the title is long and has lowercase
                if len(section_num) == 1 and section_num.isalpha() and any(c.islower() for c in section_title):
                    continue
                # 3. If it ends with a period and is long, it's a sentence
                if text.endswith('.') and len(text) > 40:
                    continue
            else:
                # Fallback: uppercase titles (only if short)
                if len(text) <= 60 and text.upper() == text and any(c.isalpha() for c in text) and len(text) > 3:
                    section_num = ""
                    section_title = text
                else:
                    continue
            
            # Now we have section_num and section_title
            page_num = item.get('page_idx', 0) + 1
            key = (section_title.lower(), page_num)
            if key not in seen_titles:
                section_headings.append({
                    'start_page': page_num,
                    'section_num': section_num,
                    'title': section_title,
                    'level': section_num.count('.') + 1 if section_num else 1,
                })
                seen_titles.add(key)

    if not section_headings:
        return {}

    # Second pass: calculate end pages
    # Find max page
    total_pages = max(item.get('page_idx', 0) for item in content_list) + 1
    
    for i, section in enumerate(section_headings):
        current_level = section['level']
        end_page = total_pages
        
        for j in range(i + 1, len(section_headings)):
            next_section = section_headings[j]
            if next_section['level'] <= current_level:
                # Next section at same or higher level - this section ends on that page
                # We use the start page of the next section to be inclusive of shared pages
                end_page = next_section['start_page']
                break
        
        section['end_page'] = max(section['start_page'], end_page)

    # Build the final index
    section_index = {}
    for section in section_headings:
        section_info = {
            'start_page': section['start_page'],
            'end_page': section['end_page'],
            'section_num': section['section_num'],
            'title': section['title'],
            'level': section['level'],
        }
        
        if section['section_num']:
            section_index[section['section_num'].lower()] = section_info
        
        title_key = section['title'].lower()
        if title_key not in section_index:
            section_index[title_key] = section_info
            
        if section['section_num']:
            full_key = f"{section['section_num']} {section['title']}".lower()
            section_index[full_key] = section_info

    # Add aliases
    common_aliases = {
        'abstract': ['abstract', 'summary'],
        'conclusion': ['conclusion', 'conclusions', 'concluding remarks'],
        'references': ['references', 'bibliography'],
        'appendix': ['appendix', 'appendices', 'supplementary'],
    }
    
    for canonical, aliases in common_aliases.items():
        if canonical in section_index:
            for alias in aliases:
                if alias not in section_index:
                    section_index[alias] = section_index[canonical]
                    
    return section_index


def html_table_to_markdown(html_content):
    """
    Converts an HTML table to a Markdown table using pandas.
    """
    try:
        import pandas as pd
        from io import StringIO
        # read_html returns a list of DataFrames
        dfs = pd.read_html(StringIO(html_content))
        if dfs:
            return dfs[0].to_markdown(index=False)
    except Exception as e:
        print(f"⚠️ Could not convert HTML table to Markdown: {e}")
    return html_content


def load_document_mineru(content_list_path):
    """
    Load MinerU content_list.json and convert to LlamaIndex Documents.
    Groups items by page to maintain page-level structure.
    """
    with open(content_list_path, 'r') as f:
        content_list = json.load(f)
    
    # Group by page_idx
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
                    # Convert HTML table to Markdown for better LLM understanding
                    markdown_table = html_table_to_markdown(item['table_body'])
                    table_parts.append(markdown_table)
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
            
            elif item_type == 'equation':
                if item.get('text'):
                    page_content.append(item.get('text'))
            
            elif item.get('text'):
                page_content.append(item.get('text'))
        
        page_text = "\n\n".join(page_content)
        
        # Create a metadata dict similar to LlamaParse
        metadata = {
            "page": p_idx + 1,
            "parser": "mineru"
        }
        
        documents.append(
            Document(
                text=page_text,
                metadata=metadata,
            )
        )
    return documents


def load_image_text_nodes_mineru(content_list_path, base_dir, article_dir, article_name):
    """
    Extract images from MinerU output and generate descriptions using Claude Vision API.
    Reuses the same logic as LlamaParse for fair comparison.
    """
    with open(content_list_path, 'r') as f:
        content_list = json.load(f)
    
    image_items = [item for item in content_list if item.get('type') == 'image']
    
    if not image_items:
        print("📷 No images found in MinerU output.")
        return []

    # Prepare image_dicts in the format expected by the Claude logic
    image_dicts = []
    for item in image_items:
        # MinerU img_path is relative to the output dir
        full_path = os.path.join(base_dir, item['img_path'])
        image_dicts.append({
            "path": full_path,
            "page_number": item.get('page_idx', 0) + 1,
            "caption": " ".join(item.get('image_caption', []))
        })

    # Initialize Anthropic client
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    
    # Cache file - use a different suffix for MinerU to avoid collision
    cache_file = Path(f"./data/{article_dir}/{article_name.replace('.pdf', '_mineru_image_descriptions.json')}")
    
    # Load existing cache
    descriptions_cache = {}
    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                descriptions_cache = json.load(f)
            print(f"\n📂 Loaded {len(descriptions_cache)} cached MinerU image descriptions")
        except:
            descriptions_cache = {}
    
    print(f"\n📷 Processing {len(image_dicts)} MinerU images with Claude Vision...")
    
    img_text_nodes = []
    
    for i, image_dict in enumerate(image_dicts):
        image_path = image_dict["path"]
        image_name = os.path.basename(image_path)
        
        # Check cache first
        if image_name in descriptions_cache:
            description = descriptions_cache[image_name]["description"]
            text_node = TextNode(
                text=description,
                metadata={
                    "path": image_path,
                    "image_name": image_name,
                    "page": image_dict.get("page_number", 0),
                    "type": "image_description",
                    "parser": "mineru"
                }
            )
            img_text_nodes.append(text_node)
            continue
        
        try:
            # Load image and check dimensions
            with Image.open(image_path) as img:
                width, height = img.size
                max_dimension = 8000
                if width > max_dimension or height > max_dimension:
                    scale = min(max_dimension / width, max_dimension / height)
                    img = img.resize((int(width * scale), int(height * scale)), Image.Resampling.LANCZOS)
                
                buffer = BytesIO()
                ext = os.path.splitext(image_path)[1].lower()
                img_format = "PNG" if ext == ".png" else "JPEG"
                img.save(buffer, format=img_format)
                image_data = base64.standard_b64encode(buffer.getvalue()).decode("utf-8")
            
            media_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
            
            # Call Claude Vision API
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_data,
                                },
                            },
                            {
                                "type": "text",
                                "text": f"Describe this image in detail. Caption: {image_dict['caption']}. Include all visual elements, labels, arrows, diagrams, charts, or any text visible in the image."
                            }
                        ],
                    }
                ]
            )
            
            description = message.content[0].text
            descriptions_cache[image_name] = {"description": description}
            
            text_node = TextNode(
                text=description,
                metadata={
                    "path": image_path,
                    "image_name": image_name,
                    "page": image_dict.get("page_number", 0),
                    "type": "image_description",
                    "parser": "mineru"
                }
            )
            img_text_nodes.append(text_node)
            
            # Save cache periodically
            with open(cache_file, "w") as f:
                json.dump(descriptions_cache, f)
                
        except Exception as e:
            print(f"⚠️ Error processing image {image_name}: {e}")

    return img_text_nodes


def create_and_save_vector_index_to_milvus_database(_base_nodes, _objects, _image_text_nodes):
    """
    Creates a VectorStoreIndex from nodes and saves it to Milvus.
    """
    _recursive_index = VectorStoreIndex(
        nodes=_base_nodes + _objects + _image_text_nodes,
        storage_context=storage_context_vector,
    )
    return _recursive_index

# Set OpenAI API key, LLM, and embedding model

# openai.api_key = os.environ['OPENAI_API_KEY']
# llm = OpenAI(model="gpt-4o", temperature=0.0)
# Settings.llm = llm

# llm = MistralAI(
#     model="mistral-large-latest", 
#     temperature=0.0,
#     max_tokens=2500,
#     api_key=MISTRAL_API_KEY
#     )
# Settings.llm = llm

llm = Anthropic(
            model="claude-sonnet-4-0",  # Updated from deprecated claude-3-5-sonnet-20240620
            temperature=0.0,
            max_tokens=2500,
            api_key=ANTHROPIC_API_KEY,
            )
Settings.llm = llm

# Create embedding model with smaller batch size to avoid token limits
embed_model = OpenAIEmbedding(
    model_name=EMBEDDING_CONFIG["model_name"],
    embed_batch_size=10  # Reduce batch size to avoid hitting 300k token limit
)
Settings.embed_model = embed_model
embed_model_dim = EMBEDDING_CONFIG["dimension"]
embed_model_name = EMBEDDING_CONFIG["short_name"]

# Get configuration from config.py
article_info = get_article_info()
rag_settings = article_info["rag_settings"]

# Article details
article_dictory = article_info["directory"]
article_name = article_info["filename"]

# Create database link
article_link = get_article_link(article_dictory,
                                article_name
                                )

# =============================================================================
# Parser Configuration: MinerU
# =============================================================================
chunk_method = "mineru"

# Global chunking configuration (from config.py)
chunk_size = rag_settings["chunk_size"]
chunk_overlap = rag_settings["chunk_overlap"]

page_filter_verbose = rag_settings.get("page_filter_verbose", False)

# Create database name and colleciton names
(database_name, 
collection_name_vector,
collection_name_summary) = get_database_and_llamaparse_collection_name(
                                                            article_dictory, 
                                                            chunk_method, 
                                                            embed_model_name, 
                                                            "mineru", # parse_method
                                                            chunk_size,
                                                            chunk_overlap,
                                                            )

# Initiate Milvus and MongoDB database (from config.py)
uri_milvus = DATABASE_CONFIG["milvus_uri"]
uri_mongo = DATABASE_CONFIG["mongo_uri"]

# Check if the vector index has already been saved to Milvus database.
save_index_vector = check_if_milvus_database_collection_exist(uri_milvus, 
                                                       database_name, 
                                                       collection_name_vector
                                                       )

# Check if the vector document has already been saved to MongoDB database.
add_document_vector = check_if_mongo_database_namespace_exist(uri_mongo, 
                                                       database_name, 
                                                       collection_name_vector
                                                       )

# Check if the summary document has already been saved to MongoDB database.
add_document_summary = check_if_mongo_database_namespace_exist(uri_mongo, 
                                                       database_name, 
                                                       collection_name_summary
                                                       )

# Create vector store, vector docstore, and vector storage context
(vector_store,
 vector_docstore,
 storage_context_vector) = get_llamaparse_vector_store_docstore_and_storage_context(uri_milvus,
                                                                    uri_mongo,
                                                                    database_name,
                                                                    collection_name_vector,
                                                                    embed_model_dim
                                                                    )

# Create summary summary storage context
storage_context_summary = get_summary_storage_context(uri_mongo,
                                                    database_name,
                                                    collection_name_summary
                                                    )

# Always load JSON (from cache if available) to build section index
mineru_base_dir = f"./data/{article_dictory}/mineru_output/{article_name.replace('.pdf', '')}/vlm"
json_cache_path = Path(os.path.join(mineru_base_dir, f"{article_name.replace('.pdf', '')}_content_list.json"))

section_index = None  # Initialize section index

if json_cache_path.exists():
    print(f"\n📂 Loading cached JSON from {json_cache_path} for section index...")
    with open(json_cache_path, "r") as f:
        json_objs_for_index = json.load(f)
    
    # Build section index from the JSON
    section_index = build_section_index_mineru(json_objs_for_index)
        
    if section_index:
        unique_sections = sorted(
            {id(v): v for v in section_index.values()}.values(),
            key=lambda x: (x['start_page'], x['section_num'] or "")
        )
        print(f"\n📑 Built section index with {len(unique_sections)} unique sections:")
        for s in unique_sections:
            prefix = f"{s['section_num']} " if s['section_num'] else ""
            print(f"   - {prefix}{s['title']} (Pages {s['start_page']}-{s['end_page']})")
else:
    print(f"⚠️ JSON cache not found at {json_cache_path}, section index not available")

# Initialize nodes
base_nodes = []
objects = []
image_text_nodes = []

# Load documnet nodes if either vector index or docstore not saved.
if save_index_vector or add_document_vector or add_document_summary: 

    # MinerU output path
    mineru_base_dir = f"./data/{article_dictory}/mineru_output/{article_name.replace('.pdf', '')}/vlm"
    content_list_path = os.path.join(mineru_base_dir, f"{article_name.replace('.pdf', '')}_content_list.json")
    
    if not os.path.exists(content_list_path):
        print(f"🌐 MinerU output not found. Running MinerU wrapper...")
        # Call the wrapper script
        subprocess.run([
            "python", "mineru_wrapper.py", 
            os.path.join(f"./data/{article_dictory}", article_name),
            f"./data/{article_dictory}/mineru_output"
        ], check=True)
    
    print(f"\n📂 Loading MinerU results from {content_list_path}...")
    docs = load_document_mineru(content_list_path)
    
    # Use SentenceSplitter for MinerU documents
    node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    base_nodes = node_parser.get_nodes_from_documents(docs)
    objects = [] # MinerU doesn't have separate objects like LlamaParse yet
    
    # Load image descriptions using the same Claude logic
    image_text_nodes = load_image_text_nodes_mineru(
        content_list_path, 
        mineru_base_dir, 
        article_dictory, 
        article_name
    )

if save_index_vector == True:
    recursive_index = create_and_save_vector_index_to_milvus_database(
                                                                    base_nodes,
                                                                    objects, 
                                                                    image_text_nodes
                                                                    )                                 
else:
    # Load from Milvus database
    recursive_index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context_vector
        )

summary_tool_description = (
    "Useful for summarization or full context questions related to the ENTIRE document. "
    "Use this function ONLY when user asks a question that requires understanding the "
    "FULL context or storyline of the WHOLE document, for example when user asks "
    "'What is this document about?', 'Give me an overview of the entire paper', "
    "'What are the main themes throughout the document?', or when the query involves "
    "comparing across ALL sections of the document. "
    "DO NOT use this tool for questions about SPECIFIC SECTIONS like 'Introduction', "
    "'Conclusion', 'Evaluation', 'Methodology', etc. - use page_filter_tool instead. "
    "NEVER use this tool for questions about EQUATIONS, FORMULAS, FIGURES, or TABLES - "
    "use keyphrase_tool instead for those. Even if asking about multiple equations "
    "collectively (e.g., 'What do equations 1-4 represent?'), use keyphrase_tool. "
    )

summary_tool = get_summary_tree_detail_tool(
                                        summary_tool_description,
                                        storage_context_summary
                                        )

if add_document_vector == True:
    # Stitch prev/next relationships so PrevNextNodePostprocessor can retrieve neighbors
    all_nodes_vector = stitch_prev_next_relationships(base_nodes + objects + image_text_nodes)
    # Save document nodes to Mongodb docstore at the server
    storage_context_vector.docstore.add_documents(all_nodes_vector)
    print(f"\n✅ Stitched prev/next relationships for {len(all_nodes_vector)} nodes")

if add_document_summary == True:
    # Stitch prev/next relationships so PrevNextNodePostprocessor can retrieve neighbors
    all_nodes_summary = stitch_prev_next_relationships(base_nodes + objects + image_text_nodes)
    # Save document nodes to Mongodb docstore at the server
    storage_context_summary.docstore.add_documents(all_nodes_summary)

# =============================================================================
# Fusion BM25 + Vector Retrieval Configuration for keyphrase_tool
# =============================================================================
# This uses a hybrid retrieval approach combining:
# 1. BM25 (keyword/lexical search) - good for exact matches like "equation (1)"
# 2. Vector embeddings (semantic search) - good for meaning-based retrieval
# 3. QueryFusionRetriever with reciprocal rank fusion
# 4. ColBERT reranking for final ranking

# Fusion retrieval parameters
# NOTE: Large values are now safe thanks to MetadataStripperPostprocessor in utils.py
# which removes bloated LlamaParse metadata before LLM synthesis (reduced 253K -> 11K tokens)
similarity_top_k_fusion = rag_settings["similarity_top_k_fusion"]
fusion_top_n = rag_settings["fusion_top_n"]
num_queries_fusion = rag_settings["num_queries"]
rerank_top_n = rag_settings["rerank_top_n"]
num_nodes_prev_next = rag_settings["num_nodes"]

reranker = ColbertRerank(
    top_n=rerank_top_n,
    model="colbert-ir/colbertv2.0",
    tokenizer="colbert-ir/colbertv2.0",
    keep_retrieval_score=True,
)

# queries for testing
# query = "What are the main findings of this paper?"
# RAG-Anything paper queries
# query = "Describe Figure 1 in detail. What visual elements and workflow does it show?"
# query = "In the Accuracy (%) on DocBench Dataset table (table 2), what methods are being compared and what is the worst performing method?"
# query = "In the Accuracy (%) on MMLongBench Dataset table (table 3), what methods are being compared, and what is the best performing method?"
# query = "Please summarize the content in the Introduction section."
# query = "Please summarize the content from pages 1 to 2."
# query = "Please summarize the content from pages 15 to 16."
# query = "Please summarize the content in the section in the Appendix: ADDITIONAL CASE STUDIES."
# query = "Please summarize the content in the Appendix section:CHALLENGES AND FUTURE DIRECTIONS FOR MULTI-MODAL RAG."
# query = "Please summarize the content in the Appendix section A.5: CHALLENGES AND FUTURE DIRECTIONS FOR MULTI-MODAL RAG."
# query = "Please summarize the content in Section A.2 ADDITIONAL CASE STUDIES"
# query = "Please summarize the content in Section 4: RELATED WORK"
# query = "Describe the content of Section 2.3 CROSS-MODAL HYBRID RETRIEVAL"
# query = "What is the content of the Evaluation section?"
# query = "Summarize the content of the Evaluation section."
# query = "Summarize the Conclusion section."
# query = "Summarize the section 3.4, CASE STUDIES."
# query = "Summarize the CASE STUDIES section."
# query = "Summarize the CROSS-MODAL HYBRID RETRIEVAL section."
# query = "What is in equation (1)?"
query = "What are in equation (3) and (4)?"
# query = "What are in equation (4)?"
# query = "What are in the equations (1), (2), (3), and (4)? What are they trying to represent collectively?"
# query = "What are in the equations (2) and (3)?"
# query = "How graphs are used in RAG-Anything's retrieval process as described in the paper?"
# query = "Did the paper mention about any tool used to parse mathematical equations from the PDF? If so, what is the name of the tool?"


# Build tools using the factory
keyphrase_tool = rag_factory.get_keyphrase_tool(
    query,
    recursive_index,
    storage_context_vector.docstore,
    reranker,
    llm,
    rag_settings,
)

page_filter_tool = rag_factory.get_page_filter_tool(
    query,
    reranker,
    recursive_index,
    storage_context_vector.docstore,
    llm,
    section_index=section_index,
    metadata_key="page",
    verbose=page_filter_verbose,
)

tools = [
    keyphrase_tool,
    summary_tool,
    page_filter_tool
]

# Build and run the engine
sub_question_engine = rag_factory.build_sub_question_engine(
    tools,
    llm,
    verbose=True
)

print(f"\n📝 QUERY: {query}\n")

# Execute query
response = sub_question_engine.query(query)

if response is not None:
    rag_factory.print_response_diagnostics(response)
    print(f"\n📝 RESPONSE:\n{response}\n")

# Cleanup resources
try:
    if 'vector_store' in dir() and hasattr(vector_store, 'client'):
        vector_store.client.release_collection(collection_name=collection_name_vector)
        vector_store.client.close()
except:
    pass

# Suppress warnings and force immediate exit
import warnings
warnings.filterwarnings('ignore')
os._exit(0)

