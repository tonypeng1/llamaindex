from copy import deepcopy
import asyncio
import anthropic
import base64
from io import BytesIO
import json
import subprocess
import nest_asyncio
import os
from dotenv import load_dotenv

from pathlib import Path
from PIL import Image
import time
from dotenv import load_dotenv

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

# Load environment variables from .env if available
try:
    load_dotenv()
except Exception:
    # Don't hard-fail if python-dotenv is not installed; print a hint for devs
    print("⚠️ python-dotenv not installed or failed to load; .env not loaded (pip install python-dotenv to enable)")


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


def create_and_save_vector_index_to_milvus_database(_base_nodes,
                                                   _objects,
                                                   _image_text_nodes):

    deepcopy_base_nodes = deepcopy(_base_nodes)
    deepcopy_image_text_nodes = deepcopy(_image_text_nodes)

    # Remove metadata from base nodes before saving to vector store (otherwise max character 
    # in dynamic field in Milvus will be exceeded).
    for node in deepcopy_base_nodes:
        node.metadata = {}

    # Convert IndexNodes to TextNodes to avoid large .obj serialization
    # IndexNodes contain .obj attribute with large OCR coordinate arrays that exceed Milvus limit
    converted_objects = []
    for obj in _objects:
        # Create a simple TextNode with just the text content
        text_node = TextNode(
            text=obj.text,
            id_=f"converted_{obj.node_id}",
            metadata={"type": "index_node_text", "original_id": obj.node_id}
        )
        converted_objects.append(text_node)
    if converted_objects:
        print(f"   🔄 Converted {len(converted_objects)} IndexNodes to TextNodes (stripped .obj attribute)")

    # Also clear large metadata from image text nodes (keep only essential fields)
    for img_node in deepcopy_image_text_nodes:
        # Keep only essential metadata, remove large OCR data if present
        img_node.metadata = {
            "type": img_node.metadata.get("type", "image_description"),
            "image_name": img_node.metadata.get("image_name", ""),
            "page_number": img_node.metadata.get("page_number", "unknown")
        }

    # Split any oversized nodes to fit within embedding model's token limit (8192 for text-embedding-3-small)
    # Using global chunk_size and chunk_overlap
    # Note: Some nodes may have very large text from OCR/table data
    text_splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    all_nodes = deepcopy_base_nodes + converted_objects + deepcopy_image_text_nodes
    
    # Debug: Find the largest nodes
    print(f"\n🔍 Analyzing node sizes...")
    node_sizes = [(i, len(node.text), type(node).__name__) for i, node in enumerate(all_nodes)]
    node_sizes.sort(key=lambda x: x[1], reverse=True)
    print(f"   Top 5 largest nodes by text length:")
    for idx, size, node_type in node_sizes[:5]:
        print(f"      Node {idx}: {size:,} chars (~{size//4:,} tokens) - {node_type}")
    
    split_nodes = []
    
    for node in all_nodes:
        # Check if node text is potentially too long
        # Using threshold based on chunk_size (approx 4 chars per token)
        # This ensures we catch all oversized nodes
        if len(node.text) > (chunk_size * 4):
            print(f"   📝 Splitting oversized node ({len(node.text)} chars, ~{len(node.text)//4} tokens)...")
            chunks = text_splitter.split_text(node.text)
            print(f"      → Split into {len(chunks)} chunks")
            for i, chunk in enumerate(chunks):
                new_node = TextNode(
                    text=chunk,
                    metadata={"chunk_index": i, "original_node_id": node.node_id}  # Minimal metadata
                )
                split_nodes.append(new_node)
        else:
            # Create a new TextNode with minimal metadata to avoid large _node_content serialization
            # The original node may have large metadata that gets serialized by Milvus into dynamic fields
            new_node = TextNode(
                text=node.text,
                id_=node.node_id,
                metadata={}  # Empty metadata - critical for Milvus field size limit
            )
            split_nodes.append(new_node)
    
    print(f"   📊 Total nodes after splitting: {len(split_nodes)} (from {len(all_nodes)} original)")

    _index = VectorStoreIndex(
        nodes=split_nodes,
        storage_context=storage_context_vector,
        )


    return _index


class LazyQueryEngine:
    """Instantiate the underlying query engine only when first queried.
    
    This wrapper defers tool creation until the sub-question engine actually calls
    the query, preventing eager page-filter initialization (and its logging) when
    the tool is merely registered but not used.
    """

    def __init__(self, factory):
        self._factory = factory
        self._engine = None
        self._lock = asyncio.Lock()

    async def _ensure_engine_async(self):
        async with self._lock:
            if self._engine is None:
                self._engine = self._factory()

    def _ensure_engine(self):
        if self._engine is None:
            self._engine = self._factory()

    def query(self, *args, **kwargs):
        self._ensure_engine()
        return self._engine.query(*args, **kwargs)

    async def aquery(self, *args, **kwargs):
        await self._ensure_engine_async()
        if hasattr(self._engine, "aquery"):
            return await self._engine.aquery(*args, **kwargs)
        return self._engine.query(*args, **kwargs)

    def __getattr__(self, item):
        self._ensure_engine()
        return getattr(self._engine, item)


import re
from typing import Dict, List, Optional


def build_section_index_mineru(content_list: List[Dict]) -> Dict[str, Dict]:
    """
    Builds a section index from MinerU content list.
    """
    import re
    
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


def format_section_list_for_prompt(section_index: Dict[str, Dict]) -> str:
    """
    Format the section index as a readable list for inclusion in LLM prompts.
    
    Args:
        section_index: The section index built by build_section_index
        
    Returns:
        A formatted string listing available sections
    """
    # Get unique sections (avoid duplicates from multiple keys)
    seen = set()
    sections = []
    
    for key, info in section_index.items():
        identifier = (info['section_num'], info['title'])
        if identifier not in seen:
            seen.add(identifier)
            sections.append(info)
    
    # Sort by start_page
    sections.sort(key=lambda x: (x['start_page'], x['section_num']))
    
    lines = []
    for s in sections:
        lines.append(f"  - Section {s['section_num']}: {s['title']} (pages {s['start_page']}-{s['end_page']})")
    
    return "\n".join(lines)


def create_custom_sub_question_prompt() -> str:
    """
    Create a custom prompt template for LLMQuestionGenerator.
    
    This function constructs a prompt template that guides the question generation
    process for sub-question decomposition. The template includes:
    - A prefix explaining the task
    - Example 1: Financial comparison (default LlamaIndex example)
    - Example 2: Page-based content summarization
    - A suffix for the actual query
    
    Returns:
    str: A prompt template string for use with LLMQuestionGenerator.
    """
    # Write in Python format string style
    PREFIX = """\
    Given a user question, and a list of tools, output a list of relevant sub-questions \
    in json markdown that when composed can help answer the full user question.
    The output MUST be a valid JSON object with an "items" key.

    IMPORTANT: Break down the user question into multiple atomic sub-questions if it contains \
    multiple parts, requests information about multiple entities, or requires multiple steps to answer. \
    Each sub-question should focus on a single aspect of the original query. Even if all sub-questions \
    use the same tool, they should be separated to ensure thorough retrieval.
    """

    # Default example from LlamaIndex
    EXAMPLE_1 = """\
    # Example 1
    <Tools>
    ```json
    [
    {
        "name": "uber_10k",
        "description": "Provides information about Uber financials for year 2021"
    },
    {
        "name": "lyft_10k",
        "description": "Provides information about Lyft financials for year 2021"
    }
    ]
    ```

    <User Question>
    Compare and contrast the revenue growth and EBITDA of Uber and Lyft for year 2021

    <Output>
    ```json
    {
    "items": [
        {"sub_question": "What is the revenue growth of Uber", "tool_name": "uber_10k"},
        {"sub_question": "What is the EBITDA of Uber", "tool_name": "uber_10k"},
        {"sub_question": "What is the revenue growth of Lyft", "tool_name": "lyft_10k"},
        {"sub_question": "What is the EBITDA of Lyft", "tool_name": "lyft_10k"}
    ]
    }
    ```

    """

    # Tailored example for page-based queries
    EXAMPLE_2 = """\
    # Example 2
    <Tools>
    ```json
    [
    {
        "name": "page_filter_tool",
        "description": "Perform a query search over the page numbers mentioned in the query"
    }
    ]
    ```

    <User Question>
    Summarize the content from pages 20 to 22 in the voice of the author by NOT retrieving the text verbatim

    <Output>
    ```json
    {
    "items": [
        {"sub_question": "Summarize the content on page 20 in the voice of the author by NOT retrieving the text verbatim", "tool_name": "page_filter_tool"},
        {"sub_question": "Summarize the content on page 21 in the voice of the author by NOT retrieving the text verbatim", "tool_name": "page_filter_tool"},
        {"sub_question": "Summarize the content on page 22 in the voice of the author by NOT retrieving the text verbatim", "tool_name": "page_filter_tool"}
    ]
    }
    ```

    """

    SUFFIX = """\
    # Example 3
    <Tools>
    ```json
    {tools_str}
    ```

    <User Question>
    {query_str}

    <Output>
    """

    # Combine and return
    custom_prompt = PREFIX + EXAMPLE_1 + EXAMPLE_2 + SUFFIX
    
    return custom_prompt


def get_fusion_tree_page_filter_sort_detail_tool_mineru(
    _query_str: str, 
    _reranker: ColbertRerank,
    _vector_index,
    _vector_docstore,
    _llm,
    _section_index: Optional[Dict[str, Dict]] = None,
    *,
    verbose: bool = False,
    ) -> QueryEngineTool:
    """
    Generate a QueryEngineTool that extracts specific pages from a document store,
    retrieves the text nodes corresponding to those pages, and creates a fusion tree.
    
    NOTE: This is adapted for MinerU which uses 'page' metadata key (integer).
    
    Parameters:
    _query_str (str): The query string from which to extract page numbers.
    _reranker (ColbertRerank): The reranker to use for the fusion tree.
    _vector_index: The vector index for retrieval.
    _vector_docstore: The document store containing the text nodes.
    _llm: The LLM to use for page number extraction.
    _section_index: Optional section index for accurate section-to-page resolution.

    Returns:
    QueryEngineTool: A tool that uses the fusion tree to answer queries about the specified pages.
    """
    
    # Build section list for prompt if section_index is available
    section_list_str = ""
    if _section_index:
        section_list_str = format_section_list_for_prompt(_section_index)
    
    # Use section-aware prompt if section_index is available
    if _section_index and section_list_str:
        # Deterministic pre-check: if the query explicitly mentions a section
        # title (e.g., "Conclusion"), resolve pages directly from the
        # `section_index` to avoid nondeterministic LLM fallback.
        q_lower = _query_str.lower()
        _page_numbers = None
        _result_str = None
        
        # Sort keys by length descending to find the most specific match first
        # This prevents "Case Studies" from matching when "Additional Case Studies" is present
        sorted_keys = sorted(_section_index.keys(), key=len, reverse=True)
        for key in sorted_keys:
            # Avoid accidental matches on very short keys (e.g., 'a', '1')
            if len(key) < 3:
                continue
                
            if key in q_lower:
                info = _section_index[key]
                _page_numbers = list(range(info['start_page'], info['end_page'] + 1))
                if verbose:
                    print(f"Deterministic section match for '{key}' -> pages {_page_numbers}")
                break

        # If we found a deterministic mapping, skip LLM and use it
        if _page_numbers is not None:
            if verbose:
                print(f"Using deterministic pages {_page_numbers} for query: {_query_str}")
        else:
            query_text = (
                '## Instruction:\n'
                'Analyze the user query and determine which pages to retrieve.\n\n'
                'If the query mentions SPECIFIC PAGE NUMBERS (e.g., "page 1", "pages 5-10"), extract those pages.\n'
                'If the query mentions a SECTION NAME or NUMBER, look up the exact pages from the section list below.\n\n'
                '## Available Sections in This Document:\n'
                '{section_list}\n\n'
                '## Output Format:\n'
                'Return a JSON object with either:\n'
                '- For explicit page numbers: {{"type": "pages", "pages": [1, 2, 3]}}\n'
                '- For section references: {{"type": "section", "section": "introduction"}}\n\n'
                '## Examples:\n'
                '**Query:** "Summarize pages 10-15"\n'
                '**Output:** {{"type": "pages", "pages": [10, 11, 12, 13, 14, 15]}}\n\n'
                '**Query:** "What does the Introduction say?"\n'
                '**Output:** {{"type": "section", "section": "introduction"}}\n\n'
                '**Query:** "Describe Section 2.1"\n'
                '**Output:** {{"type": "section", "section": "2.1"}}\n\n'
                '**Query:** "What is in the Evaluation section?"\n'
                '**Output:** {{"type": "section", "section": "evaluation"}}\n\n'
                '**Query:** "Summarize the content from page 1 to page 3"\n'
                '**Output:** {{"type": "pages", "pages": [1, 2, 3]}}\n\n'
                '## Now analyze the following query:\n'
                '**Query:** {query_str}\n'
            )

            if _page_numbers is None:
                prompt = PromptTemplate(query_text)
                _result_str = _llm.predict(prompt=prompt, query_str=_query_str, section_list=section_list_str)
            else:
                _result_str = None
        
        # If we didn't call the LLM because of deterministic mapping, _result_str
        # may be None. Only attempt to JSON-decode if we actually invoked the LLM.
        try:
            if _result_str is None:
                _result = None
            else:
                _result = json.loads(_result_str)

            if _result is not None:
                if _result.get('type') == 'section':
                    # Look up section in the index
                    section_key = _result.get('section', '').lower().strip()
                    section_info = _section_index.get(section_key)
                    
                    if section_info:
                        _page_numbers = list(range(section_info['start_page'], section_info['end_page'] + 1))
                        if verbose:
                            print(f"Section '{section_key}' resolved to pages {_page_numbers}")
                    else:
                        # Fallback: try fuzzy matching
                        for key, info in _section_index.items():
                            if section_key in key or key in section_key:
                                _page_numbers = list(range(info['start_page'], info['end_page'] + 1))
                                if verbose:
                                    print(f"Section '{section_key}' fuzzy matched to '{key}', pages {_page_numbers}")
                                break
                        else:
                            # No match found, default to page 1
                            _page_numbers = [1]
                            if verbose:
                                print(f"Section '{section_key}' not found, defaulting to page 1")
                else:
                    # Explicit page numbers
                    _page_numbers = _result.get('pages', [1])
            # if _result is None, we already set _page_numbers via deterministic pre-check
                
        except json.JSONDecodeError:
            # Fallback: try to parse as simple list
            try:
                _page_numbers = json.loads(_result_str)
                if not isinstance(_page_numbers, list):
                    _page_numbers = [1]
            except:
                _page_numbers = [1]
    else:
        # Fallback to original behavior if no section_index
        query_text = (
        '## Instruction:\n'
        'Extract all page numbers from the user query. \n'
        'The page numbers can be indicated by: \n'
        '  1. Explicit phrases like "page" or "pages" \n'
        '  2. References to document sections (e.g., "Introduction" typically spans pages 1-2, \n'
        '     "Abstract" is usually page 1, "Conclusion" is typically near the end) \n'
        'Return the page numbers as a list of integers, sorted in ascending order. \n'
        'Do NOT include "**Output:**" in your response. \n'
        'If the query mentions a section name without explicit pages, estimate reasonable page range. \n'
        'If no page numbers or sections are mentioned, output [1]. \n'

        '## Examples:\n'
        '**Query:** "Give me the main events from page 1 to page 4." \n'
        '**Output:** [1, 2, 3, 4] \n'

        '**Query:** "Give me the main events in the first 6 pages." \n'
        '**Output:** [1, 2, 3, 4, 5, 6] \n'

        '**Query:** "Summarize pages 10-15 of the document." \n'
        '**Output:** [10, 11, 12, 13, 14, 15] \n'

        '**Query:** "What are the key findings on page 2?" \n'
        '**Output:** [2] \n'

        '**Query:** "Summarize the content in the Introduction section." \n'
        '**Output:** [1, 2, 3] \n'

        '**Query:** "What does the Abstract say?" \n'
        '**Output:** [1] \n'

        '**Query:** "Describe the Methodology section." \n'
        '**Output:** [3, 4, 5] \n'

        '**Query:** "What are the lessons learned by the author at the company?" \n'
        '**Output:** [1] \n'

        '## Now extract the page numbers from the following query: \n'

        '**Query:** {query_str} \n'
        )

        prompt = PromptTemplate(query_text)
        _page_numbers_str = _llm.predict(prompt=prompt, query_str=_query_str)
        _page_numbers = json.loads(_page_numbers_str)
    
    if verbose:
        print(f"Page_numbers in page filter: {_page_numbers}")

    # Get text nodes from the vector docstore that match the page numbers
    # NOTE: MinerU uses 'page' key with integer values
    _text_nodes = []
    for _, node in _vector_docstore.docs.items():
        page_num = node.metadata.get('page')
        if page_num is not None and page_num in _page_numbers:
            _text_nodes.append(node) 

    node_length = len(_vector_docstore.docs)
    if verbose:
        print(f"Node length in docstore: {node_length}")
        print(f"Text nodes retrieved from docstore length is: {len(_text_nodes)}")
        for i, n in enumerate(_text_nodes):
            print(f"Item {i+1} of the text nodes retrieved from docstore is page: {n.metadata.get('page')}")
    
    _vector_filter_retriever = _vector_index.as_retriever(
                                    similarity_top_k=node_length,
                                    filters=MetadataFilters.from_dicts(
                                        [{
                                            "key": "page", 
                                            "value": _page_numbers,
                                            "operator": "in"
                                        }]
                                    )
                                )
    
    # Calculate the number of nodes retrieved from the vector index on these pages
    _fusion_top_n_filter = len(_text_nodes)
    _num_queries_filter = 1

    _fusion_tree_page_filter_sort_detail_engine = get_fusion_tree_page_filter_sort_detail_engine(
                                                                _vector_filter_retriever,
                                                                _fusion_top_n_filter,
                                                                _text_nodes,
                                                                _num_queries_filter,
                                                                _reranker,
                                                                _vector_docstore,
                                                                [str(p) for p in _page_numbers]  # Convert to strings for display
                                                                )
    
    page_tool_description = (
                "Perform a query search over the page numbers mentioned in the query. "
                "Use this function when user only need to retrieve information from specific PAGES, "
                "for example when user asks 'What happened on PAGE 1?' "
                "or 'What are the things mentioned on PAGES 1 and 2?' "
                "or 'Describe the contents from PAGE 1 to PAGE 4'. "
                "DO NOT GENERATE A SUB-QUESTION ASKING ABOUT ONE PAGE ONLY "
                "IF EQUAL TO OR MORE THAN 2 PAGES ARE MENTIONED IN THE QUERY. "
                )
    
    _fusion_tree_page_filter_sort_detail_tool = QueryEngineTool.from_defaults(
                                                        name="page_filter_tool",
                                                        query_engine=_fusion_tree_page_filter_sort_detail_engine,
                                                        description=page_tool_description,
                                                        )

    return _fusion_tree_page_filter_sort_detail_tool


def build_page_filter_query_engine():
    tool = get_fusion_tree_page_filter_sort_detail_tool_mineru(
        query,
        reranker,
        recursive_index,
        storage_context_vector.docstore,
        llm,
        section_index,  # Pass the section index
        verbose=page_filter_verbose,
    )
    return tool.query_engine


def build_keyphrase_query_engine():
    """
    Build a fusion BM25 + Vector query engine for keyphrase-based retrieval.
    
    This function is called lazily when the keyphrase_tool is first used.
    It creates:
    1. BM25 retriever for keyword/lexical search
    2. Vector retriever for semantic search  
    3. QueryFusionRetriever combining both with reciprocal rank fusion
    4. ColBERT reranking for final ranking
    """
    print(f"\n🔧 Building keyphrase fusion engine for query...")
    
    # Get BM25 text nodes from docstore using keyphrase extraction
    keyphrase_text_nodes = get_text_nodes_from_query_keyphrase(
        storage_context_vector.docstore,
        similarity_top_k_fusion,
        query,
    )
    
    # Create BM25 retriever for keyword/lexical search
    bm25_keyphrase_retriever = BM25Retriever.from_defaults(
        similarity_top_k=similarity_top_k_fusion,
        nodes=keyphrase_text_nodes,
    )
    
    # Create vector retriever for semantic search
    vector_retriever = recursive_index.as_retriever(
        similarity_top_k=similarity_top_k_fusion,
    )
    
    # Create fusion query engine combining BM25 + Vector with reranking
    fusion_keyphrase_engine = get_fusion_tree_keyphrase_filter_sort_detail_engine(
        vector_retriever,
        storage_context_vector.docstore,
        bm25_keyphrase_retriever,
        fusion_top_n,
        num_queries_fusion,
        reranker,
        num_nodes_prev_next,
    )
    
    print(f"✅ Keyphrase fusion engine built successfully")
    return fusion_keyphrase_engine


nest_asyncio.apply()
# MISTRAL_API_KEY = os.environ['MISTRAL_API_KEY']
ANTHROPIC_API_KEY = os.environ['ANTHROPIC_API_KEY']
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

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
        vector_store=vector_store
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

# Tool descriptions for the 3-tool architecture
specific_tool_description = (
    "Useful for retrieving SPECIFIC context from the document. "
    "Use this function when user asks a specific question that may NOT require "
    "understanding the full context of the document, for example "
    "when user seeks factual answer or a specific detail from the document, "
    "for example, when user uses interrogative words like 'what', 'who', 'where', "
    "'when', 'why', 'how', which may not require understanding the entire document "
    "to provide an answer. "
    "ALWAYS use this tool for questions about EQUATIONS, FORMULAS, FIGURES, or TABLES, "
    "including questions asking what multiple equations represent collectively. "
    "Examples: 'What is in equation (1)?', 'What do equations (1)-(4) represent?', "
    "'What is the formula for X?', 'What does Figure 2 show?', 'Describe Table 1'. "
    )

# recursive_query_engine_prompts_dict = recursive_query_engine.get_prompts()
# type = "recursive_query_engine:"
# display_prompt_dict(type.upper(), recursive_query_engine_prompts_dict)

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
    "Examples where you should NOT use this tool: "
    "'Summarize the Introduction section', 'What is in the Evaluation section?', "
    "'Describe the Conclusion', 'Summarize Section 2.1', "
    "'What is in equation (1)?', 'What do equations (1)-(4) represent collectively?', "
    "'What does Figure 1 show?', 'Describe Table 2'. "
    )

summary_tool = get_summary_tree_detail_tool(
                                        summary_tool_description,
                                        storage_context_summary
                                        )

page_tool_description = (
    "Perform a query search over specific pages or SECTIONS of the document. "
    "Use this function when user asks about specific PAGES or specific SECTIONS. "
    "Examples for PAGES: 'What happened on page 1?', 'Summarize pages 5-10', "
    "'What is on page 3?'. "
    "Examples for SECTIONS: 'Summarize the Introduction section', "
    "'What is in the Evaluation section?', 'Describe the Conclusion', "
    "'Summarize Section 2.1', 'What does the Methodology section say?', "
    "'Summarize the content of the Abstract'. "
    "This tool maps section names to their actual page numbers automatically. "
    "DO NOT GENERATE A SUB-QUESTION ASKING ABOUT ONE PAGE ONLY "
    "IF EQUAL TO OR MORE THAN 2 PAGES ARE MENTIONED IN THE QUERY. "
    )

# query = "what is UBER Short-term insurance reserves reported in 2022?"
# query = "what is UBER's common stock subject to repurchase in 2021?"
# query = "What is UBER Long-term insurance reserves reported in 2021?"
# query = "What is the number of monthly active platform consumers in Q2 2021?"
# query = "What is the number of monthly active platform consumers in 2022?"
# query = "What is the number of trips in 2021?"
# query = "What is the free cash flow in 2021?"
# query = "What is the gross bookings of delivery in Q3 2021?"
# query = "What is the gross bookings in 2022?"
# query = "What is the value of mobility adjusted EBITDA in 2022?" 
# query = "What is the status of the classification of drivers?"
# query = "What is the comprehensive income (loss) attributable to Uber reported in 2021?"
# query = "What is the comprehensive income (loss) attributable to Uber Technologies reported in 2022?"
# query = "What are the data shown in the bar graph titled 'Monthly Active Platform Consumers'?"
# query = "Can you tell me the page number on which the bar graph titled 'Monthly Active Platform Consumers' is located?"
# query = "What are the data shown in the bar graph titled 'Monthly Active Platform Consumers' on page 43?"
# query = "What is the Q2 2020 value shown in the bar graph titled 'Monthly Active Platform Consumers' on page 43?"
# query = "What are the main risk factors for Uber?"
# query = "What are the data shown in the bar graph titled 'Trips'?"
# query = "What are the data shown in the bar graph titled 'Gross Bookings'?"

# query = "What is the benefit of multi-head attention instead of single-head attention?"
# query = "Describe the content of section 3.1"  # not wortking (cannot find the section even for Claude model)
# query = "Describe the content of section 3.1 with the title 'Encoder and Decoder Stackes'."  # WORK 
# query = "What is the caption of Figure 2?"
# query = "What is in equation (1)."
# query = "What is in equation (2)."  # not working
# query = "What is in equation (3)."  # not working
# query = "Is there any equation in section 5.3?"  # not working
# query = "Is there any equation in section 5.3 titlted 'Optimizer'?"
# query = "How many equations are there in the full context of this document.?"  # WORK!
# query = "How many equations are there in this document.?"  # WORK!
# query = "What is on page 6?"  # WORK!
# query = "How many tables are there in this document?"
# query = "What is table 1 about?"
# query = "What do the results in table 1 show?"
# query = "List all sections and subsections in this document. Keep the original section/subsection numbers."  # not working
# query = "List all sections and subsections in the full context of this document. Use the original section/subsection numbers."  # WORK!
# query = "Find out how many sections and subsections does this document have and use the results to describe the content of subsection 3.1."
# query = "List all sections with the section number and section title?"  # not working
# query = "Create a table of content."  # not working
# query = "What does Figure 1 show?" 
# query = "Describe Figure 1 in detail." 
# query = "What does table 1 show?"
# query = "What are the resutls in table 1?"
# query = "Describe Figure 2 in detail."
# query = "What is the title of table 2?"
# query = "In table 2 what do 'EN-DE' and 'EN-FR' mean?"
# query = "What is the BLEU score of the model 'MoE' in EN-FR in Table 2?" 
# query = "How do a query and a set of key value pairs work together in an attention function?" 
# query = "What is the formula for Scaled Dot-Product Attention?"
# query = "Describe the Transformer architecture shown in Figure 1."
# query = "Describe Figure 2 in detail. What visual elements does it contain?"

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

# =============================================================================
# Build tools with lazy initialization
# =============================================================================
# Both keyphrase_tool and page_filter_tool use LazyQueryEngine to defer 
# initialization until first use. This ensures the 'query' variable is defined
# when the BM25 retriever is built.

# Build the keyphrase_tool lazily (uses BM25 + Vector fusion)
lazy_keyphrase_engine = LazyQueryEngine(build_keyphrase_query_engine)

keyphrase_tool = QueryEngineTool.from_defaults(
    name="keyphrase_tool",
    query_engine=lazy_keyphrase_engine,
    description=specific_tool_description,
)

# Build the page_filter_tool lazily to avoid eager initialization
page_filter_verbose = rag_settings["page_filter_verbose"]

# Use LazyQueryEngine to defer initialization until first use
lazy_page_filter_engine = LazyQueryEngine(build_page_filter_query_engine)

page_filter_tool = QueryEngineTool.from_defaults(
    name="page_filter_tool",
    query_engine=lazy_page_filter_engine,
    description=page_tool_description,
)

# Create custom sub-question prompt for SubQuestionQueryEngine
CUSTOM_SUB_QUESTION_PROMPT = create_custom_sub_question_prompt()
question_gen = LLMQuestionGenerator.from_defaults(
                            llm=llm,
                            prompt_template_str=CUSTOM_SUB_QUESTION_PROMPT
                            )

# Define the 3 tools
tools = [
    keyphrase_tool,
    summary_tool,
    page_filter_tool
]

# Use TREE_SUMMARIZE response synthesizer (default behavior)
# Allow overriding response mode with environment variable for testing
response_mode_env = os.environ.get("RESPONSE_MODE", "TREE_SUMMARIZE").upper()
if response_mode_env not in {m.name for m in ResponseMode}:
    print(f"⚠️ Invalid RESPONSE_MODE={response_mode_env}; falling back to COMPACT")
    response_mode_env = "TREE_SUMMARIZE"

# If TREE_SUMMARIZE, use a more detailed summary prompt for verbosity
summary_template = None
if response_mode_env == "TREE_SUMMARIZE":
    detailed_tree_tmpl = (
        "Context information from multiple sources is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "You are an expert assistant. Provide a detailed, structured, and thorough answer "
        "to the query below. Include key points, important details, any equations or "
        "examples present in the context, and list steps or components when applicable. "
        "Be explicit and avoid omitting technical specifics.\n\n"
        "=== MATHEMATICAL FORMULAS ===\n"
        "For any mathematical equations or formulas in your response:\n"
        "1. Use $$ ... $$ delimiters for standalone/display equations (centered on their own line).\n"
        "2. Use $ ... $ delimiters for inline math (within a sentence).\n\n"
        "Query: {query_str}\n"
        "Detailed Answer: "
    )
    summary_template = PromptTemplate(detailed_tree_tmpl, prompt_type=PromptType.SUMMARY)

synth = get_response_synthesizer(
    response_mode=ResponseMode[response_mode_env],
    summary_template=summary_template,
)
print(f"\n🔧 Using {response_mode_env} response synthesizer for final answers")

# Create SubQuestionQueryEngine with the 3 tools and the specified synthesizer
sub_question_engine = SubQuestionQueryEngine.from_defaults(
                                        question_gen=question_gen, 
                                        query_engine_tools=tools,
                                        response_synthesizer=synth,
                                        verbose=True,
                                        )

print(f"\n📝 QUERY: {query}\n")

# Retry logic for GuidanceQuestionGenerator failures
# The guidance library sometimes fails to parse LLM output as valid JSON
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

response = None
for attempt in range(1, MAX_RETRIES + 1):
    try:
        response = sub_question_engine.query(query)
        break  # Success, exit retry loop
    except Exception as e:
        error_msg = str(e)
        if "Failed to parse pydantic object from guidance program" in error_msg or \
           "json" in error_msg.lower():
            # This is a transient JSON parsing error, retry
            print(f"⚠️ Attempt {attempt}/{MAX_RETRIES}: Guidance JSON parsing failed, retrying in {RETRY_DELAY}s...")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                print(f"❌ All {MAX_RETRIES} attempts failed. Error: {e}")
        else:
            # Different error, don't retry
            print(f"❌ Error getting answer from LLM: {e}")
            break

if response is not None:
    # Collect nodes with metadata (actual document nodes)
    document_nodes = []
    
    for i, n in enumerate(response.source_nodes):
        if bool(n.metadata): # the first few nodes may not have metadata (the LLM response nodes)
            # Store node info for sequential output
            page_num = n.metadata.get('page', n.metadata.get('source', 'unknown'))
            document_nodes.append({
                'page': page_num,
                'text': n.text,
                'score': n.score,
                'node_id': n.node_id,
            })
        # else:
        #     print(f"Item {i+1} question and response:\n{n.text}\n ")
    
    # Output sequential nodes with page numbers in a list
    if document_nodes:
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4")  # cl100k_base encoding (similar to Claude)
        
        print("\n" + "="*80)
        print("SEQUENTIAL NODES WITH PAGE NUMBERS AND TOKEN COUNTS (sent to LLM for final answer):")
        print("="*80)
        total_tokens = 0
        for i, node_info in enumerate(document_nodes, 1):
            node_id_prefix = node_info['node_id'][:8] if node_info.get('node_id') else 'UNKNOWN'
            node_tokens = len(enc.encode(node_info['text']))
            total_tokens += node_tokens
            print(f"  Node {i}: Page {node_info['page']} | {node_tokens:,} tokens | {len(node_info['text']):,} chars (ID: {node_id_prefix}..., Score: {round(node_info['score'], 3) if node_info['score'] else 'N/A'})")
        print(f"\n  📊 TOTAL: {len(document_nodes)} nodes, {total_tokens:,} tokens (context only, excludes system prompt)")
        
        # # Print the contents of each page sent to LLM
        # print("\n" + "="*80)
        # print("CONTENTS OF PAGES SENT TO LLM:")
        # print("="*80)
        # for i, node_info in enumerate(document_nodes, 1):
        #     print(f"\n--- Page {node_info['page']} (Node {i}) ---")
        #     print(node_info['text'])
        #     print("-" * 40)

    print(f"\n📝 RESPONSE:\n{response}\n")

vector_store.client.release_collection(collection_name=collection_name_vector)
vector_store.client.close()

