from copy import deepcopy
import anthropic
import base64
from io import BytesIO
import json
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
                                    LlamaParseJsonNodeParser,
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
from llama_index.question_gen.guidance import GuidanceQuestionGenerator
from llama_index.core.prompts.guidance_utils import convert_to_handlebars
from llama_index.retrievers.bm25 import BM25Retriever
from llama_parse import LlamaParse
from guidance.models import OpenAI as GuidanceOpenAI

from db_operation import (
                check_if_milvus_database_collection_exist,
                check_if_mongo_database_namespace_exist
                )

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


def load_document_llamaparse_jason(_json_objs):
    json_list = _json_objs[0]["pages"]

    documents = []
    for _, page in enumerate(json_list):
        documents.append(
            Document(
                text=page.get("md") or page.get("text"),
                metadata=page,
            )
        )

    return documents


def get_nodes_from_document_llamaparse(
        _documnet, 
        _parse_method,
        ):
    
    if _parse_method == "jason":
        _node_parser = LlamaParseJsonNodeParser(
                                    num_workers=16, 
                                    include_metadata=True
                                    )
        _nodes = _node_parser.get_nodes_from_documents(documents=_documnet)  # this step may take a while
        _base_nodes, _objects = _node_parser.get_nodes_and_objects(_nodes)
        
        # Fix: Replace truncated text with full markdown content from metadata
        # LlamaParseJsonNodeParser puts table headers in text but full content in metadata.md
        for node in _base_nodes:
            md_content = node.metadata.get('md', '')
            if md_content and len(md_content) > len(node.text):
                node.text = md_content

    return _base_nodes, _objects


def load_document_nodes_llamaparse(
    _json_objs,
    _parse_method,
    ):
    # Only load and parse document if either index or docstore not saved.
    _document = load_document_llamaparse_jason(
                                        _json_objs
                                        )
    (_base_nodes,
     _object) = get_nodes_from_document_llamaparse(
                                    _document, 
                                    _parse_method
                                    )
    return _base_nodes, _object


def load_image_text_nodes_llamaparse(_json_objs, _parser, _article_dir, _article_name):
    """
    Extract images from PDF and generate descriptions using Claude Vision API.
    
    Uses raw Anthropic SDK instead of AnthropicMultiModal to avoid package 
    version conflicts with llama-index-llms-anthropic>=0.10.
    
    Caches image descriptions to avoid repeated expensive API calls.
    
    Args:
        _json_objs: JSON objects from LlamaParse containing document structure
        _parser: LlamaParse parser instance for extracting images
        _article_dir: Directory containing the article (e.g., "Rag_anything")
        _article_name: Name of the article file (e.g., "RAG_Anything.pdf")
    
    Returns:
        List of TextNode objects containing image descriptions
    """
    # Initialize Anthropic client
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    
    # Extract images from PDF
    image_dicts = _parser.get_images(_json_objs, download_path="./images/")
    
    if not image_dicts:
        print("📷 No images found in document.")
        return []
    
    # Cache file in same directory as JSON cache, with article name
    cache_file = Path(f"./data/{_article_dir}/{_article_name.replace('.pdf', '_image_descriptions.json')}")
    
    # Load existing cache
    descriptions_cache = {}
    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                descriptions_cache = json.load(f)
            print(f"📂 Loaded {len(descriptions_cache)} cached image descriptions")
        except:
            descriptions_cache = {}
    
    print(f"📷 Processing {len(image_dicts)} images with Claude Vision...")
    
    img_text_nodes = []
    new_descriptions = 0
    
    for i, image_dict in enumerate(image_dicts):
        image_path = image_dict["path"]
        image_name = os.path.basename(image_path)
        print(f"   Processing image {i+1}/{len(image_dicts)}: {image_name}")
        
        # Check cache first
        if image_name in descriptions_cache:
            print(f"      ✅ Using cached description")
            description = descriptions_cache[image_name]["description"]
            text_node = TextNode(
                text=description,
                metadata={
                    "path": image_path,
                    "image_name": image_name,
                    "page_number": image_dict.get("page_number", "unknown"),
                    "type": "image_description"
                }
            )
            img_text_nodes.append(text_node)
            continue
        
        try:
            # Load image and check dimensions
            with Image.open(image_path) as img:
                width, height = img.size
                max_dimension = 8000  # Claude's max allowed pixel dimension
                
                # Resize if any dimension exceeds max
                if width > max_dimension or height > max_dimension:
                    # Calculate scale factor to fit within max dimensions
                    scale = min(max_dimension / width, max_dimension / height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    print(f"      ⚠️ Resizing from {width}x{height} to {new_width}x{new_height}")
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Convert to bytes for base64 encoding
                buffer = BytesIO()
                # Determine format from extension
                ext = os.path.splitext(image_path)[1].lower()
                img_format = "PNG" if ext == ".png" else "JPEG"
                img.save(buffer, format=img_format)
                image_data = base64.standard_b64encode(buffer.getvalue()).decode("utf-8")
            
            # Determine media type from file extension
            media_type_map = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif",
                ".webp": "image/webp"
            }
            media_type = media_type_map.get(ext, "image/png")
            
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
                                "text": "Describe this image in detail. Include all visual elements, labels, arrows, diagrams, charts, or any text visible in the image. Be specific about the layout and relationships between elements."
                            }
                        ],
                    }
                ],
            )
            
            # Extract response text
            description = message.content[0].text
            
            # Cache the description
            descriptions_cache[image_name] = {
                "description": description,
                "page_number": image_dict.get("page_number", "unknown")
            }
            new_descriptions += 1
            
            # Create TextNode with image description
            text_node = TextNode(
                text=description,
                metadata={
                    "path": image_path,
                    "image_name": image_name,
                    "page_number": image_dict.get("page_number", "unknown"),
                    "type": "image_description"
                }
            )
            img_text_nodes.append(text_node)
            
        except Exception as e:
            print(f"   ⚠️ Error processing {image_path}: {e}")
            continue
    
    # Save cache if we have new descriptions
    if new_descriptions > 0:
        with open(cache_file, "w") as f:
            json.dump(descriptions_cache, f, indent=2)
        print(f"💾 Saved {new_descriptions} new descriptions to cache")
    
    print(f"✅ Generated descriptions for {len(img_text_nodes)} images ({new_descriptions} new, {len(img_text_nodes) - new_descriptions} cached).")
    return img_text_nodes


def create_and_save_llamaparse_vector_index_to_milvus_database(_base_nodes,
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
    print(f"   🔄 Converted {len(converted_objects)} IndexNodes to TextNodes (stripped .obj attribute)")

    # Also clear large metadata from image text nodes (keep only essential fields)
    for img_node in deepcopy_image_text_nodes:
        # Keep only essential metadata, remove large OCR data if present
        img_node.metadata = {
            "type": img_node.metadata.get("type", "image_description"),
            "image_name": img_node.metadata.get("image_name", ""),
            "page_number": img_node.metadata.get("page_number", "unknown")
        }

    # In llamaoparse for Uber 2022, save 274 nodes to vector store (200 text nodes, 
    # 74 index nodes),and also save to MongoDB with 74 index nodes. The removal of
    # metadata does not seem to impact the details of the 74 index nodes in MongoDB.

    # Split any oversized nodes to fit within embedding model's token limit (8192 for text-embedding-3-small)
    # Using chunk_size=2000 tokens with overlap to stay safely under the limit
    # Note: Some nodes may have very large text from OCR/table data
    text_splitter = SentenceSplitter(chunk_size=2000, chunk_overlap=100)
    
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
        # Using very conservative threshold: 8000 chars ≈ 2000 tokens
        # This ensures we catch all oversized nodes
        if len(node.text) > 8000:
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

    def _ensure_engine(self):
        if self._engine is None:
            self._engine = self._factory()

    def query(self, *args, **kwargs):
        self._ensure_engine()
        return self._engine.query(*args, **kwargs)

    async def aquery(self, *args, **kwargs):
        self._ensure_engine()
        if hasattr(self._engine, "aquery"):
            return await self._engine.aquery(*args, **kwargs)
        raise NotImplementedError("Underlying query engine does not support async queries.")

    def __getattr__(self, item):
        self._ensure_engine()
        return getattr(self._engine, item)


import re
from typing import Dict, List, Optional


def build_section_index(json_objs) -> Dict[str, Dict]:
    """
    Parse LlamaParse JSON to extract section headings and their page ranges.
    
    This function builds an index that maps section identifiers to their page ranges.
    Sections are indexed by multiple keys for flexible lookup:
    - Section number (e.g., "1", "2.1", "A.1")
    - Section title in lowercase (e.g., "introduction", "methodology")
    - Full identifier (e.g., "1 introduction")
    
    Args:
        json_objs: The JSON objects returned by LlamaParse
        
    Returns:
        Dict mapping section identifiers to their metadata:
        {
            'introduction': {'start_page': 1, 'end_page': 2, 'section_num': '1', 'title': 'INTRODUCTION'},
            '1': {'start_page': 1, 'end_page': 2, 'section_num': '1', 'title': 'INTRODUCTION'},
            ...
        }
    """
    # Pattern to match numbered sections like '# 1  INTRODUCTION', '# 2.1 PRELIMINARY', '# A.1 ...'
    numbered_pattern = re.compile(r'^#\s+(\d+(?:\.\d+)*|[A-Z](?:\.\d+)*)\s+(.+)$')
    
    # First pass: collect all numbered section headings with their start pages
    section_headings = []
    for page in json_objs[0]['pages']:
        page_num = page['page']
        for item in page.get('items', []):
            if item.get('type') == 'heading':
                md = item.get('md', '').strip()
                match = numbered_pattern.match(md)
                if match:
                    section_num = match.group(1)
                    section_title = match.group(2).strip()
                    section_headings.append({
                        'start_page': page_num,
                        'section_num': section_num,
                        'title': section_title,
                        'level': section_num.count('.') + 1,  # 1 -> level 1, 1.1 -> level 2
                    })
    
    # Second pass: calculate end pages
    # A section ends when the next same-level or higher-level section starts, or at document end
    total_pages = len(json_objs[0]['pages'])
    
    for i, section in enumerate(section_headings):
        current_level = section['level']
        end_page = total_pages  # Default to last page
        
        # Look for the next section at same or higher level
        for j in range(i + 1, len(section_headings)):
            next_section = section_headings[j]
            if next_section['level'] <= current_level:
                # Next section at same or higher level - this section ends before it
                end_page = next_section['start_page'] - 1
                break
        
        # Ensure end_page is at least start_page
        section['end_page'] = max(section['start_page'], end_page)
    
    # Build the index with multiple keys for flexible lookup
    section_index = {}
    
    for section in section_headings:
        section_info = {
            'start_page': section['start_page'],
            'end_page': section['end_page'],
            'section_num': section['section_num'],
            'title': section['title'],
            'level': section['level'],
        }
        
        # Index by section number (e.g., "1", "2.1")
        section_index[section['section_num'].lower()] = section_info
        
        # Index by title in lowercase (e.g., "introduction")
        title_key = section['title'].lower()
        # Only add if not already present (avoid overwriting by subsections)
        if title_key not in section_index:
            section_index[title_key] = section_info
        
        # Index by full identifier (e.g., "1 introduction")
        full_key = f"{section['section_num']} {section['title']}".lower()
        section_index[full_key] = section_info
    
    # Also add common aliases
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


def create_custom_guidance_prompt() -> str:
    """
    Create a custom prompt template for GuidanceQuestionGenerator.
    
    This function constructs a prompt template that guides the question generation
    process for sub-question decomposition. The template includes:
    - A prefix explaining the task
    - Example 1: Financial comparison (default LlamaIndex example)
    - Example 2: Page-based content summarization
    - A suffix for the actual query
    
    Returns:
    str: A Handlebars-formatted prompt template string for use with GuidanceQuestionGenerator.
    """
    # Write in Python format string style, then convert to Handlebars
    PREFIX = """\
    Given a user question, and a list of tools, output a list of relevant sub-questions \
    in json markdown that when composed can help answer the full user question:

    """

    # Default example from LlamaIndex
    EXAMPLE_1 = """\
    # Example 1
    <Tools>
    ```json
    [
    {{
        "name": "uber_10k",
        "description": "Provides information about Uber financials for year 2021"
    }},
    {{
        "name": "lyft_10k",
        "description": "Provides information about Lyft financials for year 2021"
    }}
    ]
    ```

    <User Question>
    Compare and contrast the revenue growth and EBITDA of Uber and Lyft for year 2021

    <Output>
    ```json
    {{
    "items": [
        {{"sub_question": "What is the revenue growth of Uber", "tool_name": "uber_10k"}},
        {{"sub_question": "What is the EBITDA of Uber", "tool_name": "uber_10k"}},
        {{"sub_question": "What is the revenue growth of Lyft", "tool_name": "lyft_10k"}},
        {{"sub_question": "What is the EBITDA of Lyft", "tool_name": "lyft_10k"}}
    ]
    }}
    ```

    """

    # Tailored example for page-based queries
    EXAMPLE_2 = """\
    # Example 2
    <Tools>
    ```json
    [
    {{
        "name": "page_filter_tool",
        "description": "Perform a query search over the page numbers mentioned in the query"
    }}
    ]
    ```

    <User Question>
    Summarize the content from pages 20 to 22 in the voice of the author by NOT retrieving the text verbatim

    <Output>
    ```json
    {{
    "items": [
        {{"sub_question": "Summarize the content on page 20 in the voice of the author by NOT retrieving the text verbatim", "tool_name": "page_filter_tool"}},
        {{"sub_question": "Summarize the content on page 21 in the voice of the author by NOT retrieving the text verbatim", "tool_name": "page_filter_tool"}},
        {{"sub_question": "Summarize the content on page 22 in the voice of the author by NOT retrieving the text verbatim", "tool_name": "page_filter_tool"}}
    ]
    }}
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

    # Combine and convert to Handlebars format
    custom_guidance_prompt = convert_to_handlebars(PREFIX + EXAMPLE_1 + EXAMPLE_2 + SUFFIX)
    
    return custom_guidance_prompt


def get_fusion_tree_page_filter_sort_detail_tool_llamaparse(
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
    
    NOTE: This is adapted for LlamaParse which uses 'page' metadata key (integer)
    instead of 'source' (string) used in PyMuPDF/langextract_simple.py.
    
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
        
        prompt = PromptTemplate(query_text)
        _result_str = _llm.predict(prompt=prompt, query_str=_query_str, section_list=section_list_str)
        
        try:
            _result = json.loads(_result_str)
            
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
    # NOTE: LlamaParse uses 'page' key with integer values, not 'source' with strings
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
    tool = get_fusion_tree_page_filter_sort_detail_tool_llamaparse(
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
LLAMA_CLOUD_API_KEY = os.environ['LLAMA_CLOUD_API_KEY']
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
    model_name="text-embedding-3-small",
    embed_batch_size=10  # Reduce batch size to avoid hitting 300k token limit
)
Settings.embed_model = embed_model
embed_model_dim = 1536  # for text-embedding-3-small
embed_model_name = "openai_embedding_3_small"

# Create article link
# article_dictory = "uber"
# article_name = "uber_10q_march_2022.pdf"

# article_dictory = "attention"
# article_name = "attention_all.pdf"

article_dictory = "Rag_anything"
article_name = "RAG_Anything.pdf"

article_link = get_article_link(article_dictory,
                                article_name
                                )

# Create database and collection names
chunk_method = "llamaparse"
parse_method = "jason"

# parse_method = "markdown"  # DOES NOT WORK

# Create database name and colleciton names
(database_name, 
collection_name_vector,
collection_name_summary) = get_database_and_llamaparse_collection_name(
                                                            article_dictory, 
                                                            chunk_method, 
                                                            embed_model_name, 
                                                            parse_method,
                                                            )

# Initiate Milvus and MongoDB database
uri_milvus = "http://localhost:19530"
uri_mongo = "mongodb://localhost:27017/"

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
json_cache_path = Path(f"./data/{article_dictory}/{article_name.replace('.pdf', '_llamaparse.json')}")
section_index = None  # Initialize section index

if json_cache_path.exists():
    print(f"\n📂 Loading cached JSON from {json_cache_path} for section index...")
    with open(json_cache_path, "r") as f:
        json_objs_for_index = json.load(f)
    # Build section index from the JSON
    section_index = build_section_index(json_objs_for_index)
    print(f"\n📑 Built section index with {len(set(id(v) for v in section_index.values()))} unique sections")
    # Print available sections for debugging
    # print("   Available sections:")
    # print(format_section_list_for_prompt(section_index))
else:
    print(f"⚠️ JSON cache not found at {json_cache_path}, section index not available")

# Load documnet nodes if either vector index or docstore not saved.
if save_index_vector or add_document_vector or add_document_summary: 

    # parsing_instruction = "Keep section number, sub-section number, and equation number as refeences in the output."
    # 1. A new section starts with an integer (the section number) followed by spaces and the 
    # section title in a line of its own.
    # 2. A subsection starts with a subsection number (e.g., 1.1) followed by spaces and the 
    # subsection title in a line of its own.
    # 3. An equation occupies a line of its own and is centered in the line and with the 
    # equation number in the right margin in a pair of parentheses (e.g., (1)).
    # Aggressive parsing instruction for better equation recognition
    # This helps LlamaParse correctly handle subscripts/superscripts that appear on separate lines in PDFs
    parsing_instruction_equations = """
    This is an academic paper containing mathematical equations with complex notation.

    === LATEX DELIMITER REQUIREMENTS ===
    1. Display equations (standalone, centered): Use $$ ... $$ delimiters
    2. Inline math (within text): Use $ ... $ delimiters (NOT \\( \\) notation)
    3. Equation numbers in the right margin MUST be preserved using \\tag{n} notation

    === CRITICAL: COMPLEX MATH SYMBOL HANDLING ===
    When parsing equations from this PDF, pay careful attention to these special notations:

    TILDE/ACCENT MARKS:
    - A variable with a tilde above it (like Ẽ) must be written as \\tilde{E}
    - Do NOT drop the tilde - it indicates a different variable

    BIG OPERATORS WITH LIMITS:
    - Large union symbol with subscript j underneath: \\bigcup_j (NOT just \\cup)
    - Large intersection with subscript: \\bigcap_j
    - Large summation: \\sum_j

    CALLIGRAPHIC/SCRIPT LETTERS:
    - Fancy/script letters (like a curly E or V) must use \\mathcal{}: 
    - Script E → \\mathcal{E}
    - Script V → \\mathcal{V}
    - These are DIFFERENT from regular E and V

    LABELED ARROWS:
    - Arrows with text labels above them: \\xrightarrow{\\text{label}}
    - Example: an arrow labeled "belongs_to" → \\xrightarrow{\\text{belongs\\_to}}

    SUBSCRIPTS AND SUPERSCRIPTS:
    - Subscripts: V_j, E_j, d_j
    - Superscripts: v^{mm} or v^{\\text{mm}}
    - Combined: v_j^{\\text{mm}} means v with subscript j AND superscript mm
    - In PDFs, subscripts/superscripts may appear on SEPARATE LINES - combine them correctly

    === EXAMPLE EQUATION ===
    If you see an equation that looks like:
    "E-tilde equals big-union of script-E sub j, union big-union of (u arrow-with-belongs_to v sub j superscript mm)"

    It should be parsed as:
    $$\\tilde{E} = \\bigcup_j \\mathcal{E}_j \\cup \\bigcup_j \\{(u \\xrightarrow{\\text{belongs\\_to}} v_j^{\\text{mm}}) : u \\in \\mathcal{V}_j\\} \\tag{4}$$

    === VISUAL CONTEXT ===
    Look at the VISUAL appearance in the PDF to correctly identify:
    - Tildes and other accent marks above variables
    - Big vs small operators (large ∪ vs small ∪)
    - Script/calligraphic vs regular letters
    - Text labels on arrows
    - Disconnected subscripts/superscripts that belong to nearby variables
    """
    
    if parse_method == "jason":
        parser = LlamaParse(
                    api_key=LLAMA_CLOUD_API_KEY, 
                    parsing_instruction=parsing_instruction_equations,
                    # Premium mode for highest quality parsing (uses LlamaParse's best models)
                    # Note: gpt4o_mode with external API key requires parse_mode="parse_page_with_lvm"
                    # which changes the output format. Using premium_mode instead for best results.
                    premium_mode=True,
                    # Force re-parse to use new settings
                    invalidate_cache=False,
                    verbose=True,
                    )
        
        # Cache JSON to avoid repeated API calls
        json_cache_path = Path(f"./data/{article_dictory}/{article_name.replace('.pdf', '_llamaparse.json')}")
        
        if json_cache_path.exists():
            print(f"\n📂 Loading cached JSON from {json_cache_path}...")
            with open(json_cache_path, "r") as f:
                json_objs = json.load(f)
        else:
            print(f"🌐 Fetching from LlamaParse API...")
            json_objs = parser.get_json_result(article_link)
            # Save to cache
            with open(json_cache_path, "w") as f:
                json.dump(json_objs, f)
            print(f"💾 Saved JSON cache to {json_cache_path}")

        image_text_nodes = load_image_text_nodes_llamaparse(json_objs, parser, article_dictory, article_name)

        (base_nodes,
        objects) = load_document_nodes_llamaparse(
                                                json_objs,
                                                parse_method,
                                                )
    elif parse_method == "markdown":  # DOES NOT WORK
        parser = LlamaParse(
                    api_key=LLAMA_CLOUD_API_KEY, 
                    result_type="markdown",
                    verbose=True,
                    )
        docs = parser.load_data(article_link)
        node_parser = MarkdownElementNodeParser(
                                        num_workers=8,
                                        include_metadata=True,
                                        )

        raw_nodes = node_parser.get_nodes_from_documents(docs)
        (base_nodes,
        objects) = node_parser.get_nodes_and_objects(raw_nodes)
    else:
        print("parse_method is not defined.")

if save_index_vector == True:
    recursive_index = create_and_save_llamaparse_vector_index_to_milvus_database(
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
    print(f"✅ Stitched prev/next relationships for {len(all_nodes_vector)} nodes")

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
similarity_top_k_fusion = 48  # Number of similar nodes to retrieve for fusion
fusion_top_n = 42  # Number of nodes to return from fusion (before reranking)
num_queries_fusion = 1  # Set to 1 to disable query generation (use original query)
rerank_top_n = 32  # Number of nodes after ColBERT reranking
num_nodes_prev_next = 1  # Number of neighboring nodes to retrieve (0 = disabled)

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
# query = "Please summarize the content in Section 2.1.1 MOTIVATING RAG-ANYTHING"
# query = "Describe the content of Section 2.3 CROSS-MODAL HYBRID RETRIEVAL"
# query = "What is the content of the Evaluation section?"
# query = "Summarize the content of the Evaluation section."
# query = "Summarize the Conclusion section."
# query = "What is in equation (1)?"
# query = "What are in equation (3) and (4)?"
# query = "What are in equation (4)?"
# query = "What are in the equations (1), (2), (3), and (4)? What are they trying to represent collectively?"
query = "What are in the equations (2) and (3)?"
# query = "How graphs are used in RAG-Anything's retrieval process as described in the paper?"

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
page_filter_verbose = True  # Set to True for debugging page filter

# Use LazyQueryEngine to defer initialization until first use
lazy_page_filter_engine = LazyQueryEngine(build_page_filter_query_engine)

page_filter_tool = QueryEngineTool.from_defaults(
    name="page_filter_tool",
    query_engine=lazy_page_filter_engine,
    description=page_tool_description,
)

# Create custom guidance prompt for SubQuestionQueryEngine
CUSTOM_GUIDANCE_PROMPT = create_custom_guidance_prompt()
question_gen = GuidanceQuestionGenerator.from_defaults(
                            prompt_template_str=CUSTOM_GUIDANCE_PROMPT,
                            guidance_llm=GuidanceOpenAI(
                                model="gpt-4o",
                                api_key=OPENAI_API_KEY,
                                echo=False),
                            verbose=True
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
        "Be explicit and avoid omitting technical specifics.\n"
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

