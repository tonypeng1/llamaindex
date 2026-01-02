import re
import json
import asyncio
import tiktoken
from typing import List, Dict, Any, Optional

# Normalization utilities for matching queries and references
NUM_MAP = {
    'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
    'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
    'ten': '10', 'eleven': '11', 'twelve': '12'
}


def normalize_for_matching(q: str) -> str:
    """Normalize query text for deterministic matching and reference detection."""
    q = q.lower()
    # Expand abbreviations
    q = re.sub(r"\bsec\.?\b", "section", q)
    q = re.sub(r"\bfig\.?\b", "figure", q)
    q = re.sub(r"\beq\.?\b", "equation", q)
    # Map number words
    for word, digit in NUM_MAP.items():
        q = re.sub(rf"\b{word}\b", digit, q)
    # Remove punctuation that interferes
    q = re.sub(r"[\.,;:\(\)\[\]']", " ", q)
    # Collapse spaces
    q = re.sub(r"\s+", " ", q).strip()
    return q


def _find_pages_for_reference(kind: str, num: str, vector_docstore) -> List[int]:
    """
    Search the docstore nodes to find pages containing a given figure or equation number.
    kind: 'figure' or 'equation'
    """
    pages = set()
    num = str(num)
    for _, node in getattr(vector_docstore, 'docs', {}).items():
        txt = getattr(node, 'text', '') or ''
        txt_l = txt.lower()
        # Figure detection (support decimal labels like 4.1)
        if kind == 'figure':
            # 1) Check explicit metadata first
            fig_meta = node.metadata.get('figure_label') if hasattr(node, 'metadata') else None
            if fig_meta:
                # normalized compare (e.g., '4' == '4' or '4.1' == '4.1')
                if str(fig_meta) == str(num):
                    pages.add(node.metadata.get('page'))
                    continue
            # 2) Fallback to text search (handles 'figure 4.1', 'fig. 4.1')
            if re.search(rf'figure\s*[:\s]*{re.escape(str(num))}\b', txt_l) or re.search(rf'fig\.?\s*{re.escape(str(num))}\b', txt_l):
                pages.add(node.metadata.get('page'))
        elif kind == 'equation':
            # direct eq markers
            if re.search(rf'equation\s*[:\s]*{re.escape(str(num))}\b', txt_l) or re.search(rf'eq\.?\s*{re.escape(str(num))}\b', txt_l):
                pages.add(node.metadata.get('page'))
            else:
                # parenthesis match but only if 'equation' nearby or parentheses at line start
                if re.search(rf'\(\s*{re.escape(str(num))}\s*\)', txt_l):
                    # accept if 'equation' in text or parentheses appears at start
                    if 'equation' in txt_l or txt_l.strip().startswith(f'({num})'):
                        pages.add(node.metadata.get('page'))
    return sorted(pages)

from llama_index.core import (
    PromptTemplate,
    VectorStoreIndex,
    QueryBundle,
)
from llama_index.core.callbacks import CallbackManager
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.question_gen import LLMQuestionGenerator
from llama_index.core.response_synthesizers.factory import get_response_synthesizer
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.tools import QueryEngineTool
from llama_index.core.vector_stores import MetadataFilters
from llama_index.core.schema import Document, TextNode, NodeWithScore
from llama_index.postprocessor.colbert_rerank import ColbertRerank
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.storage.docstore.mongodb import MongoDocumentStore

from langextract_integration import extract_query_metadata_filters
from gliner_extractor import GLiNERExtractor
from extraction_schemas import get_gliner_entity_labels
from utils import (
    get_fusion_tree_page_filter_sort_detail_engine,
    get_fusion_tree_keyphrase_filter_sort_detail_engine,
    get_text_nodes_from_query_keyphrase,
    PageSortNodePostprocessor,
)


class SortedResponseSynthesizer:
    """
    A wrapper for response synthesizers that deduplicates and sorts nodes 
    by page number before synthesis.
    """
    def __init__(self, base_synthesizer):
        self._base = base_synthesizer
    
    def synthesize(self, query, nodes, additional_source_nodes=None, **kwargs):
        if additional_source_nodes:
            # Deduplicate by node_id
            unique_nodes_dict = {}
            for n in additional_source_nodes:
                unique_nodes_dict[n.node.node_id] = n
            unique_nodes = list(unique_nodes_dict.values())
            
            # Sort by page number
            postprocessor = PageSortNodePostprocessor()
            additional_source_nodes = postprocessor.postprocess_nodes(unique_nodes, query)
            
        return self._base.synthesize(query, nodes, additional_source_nodes=additional_source_nodes, **kwargs)

    async def asynthesize(self, query, nodes, additional_source_nodes=None, **kwargs):
        if additional_source_nodes:
            # Deduplicate by node_id
            unique_nodes_dict = {}
            for n in additional_source_nodes:
                unique_nodes_dict[n.node.node_id] = n
            unique_nodes = list(unique_nodes_dict.values())
            
            # Sort by page number
            postprocessor = PageSortNodePostprocessor()
            additional_source_nodes = postprocessor.postprocess_nodes(unique_nodes, query)
            
        return await self._base.asynthesize(query, nodes, additional_source_nodes=additional_source_nodes, **kwargs)

    def __getattr__(self, name):
        return getattr(self._base, name)


class SortedSubQuestionQueryEngine(SubQuestionQueryEngine):
    """
    A SubQuestionQueryEngine that ensures nodes are deduplicated and sorted 
    by page number before the final response synthesis.
    """
    def _query(self, query_bundle: QueryBundle):
        original_synth = self._response_synthesizer
        self._response_synthesizer = SortedResponseSynthesizer(original_synth)
        try:
            return super()._query(query_bundle)
        finally:
            self._response_synthesizer = original_synth

    async def _aquery(self, query_bundle: QueryBundle):
        original_synth = self._response_synthesizer
        self._response_synthesizer = SortedResponseSynthesizer(original_synth)
        try:
            return await super()._aquery(query_bundle)
        finally:
            self._response_synthesizer = original_synth


class LazyQueryEngine:
    """Instantiate the underlying query engine only when first queried.
    
    This wrapper defers tool creation until the sub-question engine actually calls
    the query, preventing eager initialization (and its logging) when
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

def get_custom_sub_question_prompt() -> str:
    """Returns the standardized sub-question decomposition prompt."""
    PREFIX = """\
    Given a user question, and a list of tools, output a list of relevant sub-questions \
    in json markdown that when composed can help answer the full user question.
    The output MUST be a valid JSON object with an "items" key.

    IMPORTANT: Break down the user question into multiple atomic sub-questions if it contains \
    multiple parts, requests information about multiple entities, or requires multiple steps to answer. \
    Each sub-question should focus on a single aspect of the original query. Even if all sub-questions \
    use the same tool, they should be separated to ensure thorough retrieval.
    """

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
    return PREFIX + EXAMPLE_1 + EXAMPLE_2 + SUFFIX

def get_detailed_response_synthesizer(response_mode: str = "TREE_SUMMARIZE"):
    """Returns a response synthesizer with a detailed technical prompt."""
    detailed_tree_tmpl = (
        "Context information from multiple sources is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "You are an expert assistant. Provide a detailed, structured, and thorough answer "
        "to the query below. Include key points, important details, any equations or "
        "examples present in the context, and list steps or components when applicable. "
        "Be explicit and include technical specifics such as figure numbers, table numbers, or equation numbers when applicable.\n\n"
        "=== MATHEMATICAL FORMULAS ===\n"
        "For any mathematical equations or formulas in your response:\n"
        "1. Use $$ ... $$ delimiters for standalone/display equations (centered on their own line).\n"
        "2. Use $ ... $ delimiters for inline math (within a sentence).\n\n"
        "Query: {query_str}\n"
        "Detailed Answer: "
    )
    
    summary_template = None
    if response_mode == "TREE_SUMMARIZE":
        summary_template = PromptTemplate(detailed_tree_tmpl, prompt_type=PromptType.SUMMARY)

    return get_response_synthesizer(
        response_mode=ResponseMode[response_mode],
        summary_template=summary_template,
    )

def get_page_filter_tool(
    query_str: str,
    reranker: ColbertRerank,
    vector_index: VectorStoreIndex,
    vector_docstore: MongoDocumentStore,
    llm: Any,
    section_index: Optional[Dict[str, Dict]] = None,
    metadata_key: str = "page",
    verbose: bool = False,
) -> QueryEngineTool:
    """Unified builder for the page/section filter tool."""
    
    def build_engine():
        # 1. Resolve Page Numbers
        page_numbers = []
        
        # Deterministic and reference-based matching if index exists
        if section_index:
            # Normalize query for matching
            q_norm = normalize_for_matching(query_str)

            # 1) Figure reference detection (e.g., 'Fig. 2', 'figure 2')
            m_fig = re.search(r"\bfig(?:ure)?\s*(\d+(?:\.\d+)*)\b", q_norm)
            if m_fig:
                num = m_fig.group(1)
                page_numbers = _find_pages_for_reference('figure', num, vector_docstore)
                fig_num = num
                if page_numbers and verbose:
                    print(f"Deterministic figure match for 'figure {num}' -> pages {page_numbers}")
            else:
                fig_num = None

            # 2) Equation reference detection (e.g., 'Eq. 3', 'equation 3', or '(3)' with 'equation' context)
            if not page_numbers:
                m_eq = re.search(r"\bequation\s*(\d{1,3})\b", q_norm) or re.search(r"\b\( *([0-9]{1,3}) *\)\b", query_str)
                if m_eq:
                    num = m_eq.group(1)
                    page_numbers = _find_pages_for_reference('equation', num, vector_docstore)
                    if page_numbers and verbose:
                        print(f"Deterministic equation match for 'equation {num}' -> pages {page_numbers}")

            # 3) Deterministic section match (fallback) if still unknown
            if not page_numbers:
                sorted_keys = sorted(section_index.keys(), key=len, reverse=True)
                for key in sorted_keys:
                    key_lower = key.lower()
                    if ((len(key_lower) >= 3 or key_lower.isdigit() or key_lower.startswith('section ')) and key_lower in q_norm):
                        info = section_index[key]
                        page_numbers = list(range(info['start_page'], info['end_page'] + 1))
                        if verbose: print(f"Deterministic section match for '{key}' -> pages {page_numbers}")
                        break

        # LLM Fallback for page extraction
        if not page_numbers:
            section_list_str = ""
            if section_index:
                # Format section list for prompt
                seen = set()
                sections = []
                for key, info in section_index.items():
                    identifier = (info['section_num'], info['title'])
                    if identifier not in seen:
                        seen.add(identifier)
                        sections.append(info)
                sections.sort(key=lambda x: (x['start_page'], x['section_num'] or ""))
                section_list_str = "\n".join([f"  - Section {s['section_num']}: {s['title']} (pages {s['start_page']}-{s['end_page']})" for s in sections])

            prompt_text = (
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
                '## Now analyze the following query:\n'
                '**Query:** {query_str}\n'
            ) if section_index else (
                '## Instruction:\n'
                'Extract all page numbers from the user query. Return as a list of integers.\n'
                'If no page numbers are mentioned, output [1].\n'
                '**Query:** {query_str}\n'
            )

            prompt = PromptTemplate(prompt_text)
            result_str = llm.predict(prompt=prompt, query_str=query_str, section_list=section_list_str)
            
            try:
                result = json.loads(result_str)
                if isinstance(result, list):
                    page_numbers = result
                elif result.get('type') == 'section':
                    # Normalize and attempt robust lookups for section keys
                    def _normalize_section_key(k: str) -> str:
                        k = k.lower().strip()
                        k = re.sub(r"\bsec\.?\b", "section", k)
                        # Map simple number words to digits
                        num_map = {
                            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
                            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
                            'ten': '10'
                        }
                        for word, digit in num_map.items():
                            k = re.sub(rf"\b{word}\b", digit, k)
                        k = re.sub(r"[\.,;:\(\)\[\]']", " ", k)
                        return k.strip()

                    section_key_raw = result.get('section', '')
                    section_key = _normalize_section_key(section_key_raw)

                    # Try direct lookup
                    info = section_index.get(section_key)

                    # Try stripping leading 'section '
                    if not info and section_key.startswith('section '):
                        info = section_index.get(section_key[len('section '):])

                    # Try numeric extraction
                    if not info:
                        m = re.search(r"\b(\d{1,3})\b", section_key)
                        if m:
                            info = section_index.get(m.group(1)) or section_index.get(f"section {m.group(1)}")

                    # Try matching by title substring
                    if not info:
                        for k, v in section_index.items():
                            if k in section_key or section_key in k:
                                info = v
                                break

                    if info:
                        page_numbers = list(range(info['start_page'], info['end_page'] + 1))
                    else:
                        page_numbers = [1]
                else:
                    page_numbers = result.get('pages', [1])
            except Exception:
                page_numbers = [1]

        if verbose: print(f"Resolved page numbers: {page_numbers}")

        # 2. Retrieve Nodes
        text_nodes = []
        for _, node in vector_docstore.docs.items():
            val = node.metadata.get(metadata_key)
            if val is not None:
                # Handle both string and int metadata
                try:
                    if int(val) in [int(p) for p in page_numbers]:
                        text_nodes.append(node)
                except:
                    if str(val) in [str(p) for p in page_numbers]:
                        text_nodes.append(node)

        # If we detected a specific figure number (like 4.1), ensure image nodes with that figure_label are included
        if 'fig_num' in locals() and fig_num is not None:
            for _, node in vector_docstore.docs.items():
                fig_meta = node.metadata.get('figure_label')
                if fig_meta and str(fig_meta) == str(fig_num):
                    if node not in text_nodes:
                        text_nodes.append(node)

        # 3. Build Engine
        retriever = vector_index.as_retriever(
            similarity_top_k=len(vector_docstore.docs),
            filters=MetadataFilters.from_dicts([{"key": metadata_key, "value": page_numbers, "operator": "in"}])
        )
        
        return get_fusion_tree_page_filter_sort_detail_engine(
            retriever, len(text_nodes), text_nodes, 1, reranker, vector_docstore, [str(p) for p in page_numbers]
        )

    description = (
        "Perform a query search over specific pages or SECTIONS of the document. "
        "Use this function when user asks about specific PAGES or specific SECTIONS. "
        "Examples: 'What happened on page 1?', 'Summarize the Introduction section'. "
    )

    return QueryEngineTool.from_defaults(
        name="page_filter_tool",
        query_engine=LazyQueryEngine(build_engine),
        description=description,
    )

def extract_entities_from_query(
    query_str: str,
    llm=None,
    schema_name: str = "general"
) -> Dict[str, List[str]]:
    """Extract named entities from a query string using GLiNER."""
    
    # Use provided schema or fallback to general
    entity_labels = get_gliner_entity_labels(schema_name=schema_name)
    
    entity_extractor = GLiNERExtractor(
        model_name="urchade/gliner_medium-v2.1",
        entity_labels=entity_labels,
        threshold=0.5,
        device="mps",
    )
    
    node = TextNode(text=query_str)
    try:
        # GLiNERExtractor handles extraction and metadata update
        entity_extractor([node])
        
        # Collect entities from metadata (uppercase keys)
        entities = {k: v for k, v in node.metadata.items() if k.isupper()}
        
        if entities:
            print(f"📌 GLiNER Entities extracted from query:")
            for entity_type, entity_list in entities.items():
                print(f"   {entity_type}: {entity_list}")
        else:
            print(f"\n📌 No GLiNER entities extracted from query")
        return entities
    except Exception as e:
        print(f"Warning: GLiNER entity extraction failed: {e}")
        return {}

def create_entity_metadata_filters(
    entities: Dict[str, List[str]],
    metadata_option: str,
    query_str: Optional[str] = None,
) -> Optional[MetadataFilters]:
    """Create MetadataFilters for entity-based and semantic filtering."""
    filters = []
    if entities:
        key_mapping = {
            'PER': 'persons', 'ORG': 'organizations', 'LOC': 'locations',
            'ANIM': 'animals', 'BIO': 'biologicals', 'CEL': 'celestial_bodies',
            'DIS': 'diseases', 'EVE': 'events', 'FOOD': 'foods',
            'INST': 'instruments', 'MEDIA': 'media', 'PLANT': 'plants',
            'MYTH': 'myths', 'TIME': 'times', 'VEHI': 'vehicles',
        }
        
        if metadata_option in ["entity", "both"]:
            for entity_type, entity_list in entities.items():
                db_key = key_mapping.get(entity_type, entity_type)
                for entity in entity_list:
                    filters.append({"key": db_key, "value": entity, "operator": "=="})
                    # If the extracted entity looks like a figure reference (e.g., 'fig. 4.1'),
                    # add a filter on the image node's figure_label metadata so image descriptions
                    # can be selected directly when the query references a figure.
                    try:
                        m_fig = re.search(r"fig(?:ure)?\.?\s*(\d+(?:\.\d+)*)", entity, re.IGNORECASE)
                        if m_fig:
                            fig_val = m_fig.group(1)
                            filters.append({"key": "figure_label", "value": fig_val, "operator": "=="})
                    except Exception:
                        pass
        
        if metadata_option in ["langextract", "both"]:
            all_entities = [e for l in entities.values() for e in l]
            for entity in all_entities:
                filters.append({"key": "entity_names", "value": entity, "operator": "=="})

    if metadata_option in ["langextract", "both"] and query_str:
        try:
            semantic_filters = extract_query_metadata_filters(query_str)
            if semantic_filters:
                for key, values in semantic_filters.items():
                    for value in values:
                        filters.append({"key": key, "value": value, "operator": "=="})
        except Exception as e:
            print(f"Warning: Semantic filter extraction failed: {e}")
    
    return MetadataFilters.from_dicts(filters, condition="or") if filters else None

class DynamicFilterQueryEngine:
    """Query engine wrapper that extracts filters per sub-question."""
    def __init__(self, vector_index, vector_docstore, rag_settings, reranker, metadata_option, schema_name=None):
        self.vector_index = vector_index
        self.vector_docstore = vector_docstore
        self.rag_settings = rag_settings
        self.reranker = reranker
        self.metadata_option = metadata_option
        self.schema_name = schema_name
        self.callback_manager = CallbackManager([])
        
    def _build_engine_for_query(self, query_str: str):
        top_k = self.rag_settings["similarity_top_k_fusion"]
        extracted_entities = extract_entities_from_query(query_str, schema_name=self.schema_name)
        entity_filters = create_entity_metadata_filters(extracted_entities, self.metadata_option, query_str)
        
        if entity_filters:
            print(f"\n🔍 Dynamic filtering for query: '{query_str}'")
            for f in entity_filters.filters:
                print(f"   - {f.key} {f.operator} {f.value}")
        
        text_nodes = get_text_nodes_from_query_keyphrase(self.vector_docstore, top_k, query_str)
        bm25_retriever = BM25Retriever.from_defaults(similarity_top_k=top_k, nodes=text_nodes)
        vector_retriever = self.vector_index.as_retriever(similarity_top_k=top_k, filters=entity_filters)
        
        return get_fusion_tree_keyphrase_filter_sort_detail_engine(
            vector_retriever, self.vector_docstore, bm25_retriever,
            self.rag_settings["fusion_top_n"], self.rag_settings["num_queries"],
            self.reranker, self.rag_settings["num_nodes"]
        )
    
    def query(self, query_str: str):
        return self._build_engine_for_query(query_str).query(query_str)
    
    async def aquery(self, query_str: str):
        engine = self._build_engine_for_query(query_str)
        return await engine.aquery(query_str) if hasattr(engine, 'aquery') else engine.query(query_str)

def get_keyphrase_tool(
    query_str: str,
    vector_index: VectorStoreIndex,
    vector_docstore: MongoDocumentStore,
    reranker: ColbertRerank,
    llm: Any,
    rag_settings: Dict[str, Any],
    enable_entity_filtering: bool = False,
    metadata_option: Optional[str] = None,
    schema_name: str = "general",
) -> QueryEngineTool:
    """Unified builder for the keyphrase fusion tool."""
    
    def build_engine():
        if enable_entity_filtering and metadata_option in ["entity", "langextract", "both"]:
            print(f"\n🔄 Dynamic Entity Filtering: Enabled for {metadata_option}")
            return DynamicFilterQueryEngine(
                vector_index, vector_docstore, rag_settings, reranker, metadata_option, schema_name=schema_name
            )

        top_k = rag_settings["similarity_top_k_fusion"]
        keyphrase_nodes = get_text_nodes_from_query_keyphrase(vector_docstore, top_k, query_str)
        bm25_retriever = BM25Retriever.from_defaults(similarity_top_k=top_k, nodes=keyphrase_nodes)
        vector_retriever = vector_index.as_retriever(similarity_top_k=top_k)
        
        return get_fusion_tree_keyphrase_filter_sort_detail_engine(
            vector_retriever,
            vector_docstore,
            bm25_retriever,
            rag_settings["fusion_top_n"],
            rag_settings["num_queries"],
            reranker,
            rag_settings["num_nodes"],
        )

    description = (
        "Useful for retrieving specific, precise, or targeted content from the document. "
        "Use this tool for factual answers, specific details, or questions about "
        "EQUATIONS, FORMULAS, FIGURES, or TABLES."
    )

    return QueryEngineTool.from_defaults(
        name="keyphrase_tool",
        query_engine=LazyQueryEngine(build_engine),
        description=description,
    )

def build_sub_question_engine(
    tools: List[QueryEngineTool],
    llm: Any,
    response_mode: str = "TREE_SUMMARIZE",
    verbose: bool = True,
) -> SubQuestionQueryEngine:
    """Builds the final SubQuestionQueryEngine with retry logic and custom synthesizer."""
    
    question_gen = LLMQuestionGenerator.from_defaults(
        llm=llm,
        prompt_template_str=get_custom_sub_question_prompt()
    )
    
    synth = get_detailed_response_synthesizer(response_mode)
    
    return SortedSubQuestionQueryEngine.from_defaults(
        question_gen=question_gen,
        query_engine_tools=tools,
        response_synthesizer=synth,
        verbose=verbose,
    )

def print_response_diagnostics(response: Any):
    """Prints detailed information about the source nodes and token counts."""
    if response is None: return

    document_nodes = []
    for n in response.source_nodes:
        if bool(n.metadata):
            page_num = n.metadata.get('page', n.metadata.get('source', 'unknown'))
            document_nodes.append({
                'page': page_num,
                'text': n.text,
                'score': n.score,
                'node_id': n.node_id,
            })

    if document_nodes:
        enc = tiktoken.encoding_for_model("gpt-4")
        print("\n" + "="*80)
        print("SEQUENTIAL NODES WITH PAGE NUMBERS AND TOKEN COUNTS:")
        print("="*80)
        total_tokens = 0
        for i, node_info in enumerate(document_nodes, 1):
            node_id_prefix = node_info['node_id'][:8] if node_info.get('node_id') else 'UNKNOWN'
            node_tokens = len(enc.encode(node_info['text']))
            total_tokens += node_tokens
            print(f"  Node {i}: Page {node_info['page']} | {node_tokens:,} tokens | {len(node_info['text']):,} chars (ID: {node_id_prefix}..., Score: {round(node_info['score'], 3) if node_info['score'] else 'N/A'})")
        print(f"\n  📊 TOTAL: {len(document_nodes)} nodes, {total_tokens:,} tokens (context only)")
        print("="*80 + "\n")
