import json
import asyncio
import tiktoken
from typing import List, Dict, Any, Optional

from llama_index.core import (
    PromptTemplate,
    VectorStoreIndex,
)
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.question_gen import LLMQuestionGenerator
from llama_index.core.response_synthesizers.factory import get_response_synthesizer
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.tools import QueryEngineTool
from llama_index.core.vector_stores import MetadataFilters
from llama_index.core.schema import Document
from llama_index.postprocessor.colbert_rerank import ColbertRerank
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.storage.docstore.mongodb import MongoDocumentStore

from langextract_integration import extract_query_metadata_filters
from utils import (
    get_fusion_tree_page_filter_sort_detail_engine,
    get_fusion_tree_keyphrase_filter_sort_detail_engine,
    get_text_nodes_from_query_keyphrase,
)


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
        "Be explicit and avoid omitting technical specifics.\n\n"
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
        
        # Deterministic section match if index exists
        if section_index:
            q_lower = query_str.lower()
            sorted_keys = sorted(section_index.keys(), key=len, reverse=True)
            for key in sorted_keys:
                if len(key) >= 3 and key in q_lower:
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
                    section_key = result.get('section', '').lower().strip()
                    info = section_index.get(section_key)
                    if info:
                        page_numbers = list(range(info['start_page'], info['end_page'] + 1))
                    else:
                        page_numbers = [1]
                else:
                    page_numbers = result.get('pages', [1])
            except:
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
    llm=None
) -> Dict[str, List[str]]:
    """Extract named entities from a query string using EntityExtractor."""
    from llama_index.extractors.entity import EntityExtractor
    
    entity_extractor = EntityExtractor(
        model_name="lxyuan/span-marker-bert-base-multilingual-cased-multinerd",
        prediction_threshold=0.5,
        label_entities=True,
        device="mps",
    )
    
    temp_doc = Document(text=query_str)
    try:
        from llama_index.core.node_parser import SentenceSplitter
        node_parser = SentenceSplitter(chunk_size=512)
        nodes = node_parser.get_nodes_from_documents([temp_doc])
        processed_nodes = entity_extractor.process_nodes(nodes)
        
        entities = {
            'PER': [], 'ORG': [], 'LOC': [], 'ANIM': [], 'BIO': [], 'CEL': [], 
            'DIS': [], 'EVE': [], 'FOOD': [], 'INST': [], 'MEDIA': [], 
            'PLANT': [], 'MYTH': [], 'TIME': [], 'VEHI': []
        }
        
        key_mappings = {
            'PER': ['PER', 'persons'], 'ORG': ['ORG', 'organizations'], 'LOC': ['LOC', 'locations'],
            'ANIM': ['ANIM', 'animals'], 'BIO': ['BIO', 'biologicals'], 'CEL': ['CEL', 'celestial_bodies'],
            'DIS': ['DIS', 'diseases'], 'EVE': ['EVE', 'events'], 'FOOD': ['FOOD', 'foods'],
            'INST': ['INST', 'instruments'], 'MEDIA': ['MEDIA', 'media'], 'PLANT': ['PLANT', 'plants'],
            'MYTH': ['MYTH', 'myths'], 'TIME': ['TIME', 'times'], 'VEHI': ['VEHI', 'vehicles'],
        }
        
        for node in processed_nodes:
            for standard_key, possible_keys in key_mappings.items():
                for entity_type in possible_keys:
                    if entity_type in node.metadata and node.metadata[entity_type]:
                        for entity in node.metadata[entity_type]:
                            if entity not in entities[standard_key]:
                                entities[standard_key].append(entity)
        
        entities = {k: v for k, v in entities.items() if v}
        if entities:
            print(f"\n📌 Entities extracted from query:")
            for entity_type, entity_list in entities.items():
                print(f"   {entity_type}: {entity_list}")
        return entities
    except Exception as e:
        print(f"Warning: Entity extraction failed: {e}")
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
    def __init__(self, vector_index, vector_docstore, rag_settings, reranker, metadata_option):
        from llama_index.core.callbacks import CallbackManager
        self.vector_index = vector_index
        self.vector_docstore = vector_docstore
        self.rag_settings = rag_settings
        self.reranker = reranker
        self.metadata_option = metadata_option
        self.callback_manager = CallbackManager([])
        
    def _build_engine_for_query(self, query_str: str):
        top_k = self.rag_settings["similarity_top_k_fusion"]
        extracted_entities = extract_entities_from_query(query_str)
        entity_filters = create_entity_metadata_filters(extracted_entities, self.metadata_option, query_str)
        
        if entity_filters:
            print(f"\n🔍 Created {len(entity_filters.filters)} LangExtract filters for this sub-question")
        
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
) -> QueryEngineTool:
    """Unified builder for the keyphrase fusion tool."""
    
    def build_engine():
        if enable_entity_filtering and metadata_option in ["entity", "langextract", "both"]:
            print(f"\n🔄 Dynamic Entity Filtering: Enabled for {metadata_option}")
            return DynamicFilterQueryEngine(
                vector_index, vector_docstore, rag_settings, reranker, metadata_option
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
    
    return SubQuestionQueryEngine.from_defaults(
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
