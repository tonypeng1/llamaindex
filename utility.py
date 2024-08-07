from typing import List, Optional

from llama_index.core import (
                        QueryBundle,
                        PromptTemplate,
                        )
from llama_index.core.query_engine import RetrieverQueryEngine

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore


class SortNodePostprocessor(BaseNodePostprocessor):
    def _postprocess_nodes(
            self, 
            nodes: List[NodeWithScore], 
            query_bundle: Optional[QueryBundle]
            ) -> List[NodeWithScore]:
        
        # Custom post-processor: Order nodes based on the order it appears in a document (using "start_char_idx")

        # Create new node dictionary
        _nodes_dic = [{"start_char_idx": node.node.start_char_idx, "node": node} for node in nodes]

        # Sort based on start_char_idx
        sorted_nodes_dic = sorted(_nodes_dic, key=lambda x: x["start_char_idx"])

        # Get the new nodes from the sorted node dic
        sorted_new_nodes = [node["node"] for node in sorted_nodes_dic]

        return sorted_new_nodes


def get_article_link(
        _article_dictory, 
        _article_name
        ):
    
    return "./data/" + _article_dictory + "/" + _article_name


def get_database_and_window_collection_name(
        _article_dictory, 
        _chunk_method, 
        _embed_model_name,
        _window_size
        ):
    
    _database_name = _article_dictory + "_" + _chunk_method
    _collection_name = _embed_model_name + "_window_size_" + str(_window_size)
    return _database_name, _collection_name


def get_database_and_sentence_splitter_collection_name(
        _article_dictory, 
        _chunk_method, 
        _embed_model_name,
        _chunk_size,
        _chunk_overlap,
        ):
    
    _database_name = _article_dictory + "_" + _chunk_method
    _collection_name = _embed_model_name + "_chunk_size_" + str(_chunk_size) + "_chunk_overlap_" + str(_chunk_overlap)
    return _database_name, _collection_name


def get_database_and_automerge_collection_name(
        _article_dictory, 
        _chunk_method, 
        _embed_model_name,
        chunk_sizes,
        ):
    
    _database_name = _article_dictory + "_" + _chunk_method
    _collection_name = _embed_model_name + "_size_" + str(chunk_sizes[0]) + "_" + \
                        str(chunk_sizes[1]) + "_" + str(chunk_sizes[2])
    
    return _database_name, _collection_name


def get_compact_tree_and_accumulate_engine_from_index(
        _index,
        _similarity_top_k,
        _postproc
        ):
    
    _compact_engine = _index.as_query_engine(
                                similarity_top_k=_similarity_top_k,
                                node_postprocessors=[_postproc],
                                response_mode="compact",
                                )
    _tree_engine = _index.as_query_engine(
                                similarity_top_k=_similarity_top_k,
                                node_postprocessors=[_postproc],
                                response_mode="tree_summarize",
                                )
    _accumulate_engine = _index.as_query_engine(
                                similarity_top_k=_similarity_top_k,
                                node_postprocessors=[_postproc],
                                response_mode="accumulate",
                                )
    return _compact_engine, _tree_engine, _accumulate_engine


def get_rerank_compact_tree_and_accumulate_engine_from_index(
        _index,
        _similarity_top_k,
        _postproc,
        _rerank
        ):
    
    _compact_rerank_engine = _index.as_query_engine(
                            similarity_top_k=_similarity_top_k,
                            node_postprocessors=[_postproc, _rerank],
                            response_mode="compact",
                            )

    _tree_rerank_engine = _index.as_query_engine(
                                similarity_top_k=_similarity_top_k,
                                node_postprocessors=[_postproc, _rerank],
                                response_mode="tree_summarize",
                                )

    _accumulate_rerank_engine = _index.as_query_engine(
                                similarity_top_k=_similarity_top_k,
                                node_postprocessors=[_postproc, _rerank],
                                response_mode="accumulate",
                                )
    return _compact_rerank_engine, _tree_rerank_engine, _accumulate_rerank_engine


def get_default_query_engine_from_retriever(
    retriever_1,
    retriever_2,
    ):

    query_engine_1 = RetrieverQueryEngine.from_args(
        retriever=retriever_1
        )
    query_engine_2 = RetrieverQueryEngine.from_args(
        retriever=retriever_2
        )

    return query_engine_1, query_engine_2


def get_tree_query_engine_from_retriever(
        retriever_1,
        retriever_2,
        ):

    query_engine_1 = RetrieverQueryEngine.from_args(
        retriever=retriever_1, 
        response_mode="tree_summarize",
        )

    query_engine_2 = RetrieverQueryEngine.from_args(
        retriever=retriever_2, 
        response_mode="tree_summarize",
        )

    return query_engine_1, query_engine_2


def get_tree_query_engine_with_sort_from_retriever(
        retriever_1,
        retriever_2,
        ):

    query_engine_1 = RetrieverQueryEngine.from_args(
        retriever=retriever_1, 
        response_mode="tree_summarize",
        node_postprocessors=[SortNodePostprocessor()],
        )

    query_engine_2 = RetrieverQueryEngine.from_args(
        retriever=retriever_2, 
        response_mode="tree_summarize",
        node_postprocessors=[SortNodePostprocessor()],
        )

    return query_engine_1, query_engine_2


def get_tree_engine_from_retriever(
        retriever_1,
        retriever_2,
        retriever_3,
        ):

    query_engine_1 = RetrieverQueryEngine.from_args(
        retriever=retriever_1, 
        response_mode="tree_summarize",
        )

    query_engine_2 = RetrieverQueryEngine.from_args(
        retriever=retriever_2, 
        response_mode="tree_summarize",
        )

    query_engine_3 = RetrieverQueryEngine.from_args(
        retriever=retriever_3, 
        response_mode="tree_summarize",
        )

    return query_engine_1, query_engine_2, query_engine_3


def get_tree_engine_with_sort_from_retriever(
        retriever_1,
        retriever_2,
        retriever_3,
        ):

    query_engine_1 = RetrieverQueryEngine.from_args(
        retriever=retriever_1, 
        response_mode="tree_summarize",
        node_postprocessors=[SortNodePostprocessor()],
        )

    query_engine_2 = RetrieverQueryEngine.from_args(
        retriever=retriever_2, 
        response_mode="tree_summarize",
        node_postprocessors=[SortNodePostprocessor()],
        )

    query_engine_3 = RetrieverQueryEngine.from_args(
        retriever=retriever_3, 
        response_mode="tree_summarize",
        node_postprocessors=[SortNodePostprocessor()],
        )

    return query_engine_1, query_engine_2, query_engine_3


def get_accumulate_query_engine_from_retriever(
        retriever_1,
        retriever_2,
        ):

    query_engine_1 = RetrieverQueryEngine.from_args(
        retriever=retriever_1, 
        response_mode="accumulate",
        )

    retriever_2_engine = RetrieverQueryEngine.from_args(
        retriever=retriever_2, 
        response_mode="accumulate",
        )

    return query_engine_1, retriever_2_engine


def get_accumulate_query_engine_with_sort_from_retriever(
        retriever_1,
        retriever_2,
        ):

    query_engine_1 = RetrieverQueryEngine.from_args(
        retriever=retriever_1, 
        response_mode="accumulate",
        node_postprocessors=[SortNodePostprocessor()],
        )

    retriever_2_engine = RetrieverQueryEngine.from_args(
        retriever=retriever_2, 
        response_mode="accumulate",
        node_postprocessors=[SortNodePostprocessor()],
        )

    return query_engine_1, retriever_2_engine


def change_default_engine_prompt_to_in_detail(engine):

    new_text_qa_tmpl_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, answer the query. \n"
    "Try to include as many key details as possible.\n"
    "Query: {query_str}\n"
    "Answer: "
    )

    new_refine_tmpl_str = (
    "The original query is as follows:\n"
    "---------------------\n"
    "{query_str}\n"
    "---------------------\n"
    "We have provided an existing answer:\n"
    "---------------------\n"
    "{existing_answer}\n"
    "---------------------\n"
    "We have the opportunity to refine the existing answer (only if needed) \n"
    "with some more context below.\n"
    "---------------------\n"
    "{context_msg}\n"
    "---------------------\n"
    "Given the new context, refine the original answer to better answer the query. \n"
    "Try to include as many key details as possible. \n"
    "If the context isn't useful, return the original answer.\n"
    "Refined Answer: "
    )

    new_text_qa_tmpl = PromptTemplate(new_text_qa_tmpl_str)
    engine.update_prompts(
                    {"response_synthesizer:text_qa_template": 
                                            new_text_qa_tmpl}
                                            )
    
    new_refine_tmpl = PromptTemplate(new_refine_tmpl_str)
    engine.update_prompts(
                    {"response_synthesizer:refine_template": 
                                            new_refine_tmpl}
                                            )
    return engine


def change_tree_engine_prompt_to_in_detail(engine):

    new_summary_tmpl_str = (
    "Context information from multiple sources is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the information from multiple sources and not prior knowledge, "
    "answer the query below.\n"
    "Try to include as many key details as possible.\n"
    "Query: {query_str}\n"
    "Answer: "
    )

    new_summary_tmpl = PromptTemplate(new_summary_tmpl_str)
    engine.update_prompts(
                    {"response_synthesizer:summary_template": 
                                            new_summary_tmpl}
                                            )
    return engine


def change_accumulate_engine_prompt_to_in_detail(engine):

    new_text_qa_tmpl_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, answer the query. \n"
    "Try to include as many key details as possible.\n"
    "Query: {query_str}\n"
    "Answer: "
    )

    new_text_qa_tmpl = PromptTemplate(new_text_qa_tmpl_str)
    engine.update_prompts(
                    {"response_synthesizer:text_qa_template": 
                                            new_text_qa_tmpl}
                                            )
    return engine


def display_prompt_dict(type, _prompts_dict):
    print(f"\n{type}\n")
    for k, p in _prompts_dict.items():
        print(f"\nPrompt Key: {k} \nText:\n")
        print(p.get_template() + "\n")


def print_retreived_nodes(method, _retrieved_nodes):

    print(f"\n\nMETHOD: {method.upper()}")
    # Loop through each NodeWithScore in the retreived nodes
    for (i, node_with_score) in enumerate(_retrieved_nodes):
        _node = node_with_score.node  # The TextNode object
        score = node_with_score.score  # The similarity score
        chunk_id = _node.id_  # The chunk ID

        # Extract the relevant metadata from the node
        file_name = _node.metadata.get("file_name", "Unknown")
        file_path = _node.metadata.get("file_path", "Unknown")

        # Extract the text content from the node
        text_content = _node.text if _node.text else "No content available"

        # Print the results in a user-friendly format
        print(f"\n\n{method.upper()}:")
        print(f"Item number: {i+1}")
        print(f"Score: {score}")
        # print(f"File Name: {file_name}")
        # print(f"File Path: {file_path}")
        print(f"Id: {chunk_id}")
        print("\nExtracted Content:\n")
        print(text_content)
        # print("\n" + "=" * 40 + " End of Result " + "=" * 40 + "\n")
        # print("\n")