
def get_article_link(
        _article_dictory, 
        _article_name
        ):
    
    return "./data/" + _article_dictory + "/" + _article_name


def get_database_and_window_collection_name(
        _article_dictory, 
        _chuck_method, 
        _embed_model_name,
        _window_size
        ):
    
    _database_name = _article_dictory + "_" + _chuck_method
    _collection_name = _embed_model_name + "_window_size_" + str(_window_size)
    return _database_name, _collection_name


def get_compact_tree_and_accumulate_engine(
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


def get_rerank_compact_tree_and_accumulate_engine(
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
    

def print_retreived_nodes(_retriever):
    # Loop through each NodeWithScore in the retreived nodes
    for (i, node_with_score) in enumerate(_retriever):
        node = node_with_score.node  # The TextNode object
        score = node_with_score.score  # The similarity score
        chunk_id = node.id_  # The chunk ID

        # Extract the relevant metadata from the node
        file_name = node.metadata.get("file_name", "Unknown")
        file_path = node.metadata.get("file_path", "Unknown")

        # Extract the text content from the node
        text_content = node.text if node.text else "No content available"

        # Print the results in a user-friendly format
        print(f"\n\nItem number: {i+1}")
        print(f"Score: {score}")
        # print(f"File Name: {file_name}")
        # print(f"File Path: {file_path}")
        print(f"Id: {chunk_id}")
        print("\nExtracted Content:\n")
        print(text_content)
        # print("\n" + "=" * 40 + " End of Result " + "=" * 40 + "\n")
        # print("\n")