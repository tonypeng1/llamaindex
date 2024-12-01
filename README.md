## RAG using Llamaindex

RAG (Retrieval Augmented Generation) is a technique that combines the power of large language models (LLMs) with external data sources to enhance the accuracy and relevance of generated responses. 

A RAG that uses sub-question and tool-selecting query engines from llamaindex is demonstrated to query the article "What I Worked On" by Paul Graham.

## RAG Features
This RAG has the following features:

1. Use Milvus and MongoDB as the vector database and the document store, respectively. This code prevents duplicate database entries by checking for existing data before parsing the article and adding it to the database.
2. Use a sub-question query engine with 3 query engine tools: one for answering a query about specific information (keyphrase_tool), one for a query that requires the information on certain pages (page_filter_tool), and finally one for a query that needs a full-context summary (summary_tool). Details about these tools are discussed in the next section.
3. The keyphrase_tool and the page_filter_tool use a fusion retriever that combines a vector retriever and a BM25 retriever.
4. The noise of the BM25 retriever used in the keyphrase_tool is reduced by first extracting the key phrase of a query using the model KeyBERT, then using another BM25 retriever to retrieve the relevant text nodes with only the key phrase as the query (rather than the original full query), and finally defining the BM25 retriever with only these text nodes (rather than using all the nodes from the document store).
5. A Named Entity Recognition (NER) model extracts the named entities in each node.
6. The nodes retrieved by the fusion retriever are post-processed by a reranker, a previous-next postprocessor (to include all the adjacent nodes on a page), and finally by a custom sorting postprocessor (to sort the nodes in ascending order based on page number and node occurrence within a page using the data start_char_idx).

## File Structure

For this demonstration, 3 files are used:
1. llama_bm25_simple.py,
2. utility_simple.py, and
3. database_operation_simple.py.

The file "llama_bm25_simple.py" is the main file that uses the functions defined in the other two files.

## Requirement File
The requirement.txt file contains the necessary packages for this demonstration.

```
llama-index==0.11.7
llama-index-embeddings-huggingface==0.3.1
llama-index-storage-docstore-mongodb==0.2.0
llama-index-vector-stores-milvus==0.2.3
llama-index-retrievers-bm25==0.3.0
motor==3.5.1
pymilvus==2.4.6
PyMuPDF==1.24.10
keybert==0.8.5
llama-index-llms-mistralai==0.2.2
llama-parse==0.5.2
llama-index-llms-anthropic==0.3.1
llama-index-multi-modal-llms-anthropic==0.2.2
llama-index-postprocessor-flag-embedding-reranker==0.2.0
FlagEmbedding==1.2.11
llama-index-postprocessor-colbert-rerank==0.2.1
llama-index-question-gen-guidance==0.2.0
llama-index-extractors-entity==0.2.1
transformers==4.40.2
span-marker==1.5.0
```

Please note that there is a need to downgrade the transformers package to version 4.40.2, as the latest version is not compatible with the llamaindex packages.

## Article Directory
The .pdf file of Paull Graham's article “What I Worked On”, which is not included in this repository, is located at the directory:

```
./data/paul_graham/paul_graham_essay.pdf
```

This article can be found [here](https://drive.google.com/file/d/1YzCscCmQXn2IcGS-omcAc8TBuFrpiN4-/view?usp=sharing).

## Database and Collection Names
The database name (includes article name and parsing method) and collection/namespace name (includes embedding model, chuck size, overlap size, and metadata if available) in both the Milvus vector database and the MongoDB document store are:

- database name: "paul_graham_sentence_splitter"
- collection/namespace name: "openai_embedding_3_small_chunk_size_512_chunk_overlap_128_metadata_entity"

There is one more collection/namespace in the MongoDB document store, which is used to store the summary:

- collection/namespace name: "openai_embedding_3_small_chunk_size_512_chunk_overlap_128_summary_metadata_entity"

## Model API Key
The API keys for OpenAI and Anthropic are stored in the .env file, which is not included in this repository. The .evn file looks like:

```
OPENAI_API_KEY = your_openai_api_key
ANTHROPIC_API_KEY = your_anthropic_api_key
```

## Examples of Queries and Answers

4 sets of queries and answer that illustrate the capability of this RAG can be found in this [Medium artile](https://medium.com/@tony3t3t/rag-with-sub-question-and-tool-selecting-query-engines-using-llamaindex-05349cb4120c).
