
from gliner_extractor import GLiNERExtractor
from extraction_schemas import get_gliner_entity_labels

def test_query_extraction():
    queries = [
        "What programming languages are discussed in the document?",
        "What did Paul Graham say about Lisp and Python?",
        "Who is Jessica Livingston?"
    ]
    schema_name = "paul_graham_detailed"
    
    labels = get_gliner_entity_labels(schema_name=schema_name)
    print(f"Testing with {len(labels)} labels from '{schema_name}' schema")
    
    extractor = GLiNERExtractor(
        model_name="urchade/gliner_medium-v2.1",
        entity_labels=labels,
        threshold=0.3,
        device="mps"
    )
    
    from llama_index.core.schema import TextNode
    for query in queries:
        node = TextNode(text=query)
        extractor([node])
        print(f"\nQuery: {query}")
        print(f"Extracted: {node.metadata}")

if __name__ == "__main__":
    test_query_extraction()
