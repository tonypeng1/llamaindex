"""
GLiNER Entity Extractor for LlamaIndex.

This module provides a custom GLiNER-based entity extractor that integrates
with LlamaIndex's transformation pipeline as a replacement for span-marker
EntityExtractor.

Key Features:
- Domain-specific entity extraction (20 types per domain)
- Compatible with LlamaIndex pipelines
- Standardized metadata format for consistency with existing code
"""

from typing import List, Optional, Dict
from llama_index.core.schema import BaseNode, TransformComponent
from gliner import GLiNER


class GLiNERExtractor(TransformComponent):
    """
    Custom GLiNER extractor for flexible, domain-specific entity extraction.
    
    This extractor uses GLiNER models for zero-shot named entity recognition,
    allowing flexible entity types to be defined at runtime rather than being
    limited to a fixed set of entity categories.
    
    Note: This class stores the GLiNER model as a private attribute (_gliner_model)
    to avoid conflicts with Pydantic's field validation in TransformComponent.
    """
    
    def __init__(
        self,
        model_name: str = "urchade/gliner_medium-v2.1",
        entity_labels: Optional[List[str]] = None,
        threshold: float = 0.5,
        device: str = "mps"
    ):
        """
        Initialize the GLiNER extractor.
        
        Args:
            model_name (str): HuggingFace model name for GLiNER
                Default: "urchade/gliner_medium-v2.1" (~500MB, good performance)
                Options: "urchade/gliner_small-v2.1" (~150MB, faster)
                         "urchade/gliner_large-v2.1" (~1.5GB, best accuracy)
            entity_labels (List[str]): Entity types to extract
                Recommended: 10-20 types for optimal performance
                Examples: ["person", "organization", "location", "date"]
            threshold (float): Minimum confidence score (0.0-1.0)
                Default: 0.5 (balanced precision/recall)
                Higher = fewer but more confident predictions
            device (str): Device for inference
                "mps" = Apple Silicon GPU (recommended for Mac)
                "cuda" = NVIDIA GPU
                "cpu" = CPU fallback
        """
        super().__init__()
        
        print(f"🔄 Loading GLiNER model: {model_name}...")
        # Store as private attribute to avoid Pydantic validation issues
        self._gliner_model = GLiNER.from_pretrained(model_name)
        self._entity_labels = entity_labels or []
        self._threshold = threshold
        self._device = device
        
        print(f"\n✅ GLiNER loaded with {len(self._entity_labels)} entity types")
        
    def __call__(self, nodes: List[BaseNode], **kwargs) -> List[BaseNode]:
        """
        Extract entities from nodes using GLiNER.
        
        This method processes each node's text through GLiNER to identify
        entities, then adds them to the node's metadata in a standardized
        format compatible with existing query filtering logic.
        
        Args:
            nodes (List[BaseNode]): List of nodes to process
            **kwargs: Additional arguments (unused, for compatibility)
        
        Returns:
            List[BaseNode]: Nodes with entity metadata added
        """
        for node in nodes:
            # Extract entities from node text
            entities = self._gliner_model.predict_entities(
                node.text,
                self._entity_labels,
                threshold=self._threshold
            )
            
            # Group entities by label type
            entity_dict = {}
            label_map = self._get_label_mapping()
            
            for entity in entities:
                label = entity["label"]
                text = entity["text"]
                
                # Standardize label to uppercase for consistency
                # This ensures compatibility with existing entity filtering code
                standardized_label = label_map.get(label.lower(), label.upper())
                
                # Add to entity dictionary, avoiding duplicates
                if standardized_label not in entity_dict:
                    entity_dict[standardized_label] = []
                if text not in entity_dict[standardized_label]:
                    entity_dict[standardized_label].append(text)
            
            # Update node metadata with extracted entities
            node.metadata.update(entity_dict)
            
            # Output found entities if any (useful for debugging query extraction)
            if entity_dict:
                print(f"🔍 GLiNER Entities found: {entity_dict}")
        
        return nodes
    
    def _get_label_mapping(self) -> Dict[str, str]:
        """
        Map GLiNER entity labels to standardized uppercase format.
        
        This mapping ensures consistency with existing code that expects
        uppercase entity type keys (e.g., PER, ORG, LOC from MultiNERD).
        
        Returns:
            Dict[str, str]: Mapping from lowercase GLiNER labels to standardized keys
        """
        return {
            # Academic (20)
            "author": "AUTHOR", "researcher": "RESEARCHER", "institution": "INSTITUTION",
            "university": "UNIVERSITY", "laboratory": "LABORATORY", "model": "MODEL",
            "algorithm": "ALGORITHM", "method": "METHOD", "framework": "FRAMEWORK",
            "architecture": "ARCHITECTURE", "dataset": "DATASET", "benchmark": "BENCHMARK",
            "baseline": "BASELINE", "metric": "METRIC", "experiment": "EXPERIMENT",
            "result": "RESULT", "finding": "FINDING", "theory": "THEORY",
            "technique": "TECHNIQUE", "publication": "PUBLICATION",
            
            # Technical (20)
            "function": "FUNCTION", "class": "CLASS", "parameter": "PARAMETER",
            "variable": "VARIABLE", "module": "MODULE", "package": "PACKAGE",
            "library": "LIBRARY", "command": "COMMAND", "file": "FILE",
            "directory": "DIRECTORY", "config_option": "CONFIG_OPTION",
            "environment_variable": "ENV_VAR", "api": "API", "endpoint": "ENDPOINT",
            "protocol": "PROTOCOL", "service": "SERVICE", "dependency": "DEPENDENCY",
            "tool": "TOOL", # 'method' and 'framework' shared with academic
            
            # Financial (20)
            "company": "COMPANY", "corporation": "CORPORATION", "subsidiary": "SUBSIDIARY",
            "competitor": "COMPETITOR", "partner": "PARTNER", "executive": "EXECUTIVE",
            "ceo": "CEO", "cfo": "CFO", "director": "DIRECTOR", "analyst": "ANALYST",
            "revenue": "REVENUE", "profit": "PROFIT", "margin": "MARGIN",
            "growth": "GROWTH", "guidance": "GUIDANCE", "regulator": "REGULATOR",
            "auditor": "AUDITOR", "regulation": "REGULATION", "market": "MARKET",
            "sector": "SECTOR",
            
            # General (20)
            "person": "PER", "organization": "ORG", "location": "LOC",
            "city": "CITY", "country": "COUNTRY", "date": "DATE", "time": "TIME",
            "event": "EVENT", "product": "PRODUCT", "service": "SERVICE",
            "technology": "TECH", "concept": "CONCEPT", "topic": "TOPIC",
            "award": "AWARD", "project": "PROJECT", "initiative": "INITIATIVE",
            "government": "GOVERNMENT", # 'institution', 'company', 'publication' shared
            
            # Paul Graham (21)
            "founder": "FOUNDER", "entrepreneur": "ENTREPRENEUR", "investor": "INVESTOR",
            "colleague": "COLLEAGUE", "friend": "FRIEND", "programmer": "PROGRAMMER",
            "artist": "ARTIST", "startup": "STARTUP", "programming_language": "PROG_LANG",
            "software": "SOFTWARE", # others shared
            
            # Career (20)
            "mentor": "MENTOR", "manager": "MANAGER", "recruiter": "RECRUITER",
            "industry": "INDUSTRY", "role": "ROLE", "job_title": "JOB_TITLE",
            "skill": "SKILL", "certification": "CERTIFICATION", "resource": "RESOURCE",
            "platform": "PLATFORM" # others shared
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_extraction_summary(nodes: List[BaseNode], max_nodes: int = 3):
    """
    Print a summary of entity extraction results.
    
    Args:
        nodes (List[BaseNode]): Nodes with extracted entities
        max_nodes (int): Maximum number of nodes to display
    """
    print(f"\n📊 Entity Extraction Summary")
    print(f"{'='*80}")
    
    total_entities = 0
    entity_type_counts = {}
    
    for i, node in enumerate(nodes[:max_nodes]):
        print(f"\n📄 Node {i+1} (first 100 chars): {node.text[:100]}...")
        
        if not any(k for k in node.metadata.keys() if k.isupper()):
            print("   ⚠️  No entities extracted")
            continue
        
        print("   🏷️  Extracted entities:")
        for entity_type, entities in node.metadata.items():
            if entity_type.isupper() and entities:  # Only show entity metadata
                print(f"      {entity_type}: {entities}")
                total_entities += len(entities)
                entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + len(entities)
    
    if len(nodes) > max_nodes:
        print(f"\n   ... and {len(nodes) - max_nodes} more nodes")
    
    print(f"\n{'='*80}")
    print(f"Total entities extracted: {total_entities}")
    print(f"Entity types found: {list(entity_type_counts.keys())}")
    print(f"{'='*80}\n")


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    from llama_index.core.schema import Document
    from llama_index.core.node_parser import SentenceSplitter
    from extraction_schemas import get_gliner_entity_labels
    
    print("\n🧪 GLiNER Extractor Test\n")
    
    # Test 1: Academic content
    print("="*80)
    print("TEST 1: Academic Paper Entities")
    print("="*80)
    
    academic_labels = get_gliner_entity_labels(schema_name="academic")
    extractor = GLiNERExtractor(
        model_name="urchade/gliner_medium-v2.1",
        entity_labels=academic_labels,
        threshold=0.5,
        device="mps"
    )
    
    text = """
    The Transformer architecture was introduced by Vaswani et al. at Google Brain.
    We evaluated it on the WMT 2014 dataset and achieved a BLEU score of 28.4.
    The experiments were conducted at Stanford University using 8 NVIDIA P100 GPUs.
    """
    
    doc = Document(text=text)
    parser = SentenceSplitter(chunk_size=512)
    nodes = parser.get_nodes_from_documents([doc])
    nodes_with_entities = extractor(nodes)
    
    print_extraction_summary(nodes_with_entities)
    
    print("\n✅ Test complete!")
