"""
LangExtract integration for enriching document nodes with structured metadata.

This module provides functions to extract structured metadata from text chunks
using Google's LangExtract library with OpenAI GPT-4.
"""

import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv

import langextract as lx
from llama_index.core.schema import TextNode
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI

from langextract_schemas import get_schema, get_paul_graham_schema_definitions

# Load environment variables
load_dotenv()


def extract_metadata_from_text(text: str, schema_name: str = "paul_graham_detailed") -> Dict[str, Any]:
    """
    Extract structured metadata from a text chunk using LangExtract.
    
    Parameters:
    text (str): The text to extract metadata from (typically a 256-char chunk)
    schema_name (str): The extraction schema to use
    
    Returns:
    Dict[str, Any]: Extracted metadata with flattened structure for Milvus
    """
    # Get the extraction schema
    schema_config = get_schema(schema_name)
    
    try:
        # Extract using LangExtract with OpenAI GPT-4
        result = lx.extract(
            text_or_documents=text,
            prompt_description=schema_config["prompt"],
            examples=schema_config["examples"],
            model_id="gpt-4o",  # Using GPT-4 Omni
        )
        
        # Flatten the extraction results into Milvus-friendly metadata
        metadata = flatten_extraction_result(result)
        return metadata
        
    except Exception as e:
        print(f"Warning: LangExtract failed for chunk: {e}")
        return {}


def flatten_extraction_result(result) -> Dict[str, Any]:
    """
    Convert LangExtract result into flat metadata structure for Milvus.
    
    Parameters:
    result: LangExtract extraction result
    
    Returns:
    Dict[str, Any]: Flattened metadata dictionary
    """
    metadata = {}
    
    if not result or not result.extractions:
        return metadata
    
    # Lists to collect extractions by type
    concepts = []
    concept_categories = set()
    concept_importance = set()
    
    advice_items = []
    advice_types = set()
    advice_domains = set()
    
    entities = []
    entity_roles = set()
    entity_names = []
    entity_significance = set()
    
    experiences = []
    experience_periods = set()
    experience_sentiments = set()
    
    time_refs = []
    time_decades = set()
    time_specificity = set()
    
    # Process each extraction
    for extraction in result.extractions:
        attrs = extraction.attributes or {}
        
        if extraction.extraction_class == "concept":
            concepts.append(extraction.extraction_text)
            if "category" in attrs:
                concept_categories.add(attrs["category"])
            if "importance" in attrs:
                concept_importance.add(attrs["importance"])
                
        elif extraction.extraction_class == "advice":
            advice_items.append(extraction.extraction_text)
            if "type" in attrs:
                advice_types.add(attrs["type"])
            if "domain" in attrs:
                advice_domains.add(attrs["domain"])
                
        elif extraction.extraction_class == "entity":
            entities.append(extraction.extraction_text)
            entity_names.append(extraction.extraction_text)
            if "role" in attrs:
                entity_roles.add(attrs["role"])
            if "significance" in attrs:
                entity_significance.add(attrs["significance"])
                
        elif extraction.extraction_class == "experience":
            experiences.append(extraction.extraction_text)
            if "period" in attrs:
                experience_periods.add(attrs["period"])
            if "sentiment" in attrs:
                experience_sentiments.add(attrs["sentiment"])
                
        elif extraction.extraction_class == "time":
            time_refs.append(extraction.extraction_text)
            if "decade" in attrs:
                time_decades.add(attrs["decade"])
            if "specificity" in attrs:
                time_specificity.add(attrs["specificity"])
    
    # Add to metadata (convert sets to lists for JSON serialization)
    if concepts:
        metadata["langextract_concepts"] = concepts
    if concept_categories:
        metadata["concept_categories"] = list(concept_categories)
    if concept_importance:
        metadata["concept_importance"] = list(concept_importance)
    
    if advice_items:
        metadata["langextract_advice"] = advice_items
    if advice_types:
        metadata["advice_types"] = list(advice_types)
    if advice_domains:
        metadata["advice_domains"] = list(advice_domains)
    
    if entities:
        metadata["langextract_entities"] = entities
        metadata["entity_names"] = entity_names
    if entity_roles:
        metadata["entity_roles"] = list(entity_roles)
    if entity_significance:
        metadata["entity_significance"] = list(entity_significance)
    
    if experiences:
        metadata["langextract_experiences"] = experiences
    if experience_periods:
        metadata["experience_periods"] = list(experience_periods)
    if experience_sentiments:
        metadata["experience_sentiments"] = list(experience_sentiments)
    
    if time_refs:
        metadata["time_references"] = time_refs
    if time_decades:
        metadata["time_decades"] = list(time_decades)
    if time_specificity:
        metadata["time_specificity"] = list(time_specificity)
    
    return metadata


def enrich_nodes_with_langextract(
    nodes: List[TextNode],
    schema_name: str = "paul_graham_detailed",
    verbose: bool = True
) -> List[TextNode]:
    """
    Enrich a list of TextNode objects with LangExtract metadata.
    
    This function processes each node (chunk), extracts structured metadata,
    and adds it to the node's metadata dictionary.
    
    Parameters:
    nodes (List[TextNode]): List of nodes to enrich
    schema_name (str): The extraction schema to use
    verbose (bool): Print progress information
    
    Returns:
    List[TextNode]: The same nodes with enriched metadata
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"LangExtract Metadata Enrichment")
        print(f"{'='*80}")
        print(f"Processing {len(nodes)} nodes with schema: {schema_name}")
        print(f"This may take several minutes (API calls to OpenAI)...")
    
    # Check for API key
    if not os.environ.get('OPENAI_API_KEY'):
        print("\n⚠️  Warning: OPENAI_API_KEY not found!")
        print("   Skipping LangExtract enrichment.")
        return nodes
    
    enriched_count = 0
    failed_count = 0
    
    for i, node in enumerate(nodes):
        if verbose and (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(nodes)} nodes...")
        
        try:
            # Extract metadata from node text
            langextract_metadata = extract_metadata_from_text(
                node.text,
                schema_name=schema_name
            )
            
            # Add to node metadata (preserves existing metadata)
            if langextract_metadata:
                node.metadata.update(langextract_metadata)
                enriched_count += 1
            
        except Exception as e:
            if verbose:
                print(f"  Warning: Failed to enrich node {i}: {e}")
            failed_count += 1
            continue
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"✅ Enrichment complete!")
        print(f"   Successfully enriched: {enriched_count}/{len(nodes)} nodes")
        if failed_count > 0:
            print(f"   Failed: {failed_count} nodes")
        print(f"{'='*80}\n")
    
    return nodes


def print_sample_metadata(nodes: List[TextNode], num_samples: int = 3):
    """
    Print sample metadata from enriched nodes for debugging.
    
    Parameters:
    nodes (List[TextNode]): List of enriched nodes
    num_samples (int): Number of samples to print
    """
    print(f"\n{'='*80}")
    print(f"Sample Enriched Metadata (first {num_samples} nodes)")
    print(f"{'='*80}")
    
    for i, node in enumerate(nodes[:num_samples]):
        print(f"\nNode {i+1}:")
        print(f"Text: {node.text[:100]}...")
        print(f"Metadata keys: {list(node.metadata.keys())}")
        
        # Print LangExtract-specific metadata
        langextract_keys = [k for k in node.metadata.keys() if 'langextract' in k or 'concept' in k or 'advice' in k or 'experience' in k or 'time' in k]
        if langextract_keys:
            print(f"LangExtract metadata:")
            for key in langextract_keys:
                print(f"  {key}: {node.metadata[key]}")
        print("-" * 80)


def extract_query_metadata_filters(query_str: str, schema_name: str = "paul_graham_detailed") -> Dict[str, List[str]]:
    """
    Uses an LLM to analyze the query and extract metadata filters based on the schema.
    
    Parameters:
    query_str (str): The user query
    schema_name (str): The schema to use
    
    Returns:
    Dict[str, List[str]]: Dictionary of filters (e.g., {'concept_categories': ['programming']})
    """
    # Only support Paul Graham schema for now
    if "paul_graham" not in schema_name:
        return {}
        
    defs = get_paul_graham_schema_definitions()
    
    # Construct prompt
    prompt = f"""
    Analyze the following user query and identify if the user is looking for specific categories of information defined in our schema.
    
    Schema Definitions:
    - Concept Categories: {defs['concept_categories']}
    - Advice Domains: {defs['advice_domains']}
    - Experience Periods: {defs['experience_periods']}
    - Experience Sentiments: {defs['experience_sentiments']}
    - Time Decades: {defs['time_decades']}
    - Entity Roles: {defs['entity_roles']}
    
    Query: "{query_str}"
    
    If the query implies a filter on any of these attributes, return a JSON object with the attribute name as key and a list of matching values.
    Only return keys that have matches. If no matches, return an empty JSON object {{}}.
    
    Example 1:
    Query: "What advice does he give about startups?"
    Output: {{"advice_domains": ["startups"], "concept_categories": ["startups"]}}
    
    Example 2:
    Query: "Tell me about his childhood experiences"
    Output: {{"experience_periods": ["childhood"]}}
    
    Example 3:
    Query: "What does he say about Lisp?"
    Output: {{"concept_categories": ["programming"]}}
    
    Example 4:
    Query: "Who were the founders he worked with?"
    Output: {{"entity_roles": ["founder"]}}
    
    Output JSON:
    """
    
    try:
        # Use the configured LLM or default to OpenAI
        llm = Settings.llm or OpenAI(model="gpt-4o")
        response = llm.complete(prompt)
        
        # Parse JSON response - handle various formats LLMs might return
        response_text = response.text.strip()
        
        # Try to extract JSON from markdown code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            # Handle plain code blocks - extract content between first pair of ```
            parts = response_text.split("```")
            if len(parts) >= 3:
                response_text = parts[1].strip()
            elif len(parts) == 2:
                # Possibly just opening ``` with content after
                response_text = parts[1].strip() if parts[1].strip() else parts[0].strip()
        
        # Try to find JSON object in the response if not already clean JSON
        if not response_text.startswith("{"):
            # Look for a JSON object anywhere in the response
            import re
            json_match = re.search(r'\{[^{}]*\}', response_text)
            if json_match:
                response_text = json_match.group()
        
        # Handle empty or whitespace-only response
        if not response_text or response_text.isspace():
            return {}
            
        filters = json.loads(response_text)
        
        if filters:
            print(f"\n🔍 Extracted Query Filters: {filters}")
            
        return filters
        
    except json.JSONDecodeError as e:
        # Log the actual response for debugging
        print(f"Warning: Failed to extract query filters (JSON parse error): {e}")
        print(f"   LLM response was: '{response.text[:200]}...' " if len(response.text) > 200 else f"   LLM response was: '{response.text}'")
        return {}
    except Exception as e:
        print(f"Warning: Failed to extract query filters: {e}")
        return {}
