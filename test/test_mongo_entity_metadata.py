"""Utility test for inspecting entity metadata stored in MongoDB.

Run with `python -m pytest test/test_mongo_entity_metadata.py -k entity`
or execute the file directly to print all entity-related keys and valuesper node. 
The test is intentionally read-only:

"python test/test_mongo_entity_metadata.py" (lists all entity-related keys/values)
"python test/test_mongo_entity_metadata.py --entity-types persons locations" (filtering)


Filtering tips:
- Set `MONGO_ENTITY_FILTERS="persons,locations"` to limit output via env
- Use `python test/test_mongo_entity_metadata.py --entity-types persons locations`
    for ad-hoc filtering (pass `all` to disable filtering even if env is set)
- The script now prints a merged dictionary of entity_type -> unique values
    after scanning all nodes (respects any active filters)
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from collections import defaultdict
from typing import Any, DefaultDict, Iterator, List, Optional, Set, Tuple

try:  # pytest is optional; we skip gracefully when it is unavailable.
    pytest = importlib.import_module("pytest")
except ModuleNotFoundError:  # pragma: no cover - fallback path when pytest missing
    pytest = None  # type: ignore

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB_NAME = os.environ.get(
    "MONGO_DB_NAME", "paul_graham_paul_graham_essay_sentence_splitter"
)
ENTITY_COLLECTION_NAME = os.environ.get(
    "MONGO_ENTITY_COLLECTION",
    "openai_embedding_3_small_chunk_size_256_chunk_overlap_64_metadata_entity/data",
)
SERVER_SELECTION_TIMEOUT_MS = int(os.environ.get("MONGO_TIMEOUT_MS", "5000"))

DEFAULT_ENTITY_KEY_HINTS: Set[str] = {
    "locations",
    "location",
    "persons",
    "people",
    "organizations",
    "organization",
    "companies",
    "company",
    "diseases",
    "disease",
    "entities",
    "entity_names",
    "entity_roles",
    "entity_ids",
    "entity_types",
    "roles",
    "titles",
    "keywords",
    "concepts",
}

CUSTOM_HINTS = {
    hint.strip().lower()
    for hint in os.environ.get("MONGO_ENTITY_KEY_HINTS", "").split(",")
    if hint.strip()
}

ENTITY_KEY_HINTS: Set[str] = DEFAULT_ENTITY_KEY_HINTS | CUSTOM_HINTS


def _is_entity_key(key: str) -> bool:
    lowered = key.lower()
    return "entity" in lowered or lowered in ENTITY_KEY_HINTS


ENTITY_TYPE_FILTERS: Set[str] = {
    part.strip().lower()
    for part in os.environ.get("MONGO_ENTITY_FILTERS", "").split(",")
    if part.strip()
}


def _entity_type_from_path(path: str) -> str:
    """Return the terminal field name (minus list indexes) for filtering."""

    terminal = path.rsplit(".", 1)[-1]
    if "[" in terminal:
        terminal = terminal.split("[", 1)[0]
    return terminal.lower()


def _connect_to_entity_collection() -> Tuple[MongoClient, Collection]:
    """Return a client and the target collection, raising if not available."""

    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=SERVER_SELECTION_TIMEOUT_MS)
    db = client[MONGO_DB_NAME]

    if ENTITY_COLLECTION_NAME not in db.list_collection_names():
        client.close()
        raise RuntimeError(
            f"Collection '{ENTITY_COLLECTION_NAME}' not found in database '{MONGO_DB_NAME}'."
        )

    return client, db[ENTITY_COLLECTION_NAME]


def _iter_entity_pairs(obj: Any, prefix: str) -> Iterator[Tuple[str, Any]]:
    """Yield dotted key paths and their values for keys containing 'entity'."""

    if isinstance(obj, dict):
        for key, value in obj.items():
            current_path = f"{prefix}.{key}" if prefix else key
            if _is_entity_key(key):
                yield current_path, value
            if isinstance(value, (dict, list)):
                yield from _iter_entity_pairs(value, current_path)
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            current_path = f"{prefix}[{index}]" if prefix else f"[{index}]"
            if isinstance(value, (dict, list)):
                yield from _iter_entity_pairs(value, current_path)


def _collect_entity_pairs(doc: dict) -> List[Tuple[str, Any]]:
    """Collect all entity-related key/value pairs from a Mongo document."""

    pairs: List[Tuple[str, Any]] = []
    sections: List[Tuple[str, Any]] = []

    def _add_sections(base_prefix: str, container: Any) -> None:
        if not isinstance(container, dict):
            return

        metadata = container.get("metadata")
        extra_info = container.get("extra_info")
        relationships = container.get("relationships")

        if isinstance(metadata, dict):
            prefix = f"{base_prefix}.metadata" if base_prefix else "metadata"
            sections.append((prefix, metadata))
        if isinstance(extra_info, dict):
            prefix = f"{base_prefix}.extra_info" if base_prefix else "extra_info"
            sections.append((prefix, extra_info))
        if isinstance(relationships, dict):
            prefix = f"{base_prefix}.relationships" if base_prefix else "relationships"
            sections.append((prefix, relationships))

    _add_sections("", doc)
    _add_sections("__data__", doc.get("__data__"))

    for prefix, payload in sections:
        pairs.extend(list(_iter_entity_pairs(payload, prefix)))

    def _scan_container(container: Any, base_prefix: str = "") -> None:
        if not isinstance(container, dict):
            return
        for key, value in container.items():
            if key in {"metadata", "extra_info", "relationships"}:
                continue
            if not _is_entity_key(key):
                continue
            path = f"{base_prefix}.{key}" if base_prefix else key
            pairs.append((path, value))
            if isinstance(value, (dict, list)):
                pairs.extend(list(_iter_entity_pairs(value, path)))

    _scan_container(doc)
    _scan_container(doc.get("__data__"), "__data__")

    return pairs


def _filter_pairs_by_type(
    pairs: List[Tuple[str, Any]], filters: Optional[Set[str]]
) -> List[Tuple[str, Any]]:
    """Return only the pairs whose terminal key matches the filter set."""

    if not filters:
        return pairs

    filtered: List[Tuple[str, Any]] = []
    for path, value in pairs:
        entity_type = _entity_type_from_path(path)
        if entity_type in filters:
            filtered.append((path, value))
    return filtered


def _iter_entity_values(value: Any) -> Iterator[str]:
    """Yield flattened scalar values for aggregation."""

    if isinstance(value, dict):
        yield json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _iter_entity_values(item)
    else:
        yield str(value)


def _to_json(value: Any) -> str:
    """Serialize values for readable console output."""

    try:
        return json.dumps(value, indent=2, default=str, ensure_ascii=True)
    except TypeError:
        return repr(value)


def test_list_entity_keys_and_values(
    entity_type_filters: Optional[Set[str]] = None,
) -> None:
    """List every entity key/value pair found in the target Mongo collection."""

    client = None

    try:
        client, collection = _connect_to_entity_collection()
    except (PyMongoError, RuntimeError) as exc:
        if pytest is not None:
            pytest.skip(f"MongoDB unavailable: {exc}")
        else:
            print(f"Skipping entity metadata listing test: {exc}")
            return

    try:
        cursor = collection.find(
            {},
            {
                "id_": 1,
                "metadata": 1,
                "extra_info": 1,
                "relationships": 1,
                "__data__": 1,
            },
        )

        total_nodes = 0
        nodes_with_entities = 0
        total_pairs = 0
        merged_entities: DefaultDict[str, Set[str]] = defaultdict(set)

        print(
            f"\nInspecting '{MONGO_DB_NAME}.{ENTITY_COLLECTION_NAME}' for entity metadata..."
        )

        active_filters = entity_type_filters if entity_type_filters is not None else ENTITY_TYPE_FILTERS

        if active_filters:
            print(f"\nApplying entity-type filter: {sorted(active_filters)}")

        for doc in cursor:
            total_nodes += 1
            entity_pairs = _collect_entity_pairs(doc)
            entity_pairs = _filter_pairs_by_type(entity_pairs, active_filters)
            if not entity_pairs:
                continue

            nodes_with_entities += 1
            node_id = (
                doc.get("id_")
                or (doc.get("__data__") or {}).get("id_")
                or doc.get("_id")
                or "<missing id>"
            )
            print(f"\nNode: {node_id}")

            for path, value in entity_pairs:
                total_pairs += 1
                print(f"  - {path}: {_to_json(value)}")

                entity_type = _entity_type_from_path(path)
                for flattened_value in _iter_entity_values(value):
                    merged_entities[entity_type].add(flattened_value)

            print("  ----")

        print(
            f"\nSummary: scanned {total_nodes} nodes, found entity metadata in {nodes_with_entities} nodes "
            f"({total_pairs} entity key/value pairs)."
        )

        if merged_entities:
            merged_dict = {
                key: sorted(values) for key, values in sorted(merged_entities.items())
            }
            print("\nMerged entity dictionaries:")
            print(json.dumps(merged_dict, indent=2, ensure_ascii=True))
        else:
            print("\nMerged entity dictionaries: {}")

    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="List entity-related metadata from a MongoDB collection."
    )
    parser.add_argument(
        "--entity-types",
        nargs="+",
        help=(
            "Optional list of terminal keys to include (e.g. persons locations). "
            "Pass 'all' to disable filtering regardless of environment settings."
        ),
    )

    args = parser.parse_args()

    cli_filters: Optional[Set[str]]
    if args.entity_types is None:
        cli_filters = None
    else:
        normalized = {value.strip().lower() for value in args.entity_types if value.strip()}
        if not normalized or "all" in normalized:
            cli_filters = set()
        else:
            cli_filters = normalized

    test_list_entity_keys_and_values(cli_filters)
