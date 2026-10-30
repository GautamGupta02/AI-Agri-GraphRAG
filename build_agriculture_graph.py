import json
import pickle
import re
from pathlib import Path
from collections import Counter

import networkx as nx


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

NORMALIZED_DIR = ROOT / "processed" / "graph_normalized"
GRAPH_DIR = ROOT / "graph"

GRAPH_DIR.mkdir(parents=True, exist_ok=True)

ENTITIES_FILE = NORMALIZED_DIR / "entities_normalized.jsonl"
RELATIONS_FILE = NORMALIZED_DIR / "relation_candidates_normalized.jsonl"

OUTPUT_PICKLE = GRAPH_DIR / "agriculture_graph.pkl"
OUTPUT_NODES = GRAPH_DIR / "nodes.jsonl"
OUTPUT_EDGES = GRAPH_DIR / "edges.jsonl"
OUTPUT_SUMMARY = GRAPH_DIR / "graph_summary.json"


# ============================================================
# VALID NODE TYPES
# ============================================================

VALID_NODE_TYPES = {
    "Crop",
    "Disease",
    "Pest",
    "Symptom",
    "Management",
    "VisualFeature",
    "ImageCase",
}


# ============================================================
# VALID RELATIONS
# ============================================================

VALID_RELATIONS = {
    "HAS_CROP",
    "HAS_DISEASE",
    "HAS_PEST",
    "HAS_SYMPTOM",
    "HAS_MANAGEMENT",
    "HAS_VISUAL_FEATURE",
    "AFFECTED_BY",
    "ATTACKED_BY",
    "MANAGED_BY",
}


# ============================================================
# VALID RELATION SHAPES
# ============================================================

VALID_RELATION_SHAPES = {

    "HAS_CROP": {
        ("ImageCase", "Crop"),
    },

    "HAS_DISEASE": {
        ("ImageCase", "Disease"),
    },

    "HAS_PEST": {
        ("ImageCase", "Pest"),
    },

    "HAS_SYMPTOM": {
        ("ImageCase", "Symptom"),
        ("Disease", "Symptom"),
    },

    "HAS_MANAGEMENT": {
        ("ImageCase", "Management"),
    },

    "HAS_VISUAL_FEATURE": {
        ("ImageCase", "VisualFeature"),
    },

    "AFFECTED_BY": {
        ("Crop", "Disease"),
    },

    "ATTACKED_BY": {
        ("Crop", "Pest"),
    },

    "MANAGED_BY": {
        ("Disease", "Management"),
    },
}


# ============================================================
# HELPERS
# ============================================================

def read_jsonl(path):

    records = []

    with open(path, "r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))

            except json.JSONDecodeError as e:

                print(
                    f"WARNING: invalid JSON "
                    f"{path.name}:{line_number}: {e}"
                )

    return records


def clean_value(value):

    if value is None:
        return ""

    return str(value).strip()


def safe_xml_text(value):

    """
    Remove characters that are invalid in XML/GraphML.

    This is kept here only if GraphML export is added later.
    """

    if value is None:
        return ""

    value = str(value)

    return re.sub(
        r"[\x00-\x08\x0B\x0C\x0E-\x1F]",
        "",
        value
    )


# ============================================================
# MAIN
# ============================================================

print("=" * 72)
print("BUILD AGRICULTURE KNOWLEDGE GRAPH")
print("=" * 72)

print()
print("Project root:")
print(ROOT)

print()
print("Entities:")
print(ENTITIES_FILE)

print()
print("Relations:")
print(RELATIONS_FILE)


# ============================================================
# 1. LOAD ENTITIES
# ============================================================

print()
print("[1/7] Loading normalized entities...")

entity_records = read_jsonl(ENTITIES_FILE)

print(
    f"      Entity records loaded: "
    f"{len(entity_records):,}"
)


# ============================================================
# 2. CREATE GRAPH
# ============================================================

print()
print("[2/7] Creating NetworkX graph...")

G = nx.MultiDiGraph()


# ============================================================
# 3. ADD NODES
# ============================================================

print()
print("[3/7] Adding nodes...")

node_type_counts = Counter()

invalid_nodes = 0
duplicate_nodes = 0


for entity in entity_records:

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # entities_normalized.jsonl uses entity_id.
    # --------------------------------------------------------

    entity_id = clean_value(
        entity.get("entity_id")
    )

    entity_type = clean_value(
        entity.get("entity_type")
    )

    entity_name = clean_value(
        entity.get("entity_name")
    )

    if not entity_id:

        invalid_nodes += 1
        continue

    if entity_type not in VALID_NODE_TYPES:

        invalid_nodes += 1
        continue

    if G.has_node(entity_id):

        duplicate_nodes += 1
        continue

    node_attributes = {

        "entity_id": entity_id,

        "entity_type": entity_type,

        "entity_name": entity_name,

        "canonical_name": clean_value(
            entity.get("canonical_name")
        ),

        "source_dataset": clean_value(
            entity.get("source_dataset")
        ),

        "source_record_id": clean_value(
            entity.get("source_record_id")
        ),

        "provenance": clean_value(
            entity.get("provenance")
        ),
    }

    G.add_node(
        entity_id,
        **node_attributes
    )

    node_type_counts[entity_type] += 1


print()
print("Node counts:")

for node_type, count in sorted(
    node_type_counts.items()
):

    print(
        f"  {node_type:<18} {count:,}"
    )

print()
print(
    f"Invalid nodes: {invalid_nodes:,}"
)

print(
    f"Duplicate nodes skipped: "
    f"{duplicate_nodes:,}"
)


# ============================================================
# 4. LOAD RELATIONS
# ============================================================

print()
print("[4/7] Loading normalized relations...")

relation_records = read_jsonl(RELATIONS_FILE)

print(
    f"      Relation records loaded: "
    f"{len(relation_records):,}"
)


# ============================================================
# 5. ADD EDGES
# ============================================================

print()
print("[5/7] Adding edges...")

relation_counts = Counter()
layer_counts = Counter()

invalid_relations = 0
missing_source_nodes = 0
missing_target_nodes = 0

seen_edges = set()


for rel in relation_records:

    # ========================================================
    # IMPORTANT:
    #
    # relation_candidates_normalized.jsonl uses:
    #
    # source
    # target
    #
    # NOT source_id / target_id.
    #
    # Example:
    #
    # source = "Crop:american sycamore"
    # target = "Pest:sycamore scale"
    # ========================================================

    source_id = clean_value(
        rel.get("source")
    )

    target_id = clean_value(
        rel.get("target")
    )

    relation = clean_value(
        rel.get("relation")
    )

    source_type = clean_value(
        rel.get("source_type")
    )

    target_type = clean_value(
        rel.get("target_type")
    )

    source_name = clean_value(
        rel.get("source_name")
    )

    target_name = clean_value(
        rel.get("target_name")
    )

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    if not source_id:

        invalid_relations += 1
        continue

    if not target_id:

        invalid_relations += 1
        continue

    if not relation:

        invalid_relations += 1
        continue

    if relation not in VALID_RELATIONS:

        invalid_relations += 1
        continue

    # --------------------------------------------------------
    # Node existence
    # --------------------------------------------------------

    if source_id not in G:

        missing_source_nodes += 1

        continue

    if target_id not in G:

        missing_target_nodes += 1

        continue

    # --------------------------------------------------------
    # Relation shape
    # --------------------------------------------------------

    shape = (
        source_type,
        target_type
    )

    valid_shapes = VALID_RELATION_SHAPES.get(
        relation,
        set()
    )

    if shape not in valid_shapes:

        invalid_relations += 1

        continue

    # --------------------------------------------------------
    # Determine layer
    # --------------------------------------------------------

    if relation == "HAS_SYMPTOM":

        if source_type == "Disease":

            layer = "semantic"

        else:

            layer = "evidence"

    elif relation in {
        "AFFECTED_BY",
        "ATTACKED_BY",
        "MANAGED_BY",
    }:

        layer = "semantic"

    else:

        layer = "evidence"

    # --------------------------------------------------------
    # Audit category
    # --------------------------------------------------------

    audit_category = clean_value(
        rel.get("audit_category")
    )

    if not audit_category:

        if layer == "semantic":

            audit_category = "NOT_AUDITED"

        else:

            audit_category = "EVIDENCE_RELATION"

    # --------------------------------------------------------
    # Uncertainty
    # --------------------------------------------------------

    original_uncertain = rel.get(
        "uncertain",
        False
    )

    uncertain = bool(original_uncertain)

    if audit_category == "TARGET_UNCERTAIN_TEXT":

        uncertain = True

    # --------------------------------------------------------
    # Unique edge signature
    # --------------------------------------------------------

    edge_signature = (
        source_id,
        relation,
        target_id,
        layer,
    )

    if edge_signature in seen_edges:

        continue

    seen_edges.add(edge_signature)

    # --------------------------------------------------------
    # Edge attributes
    # --------------------------------------------------------

    edge_attributes = {

        "relation": relation,

        "layer": layer,

        "source_id": source_id,

        "target_id": target_id,

        "source_type": source_type,

        "target_type": target_type,

        "source_name": source_name,

        "target_name": target_name,

        "evidence_type": clean_value(
            rel.get("evidence_type")
        ),

        "rule": clean_value(
            rel.get("rule")
        ),

        "source_dataset": clean_value(
            rel.get("source_dataset")
        ),

        "source_record_id": clean_value(
            rel.get("source_record_id")
        ),

        "evidence_case_id": clean_value(
            rel.get("evidence_case_id")
        ),

        "evidence_count": int(
            rel.get("evidence_count", 1)
            or 1
        ),

        "audit_category": audit_category,

        "uncertain": uncertain,
    }

    # --------------------------------------------------------
    # Preserve provenance
    #
    # JSON strings are used so the graph remains serializable.
    # --------------------------------------------------------

    provenance = rel.get(
        "provenance",
        []
    )

    if provenance:

        edge_attributes["provenance"] = json.dumps(
            provenance,
            ensure_ascii=False
        )

    else:

        edge_attributes["provenance"] = ""

    # --------------------------------------------------------
    # Add edge
    # --------------------------------------------------------

    G.add_edge(
        source_id,
        target_id,
        **edge_attributes
    )

    relation_counts[relation] += 1

    layer_counts[layer] += 1


# ============================================================
# 6. GRAPH SUMMARY
# ============================================================

print()
print("[6/7] Graph summary...")

print()
print(
    f"Nodes: {G.number_of_nodes():,}"
)

print(
    f"Edges: {G.number_of_edges():,}"
)

print()
print("Edges by layer:")

for layer, count in sorted(
    layer_counts.items()
):

    print(
        f"  {layer:<15} {count:,}"
    )

print()
print("Edges by relation:")

for relation, count in sorted(
    relation_counts.items()
):

    print(
        f"  {relation:<22} {count:,}"
    )

print()
print(
    f"Invalid relations: "
    f"{invalid_relations:,}"
)

print(
    f"Missing source nodes: "
    f"{missing_source_nodes:,}"
)

print(
    f"Missing target nodes: "
    f"{missing_target_nodes:,}"
)


# ============================================================
# 7. SAVE
# ============================================================

print()
print("[7/7] Saving graph files...")


# ------------------------------------------------------------
# NetworkX pickle
# ------------------------------------------------------------

with open(
    OUTPUT_PICKLE,
    "wb"
) as f:

    pickle.dump(
        G,
        f,
        protocol=pickle.HIGHEST_PROTOCOL
    )


# ------------------------------------------------------------
# Nodes JSONL
# ------------------------------------------------------------

with open(
    OUTPUT_NODES,
    "w",
    encoding="utf-8"
) as f:

    for node_id, attrs in G.nodes(
        data=True
    ):

        record = {
            "node_id": node_id,
            **attrs,
        }

        f.write(
            json.dumps(
                record,
                ensure_ascii=False
            )
            + "\n"
        )


# ------------------------------------------------------------
# Edges JSONL
# ------------------------------------------------------------

with open(
    OUTPUT_EDGES,
    "w",
    encoding="utf-8"
) as f:

    for source, target, attrs in G.edges(
        data=True
    ):

        record = {
            "source": source,
            "target": target,
            **attrs,
        }

        f.write(
            json.dumps(
                record,
                ensure_ascii=False
            )
            + "\n"
        )


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

summary = {

    "graph": {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
    },

    "node_type_counts": dict(
        node_type_counts
    ),

    "relation_counts": dict(
        relation_counts
    ),

    "layer_counts": dict(
        layer_counts
    ),

    "invalid_nodes": invalid_nodes,

    "duplicate_nodes_skipped": duplicate_nodes,

    "invalid_relations": invalid_relations,

    "missing_source_nodes": missing_source_nodes,

    "missing_target_nodes": missing_target_nodes,

    "input_entities": str(
        ENTITIES_FILE
    ),

    "input_relations": str(
        RELATIONS_FILE
    ),

    "output_graph": str(
        OUTPUT_PICKLE
    ),
}


with open(
    OUTPUT_SUMMARY,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2,
        ensure_ascii=False
    )


print()
print("Files created:")

print(
    f"  {OUTPUT_PICKLE.name}"
)

print(
    f"  {OUTPUT_NODES.name}"
)

print(
    f"  {OUTPUT_EDGES.name}"
)

print(
    f"  {OUTPUT_SUMMARY.name}"
)

print()
print("=" * 72)
print("GRAPH CONSTRUCTION COMPLETE")
print("=" * 72)