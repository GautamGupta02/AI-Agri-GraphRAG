
import sys
from pathlib import Path

PROJECT_ROOT = Path(
    "/content/drive/MyDrive/AI_Agri_GraphRAG"
)

sys.path.insert(
    0,
    str(PROJECT_ROOT / "scripts")
)

from graph_retriever import (
    GraphRetriever,
    print_retrieval_results,
)


# ============================================================
# TEST CASE
# ============================================================

query_understanding = {
    "host_or_crop": [
        "coniferous tree"
    ],

    "plant_part": [
        "trunk",
        "branches",
        "bark",
        "needles"
    ],

    "visible_damage": [],

    "visible_symptoms": [],

    "visual_features": [
        "bark with lichen or moss",
        "sunlight filtering through needles",
        "tree growing against building"
    ],

    "insect_or_pest_observation": [],

    "environmental_context": [
        "adjacent to building",
        "outdoor"
    ],

    "retrieval_terms": [
        "coniferous tree",
        "bark texture",
        "lichen growth",
        "sunlight through branches",
        "tree against building"
    ],
}


question = "What insect is causing this damage?"

qtype = "insect/pest"


# ============================================================
# LOAD RETRIEVER
# ============================================================

retriever = GraphRetriever()


# ============================================================
# RUN RETRIEVAL
# ============================================================

results = retriever.retrieve(
    query_understanding=query_understanding,
    question=question,
    qtype=qtype,
    entity_top_k=15,
    case_top_k=10,
    raw_entity_top_k=50,
)


# ============================================================
# DISPLAY
# ============================================================

print_retrieval_results(results)
