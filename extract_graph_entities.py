import json
import re
from pathlib import Path
from collections import defaultdict


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "processed"
    / "unified_agriculture_knowledge.jsonl"
)

OUTPUT_DIR = (
    BASE_DIR
    / "processed"
    / "graph_extraction"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

ENTITIES_FILE = (
    OUTPUT_DIR
    / "entities.jsonl"
)

RELATION_CANDIDATES_FILE = (
    OUTPUT_DIR
    / "relation_candidates.jsonl"
)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def clean_text(value):

    if value is None:
        return ""

    if isinstance(value, list):

        value = " ".join(
            str(x)
            for x in value
            if x
        )

    value = str(value)

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


def normalize_entity_name(value):

    value = clean_text(value)

    if not value:
        return ""

    value = value.lower()

    # Normalize unicode apostrophes
    value = value.replace(
        "’",
        "'"
    )

    # Remove surrounding punctuation
    value = value.strip(
        " .,;:!?\"'()[]{}"
    )

    # Normalize spaces
    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value


# ============================================================
# ENTITY ID
# ============================================================

def make_entity_id(
    entity_type,
    name
):

    normalized = normalize_entity_name(
        name
    )

    return (
        f"{entity_type.upper()}:"
        f"{normalized}"
    )


# ============================================================
# ENTITY STORE
# ============================================================

entities = {}


def add_entity(
    entity_type,
    name,
    provenance,
    properties=None
):

    name = clean_text(name)

    if not name:
        return ""

    normalized = normalize_entity_name(
        name
    )

    if not normalized:
        return ""

    entity_id = make_entity_id(
        entity_type,
        normalized
    )

    if entity_id not in entities:

        entities[entity_id] = {

            "entity_id":
                entity_id,

            "entity_type":
                entity_type,

            "name":
                name,

            "normalized_name":
                normalized,

            "properties":
                properties or {},

            "provenance":
                []
        }

    # --------------------------------------------------------
    # Add provenance
    # --------------------------------------------------------

    if provenance:

        if provenance not in entities[
            entity_id
        ]["provenance"]:

            entities[
                entity_id
            ]["provenance"].append(
                provenance
            )

    # --------------------------------------------------------
    # Merge properties
    # --------------------------------------------------------

    if properties:

        for key, value in properties.items():

            if not value:
                continue

            if key not in entities[
                entity_id
            ]["properties"]:

                entities[
                    entity_id
                ]["properties"][key] = value

            elif (
                entities[
                    entity_id
                ]["properties"][key]
                != value
            ):

                # Preserve multiple values
                existing = entities[
                    entity_id
                ]["properties"][key]

                if not isinstance(
                    existing,
                    list
                ):

                    existing = [
                        existing
                    ]

                if isinstance(
                    value,
                    list
                ):

                    existing.extend(
                        value
                    )

                else:

                    existing.append(
                        value
                    )

                # Remove duplicates
                existing = list(
                    dict.fromkeys(
                        existing
                    )
                )

                entities[
                    entity_id
                ]["properties"][key] = existing

    return entity_id


# ============================================================
# RELATIONSHIP CANDIDATES
# ============================================================

relation_candidates = {}


def add_relation_candidate(
    source_entity,
    relation,
    target_entity,
    provenance
):

    if not source_entity:
        return

    if not target_entity:
        return

    key = (
        source_entity,
        relation,
        target_entity
    )

    if key not in relation_candidates:

        relation_candidates[key] = {

            "source":
                source_entity,

            "relation":
                relation,

            "target":
                target_entity,

            "provenance":
                []
        }

    if provenance not in relation_candidates[
        key
    ]["provenance"]:

        relation_candidates[
            key
        ]["provenance"].append(
            provenance
        )


# ============================================================
# PROVENANCE
# ============================================================

def make_provenance(
    record
):

    return {

        "source":
            record.get(
                "source",
                ""
            ),

        "source_id":
            record.get(
                "source_id",
                ""
            ),

        "record_id":
            record.get(
                "record_id",
                ""
            ),

        "category":
            record.get(
                "category",
                ""
            )
    }


# ============================================================
# AGMMU EXTRACTION
# ============================================================

def extract_agmmu_record(
    record
):

    category = clean_text(
        record.get(
            "category",
            ""
        )
    )

    category_normalized = (
        normalize_entity_name(
            category
        )
    )

    answer = clean_text(
        record.get(
            "answer",
            ""
        )
    )

    if not answer:
        return

    provenance = make_provenance(
        record
    )

    # --------------------------------------------------------
    # IMAGE CASE
    # --------------------------------------------------------

    source = clean_text(
        record.get(
            "source",
            ""
        )
    )

    source_id = clean_text(
        record.get(
            "source_id",
            ""
        )
    )

    if source and source_id:

        case_name = (
            f"{source}_{source_id}"
        )

        case_id = add_entity(
            "ImageCase",
            case_name,
            provenance,
            {
                "source":
                    source,

                "source_id":
                    source_id,

                "image_reference":
                    record.get(
                        "image_reference",
                        ""
                    )
            }
        )

    else:

        case_id = ""


    # --------------------------------------------------------
    # SPECIES
    # --------------------------------------------------------

    if category_normalized == "species":

        crop_id = add_entity(
            "Crop",
            answer,
            provenance
        )

        add_relation_candidate(
            case_id,
            "HAS_CROP",
            crop_id,
            provenance
        )

        return


    # --------------------------------------------------------
    # DISEASE
    # --------------------------------------------------------

    if (
        category_normalized
        == "disease/issue identification"
    ):

        disease_id = add_entity(
            "Disease",
            answer,
            provenance
        )

        add_relation_candidate(
            case_id,
            "HAS_DISEASE",
            disease_id,
            provenance
        )

        return


    # --------------------------------------------------------
    # SYMPTOM
    # --------------------------------------------------------

    if (
        category_normalized
        == "symptom description"
    ):

        symptom_id = add_entity(
            "Symptom",
            answer,
            provenance
        )

        add_relation_candidate(
            case_id,
            "HAS_SYMPTOM",
            symptom_id,
            provenance
        )

        return


    # --------------------------------------------------------
    # MANAGEMENT
    # --------------------------------------------------------

    if (
        category_normalized
        == "management instructions"
    ):

        management_id = add_entity(
            "Management",
            answer,
            provenance
        )

        add_relation_candidate(
            case_id,
            "HAS_MANAGEMENT",
            management_id,
            provenance
        )

        return


    # --------------------------------------------------------
    # PEST
    # --------------------------------------------------------

    if category_normalized == "insect/pest":

        pest_id = add_entity(
            "Pest",
            answer,
            provenance
        )

        add_relation_candidate(
            case_id,
            "HAS_PEST",
            pest_id,
            provenance
        )

        return


    # --------------------------------------------------------
    # IMAGE DESCRIPTION
    # --------------------------------------------------------

    if (
        category_normalized
        == "image description"
    ):

        feature_id = add_entity(
            "VisualFeature",
            answer,
            provenance
        )

        add_relation_candidate(
            case_id,
            "HAS_VISUAL_FEATURE",
            feature_id,
            provenance
        )


# ============================================================
# MIRAGE EXTRACTION
# ============================================================

def extract_mirage_record(
    record
):

    provenance = make_provenance(
        record
    )

    source = clean_text(
        record.get(
            "source",
            ""
        )
    )

    source_id = clean_text(
        record.get(
            "source_id",
            ""
        )
    )

    # --------------------------------------------------------
    # IMAGE CASE
    # --------------------------------------------------------

    if source and source_id:

        case_name = (
            f"{source}_{source_id}"
        )

        case_id = add_entity(
            "ImageCase",
            case_name,
            provenance,
            {
                "source":
                    source,

                "source_id":
                    source_id
            }
        )

    else:

        case_id = ""


    # --------------------------------------------------------
    # MIRAGE ENTITY
    # --------------------------------------------------------

    entity_name = clean_text(
        record.get(
            "entity_name",
            ""
        )
    )

    entity_type = (
        normalize_entity_name(
            record.get(
                "entity_type",
                ""
            )
        )
    )

    scientific_name = clean_text(
        record.get(
            "scientific_name",
            ""
        )
    )

    common_names = record.get(
        "common_names",
        []
    )

    if not isinstance(
        common_names,
        list
    ):

        common_names = [
            clean_text(
                common_names
            )
        ]

    common_names = [
        clean_text(x)
        for x in common_names
        if clean_text(x)
    ]

    # --------------------------------------------------------
    # Determine graph entity type
    # --------------------------------------------------------

    graph_entity_type = None

    if entity_type == "plant":

        graph_entity_type = "Crop"

    elif entity_type == "plant disease":

        graph_entity_type = "Disease"

    elif entity_type == "insect/pest":

        graph_entity_type = "Pest"

    if (
        graph_entity_type
        and entity_name
    ):

        entity_id = add_entity(

            graph_entity_type,

            entity_name,

            provenance,

            {
                "scientific_name":
                    scientific_name,

                "common_names":
                    common_names
            }
        )

        add_relation_candidate(
            case_id,
            "MENTIONS",
            entity_id,
            provenance
        )


# ============================================================
# GROUPED CASE RELATIONSHIPS
# ============================================================

def infer_agmmu_case_relations(
    records
):

    grouped = defaultdict(list)

    for record in records:

        if record.get(
            "source"
        ) != "AgMMU":

            continue

        key = (

            record.get(
                "source",
                ""
            ),

            record.get(
                "source_id",
                ""
            )
        )

        grouped[key].append(
            record
        )

    print(
        f"AgMMU evidence cases: "
        f"{len(grouped):,}"
    )

    for (
        source_key,
        case_records
    ) in grouped.items():

        crops = []
        diseases = []
        symptoms = []
        pests = []
        management = []
        visual_features = []

        case_id = ""

        for record in case_records:

            category = (
                normalize_entity_name(
                    record.get(
                        "category",
                        ""
                    )
                )
            )

            answer = clean_text(
                record.get(
                    "answer",
                    ""
                )
            )

            if not answer:
                continue

            provenance = make_provenance(
                record
            )

            source = record.get(
                "source",
                ""
            )

            source_id = record.get(
                "source_id",
                ""
            )

            case_id = make_entity_id(
                "ImageCase",
                f"{source}_{source_id}"
            )

            if category == "species":

                crops.append(
                    (
                        answer,
                        provenance
                    )
                )

            elif (
                category
                == "disease/issue identification"
            ):

                diseases.append(
                    (
                        answer,
                        provenance
                    )
                )

            elif (
                category
                == "symptom description"
            ):

                symptoms.append(
                    (
                        answer,
                        provenance
                    )
                )

            elif category == "insect/pest":

                pests.append(
                    (
                        answer,
                        provenance
                    )
                )

            elif (
                category
                == "management instructions"
            ):

                management.append(
                    (
                        answer,
                        provenance
                    )
                )

            elif (
                category
                == "image description"
            ):

                visual_features.append(
                    (
                        answer,
                        provenance
                    )
                )

        # ----------------------------------------------------
        # Disease → Crop
        # ----------------------------------------------------

        for crop, crop_prov in crops:

            crop_id = add_entity(
                "Crop",
                crop,
                crop_prov
            )

            for disease, disease_prov in diseases:

                disease_id = add_entity(
                    "Disease",
                    disease,
                    disease_prov
                )

                provenance = {

                    "source":
                        source_key[0],

                    "source_id":
                        source_key[1],

                    "reason":
                        "same_evidence_case"
                }

                add_relation_candidate(
                    crop_id,
                    "AFFECTED_BY",
                    disease_id,
                    provenance
                )

        # ----------------------------------------------------
        # Disease → Symptom
        # ----------------------------------------------------

        for disease, disease_prov in diseases:

            disease_id = add_entity(
                "Disease",
                disease,
                disease_prov
            )

            for symptom, symptom_prov in symptoms:

                symptom_id = add_entity(
                    "Symptom",
                    symptom,
                    symptom_prov
                )

                provenance = {

                    "source":
                        source_key[0],

                    "source_id":
                        source_key[1],

                    "reason":
                        "same_evidence_case"
                }

                add_relation_candidate(
                    disease_id,
                    "HAS_SYMPTOM",
                    symptom_id,
                    provenance
                )

        # ----------------------------------------------------
        # Disease → Management
        # ----------------------------------------------------

        for disease, disease_prov in diseases:

            disease_id = add_entity(
                "Disease",
                disease,
                disease_prov
            )

            for action, action_prov in management:

                management_id = add_entity(
                    "Management",
                    action,
                    action_prov
                )

                provenance = {

                    "source":
                        source_key[0],

                    "source_id":
                        source_key[1],

                    "reason":
                        "same_evidence_case"
                }

                add_relation_candidate(
                    disease_id,
                    "MANAGED_BY",
                    management_id,
                    provenance
                )

        # ----------------------------------------------------
        # Disease → Visual Feature
        # ----------------------------------------------------

        for disease, disease_prov in diseases:

            disease_id = add_entity(
                "Disease",
                disease,
                disease_prov
            )

            for feature, feature_prov in visual_features:

                feature_id = add_entity(
                    "VisualFeature",
                    feature,
                    feature_prov
                )

                provenance = {

                    "source":
                        source_key[0],

                    "source_id":
                        source_key[1],

                    "reason":
                        "same_evidence_case"
                }

                add_relation_candidate(
                    disease_id,
                    "HAS_VISUAL_FEATURE",
                    feature_id,
                    provenance
                )

        # ----------------------------------------------------
        # Crop → Pest
        # ----------------------------------------------------

        for crop, crop_prov in crops:

            crop_id = add_entity(
                "Crop",
                crop,
                crop_prov
            )

            for pest, pest_prov in pests:

                pest_id = add_entity(
                    "Pest",
                    pest,
                    pest_prov
                )

                provenance = {

                    "source":
                        source_key[0],

                    "source_id":
                        source_key[1],

                    "reason":
                        "same_evidence_case"
                }

                add_relation_candidate(
                    crop_id,
                    "ATTACKED_BY",
                    pest_id,
                    provenance
                )


# ============================================================
# SAVE
# ============================================================

def save_entities():

    with open(
        ENTITIES_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for entity in entities.values():

            f.write(
                json.dumps(
                    entity,
                    ensure_ascii=False
                )
                + "\n"
            )


def save_relations():

    with open(
        RELATION_CANDIDATES_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for relation in relation_candidates.values():

            f.write(
                json.dumps(
                    relation,
                    ensure_ascii=False
                )
                + "\n"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print(" GRAPH ENTITY AND RELATION EXTRACTION")
    print("=" * 70)

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    records = []

    print(
        f"\nLoading:\n{INPUT_FILE}"
    )

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if line:

                records.append(
                    json.loads(line)
                )

    print(
        f"Records loaded: "
        f"{len(records):,}"
    )

    # --------------------------------------------------------
    # First-pass entity extraction
    # --------------------------------------------------------

    print(
        "\nExtracting entities..."
    )

    for record in records:

        source = record.get(
            "source",
            ""
        )

        if source == "AgMMU":

            extract_agmmu_record(
                record
            )

        elif source == "MIRAGE":

            extract_mirage_record(
                record
            )

    # --------------------------------------------------------
    # Case-level relationship extraction
    # --------------------------------------------------------

    print(
        "\nExtracting case-level relationships..."
    )

    infer_agmmu_case_relations(
        records
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_entities()

    save_relations()

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    from collections import Counter

    entity_counter = Counter(
        entity["entity_type"]
        for entity in entities.values()
    )

    relation_counter = Counter(
        relation["relation"]
        for relation in relation_candidates.values()
    )

    print("\n" + "=" * 70)
    print("EXTRACTION STATISTICS")
    print("=" * 70)

    print(
        f"\nUnique entities: "
        f"{len(entities):,}"
    )

    print(
        f"Relation candidates: "
        f"{len(relation_candidates):,}"
    )

    print("\nEntities by type:")

    for entity_type, count in (
        entity_counter.most_common()
    ):

        print(
            f"  {entity_type}: {count:,}"
        )

    print("\nRelations:")

    for relation, count in (
        relation_counter.most_common()
    ):

        print(
            f"  {relation}: {count:,}"
        )

    print("\nOutput:")

    print(
        f"Entities:\n{ENTITIES_FILE}"
    )

    print(
        f"Relations:\n{RELATION_CANDIDATES_FILE}"
    )

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":

    main()