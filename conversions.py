"""
conversions.py
================
Shared conversion utilities for the annotation pipeline:

    source text  <->  Label Studio  <->  spaCy JSON  <->  GLiNER JSON

Used by ``import-export.ipynb`` to keep the notebook's code cells short.
Every function here is documented and can also be imported/used directly
in other scripts (e.g. training pipelines).

Sections (mirrors the notebook index):
    0. Generic I/O + shared helpers
    1. Source text            -> Label Studio importable JSON
    2. LS minified export     -> LS importable JSON (with annotations)
    3. LS full export         -> spaCy JSON (entities + relations)
    4. LS full export         -> GLiNER JSON
    5. GLiNER JSON            -> spaCy JSON
    6. GLiNER JSON            -> LS importable JSON (annotations)
    7. GLiNER JSON            -> LS importable JSON (predictions)
    U. Utilities (wrap/combine, split, merge, strip relations)
"""

import json
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

PathLike = Union[str, Path]


# =====================================================================
# 0. Generic I/O + shared helpers
# =====================================================================

def load_json_input(input_path: PathLike) -> List[dict]:
    """
    Load a single JSON file, or every ``*.json`` file in a folder.

    A file holding a dict is wrapped as a one-item list, so the return
    value is always a flat list of top-level objects (Label Studio
    tasks or GLiNER items).
    """
    p = Path(input_path)
    file_paths = sorted(p.glob("*.json")) if p.is_dir() else [p]

    items: List[dict] = []
    for fp in file_paths:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            items.extend(data)
        else:
            items.append(data)
    return items


def save_json(data: Any, output_path: PathLike, indent: int = 2) -> Path:
    """Write ``data`` to ``output_path`` as UTF-8 JSON, creating parent dirs."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
    return out


def get_task_id(task: dict) -> Optional[int]:
    """Return a Label Studio task's original id (full export uses 'task' or 'id')."""
    return task.get("task") or task.get("id") or task.get("inner_id")


def normalize_spans(text: str, start: Optional[int], end: Optional[int]) -> Tuple[int, int]:
    """
    Clean up a character span before it is used elsewhere:
      - clamp to the text's bounds (exports can contain out-of-range offsets)
      - trim leading/trailing whitespace so boundaries don't include stray
        spaces/newlines picked up by the labeling UI
    """
    if start is None:
        start = 0
    if end is None:
        end = start
    start = max(0, min(int(start), len(text)))
    end = max(start, min(int(end), len(text)))
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def filter_ls_tasks(
    tasks: List[dict],
    allowed_annotators: Optional[set] = None,
    allowed_task_ids: Optional[set] = None,
    allowed_updated_by: Optional[set] = None,
) -> List[dict]:
    """
    Keep only Label Studio (full-export) tasks that:
      - have at least one annotation
      - match ``allowed_task_ids`` (if given)
      - were completed by one of ``allowed_annotators`` (if given)
      - were last updated by one of ``allowed_updated_by`` (if given)

    Leave a filter as ``None`` (or an empty set) to skip it.
    Entity/relation label filtering is handled separately, per-span,
    inside the individual record converters below.
    """
    filtered = []
    for task in tasks:
        annotations = task.get("annotations", [])
        if not annotations:
            continue

        if allowed_task_ids and get_task_id(task) not in allowed_task_ids:
            continue

        if allowed_annotators:
            completed_by = annotations[0].get("completed_by")
            if completed_by not in allowed_annotators:
                continue

        if allowed_updated_by:
            updated_by = annotations[0].get("updated_by")
            if updated_by not in allowed_updated_by:
                continue

        filtered.append(task)
    return filtered


# =====================================================================
# 1. Source text -> Label Studio importable JSON
# =====================================================================

def split_text_into_segments(text: str, pattern: str = r"(?=\[\d+\])") -> List[str]:
    """
    Split a source text into segments using a lookahead regex pattern.
    Default pattern splits before numbered markers like ``[12]`` (as used
    for the Hagiographics text editions) -- adjust for other editions.
    """
    return [seg.strip() for seg in re.split(pattern, text) if seg.strip()]


def text_file_to_ls_json(
    input_file: PathLike,
    output_file: Optional[PathLike] = None,
    pattern: str = r"(?=\[\d+\])",
) -> Path:
    """
    Convert one source text file into a single Label Studio importable
    JSON file, where every segment becomes one task: ``{"data": {"text": ...}}``.
    """
    input_file = Path(input_file)
    text = input_file.read_text(encoding="utf-8")
    segments = split_text_into_segments(text, pattern)

    tasks = [{"data": {"text": seg}} for seg in segments]

    if output_file is None:
        output_file = input_file.with_name(input_file.stem + " - LS.json")
    save_json(tasks, output_file)

    print(f"Converted {len(segments)} segments to LS format and saved to {output_file}")
    return Path(output_file)


def text_file_to_individual_files(
    input_file: PathLike,
    output_dir: Optional[PathLike] = None,
    pattern: str = r"(?=\[\d+\])",
) -> Path:
    """
    Convert one source text file into many individual ``.txt`` files
    (one per segment), useful when texts need to be handled/uploaded
    one at a time rather than as a single LS import file.
    """
    input_file = Path(input_file)
    text = input_file.read_text(encoding="utf-8")
    segments = split_text_into_segments(text, pattern)

    if output_dir is None:
        output_dir = input_file.parent / f"{input_file.stem}_segments"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for i, seg in enumerate(segments, start=1):
        (output_dir / f"{input_file.stem}_{i}.txt").write_text(seg, encoding="utf-8")
        count += 1

    print(f"Converted {count} segments to individual text files and saved to {output_dir}")
    return output_dir


# =====================================================================
# 2. LS minified export -> LS importable JSON (with annotations)
# =====================================================================

def minified_ls_to_annotated_ls(
    input_path: PathLike,
    output_path: Optional[PathLike] = None,
    from_name: str = "label",
    to_name: str = "text",
    allowed_annotators: Optional[set] = None,
    allowed_ids: Optional[set] = None,
) -> Path:
    """
    Convert a *minified* Label Studio export (``{"text", "label", "annotator",
    "id", ...}`` per task -- the free-tier export format) into a full,
    importable Label Studio JSON that carries the annotated entity labels.

    ``allowed_annotators`` / ``allowed_ids`` optionally restrict which tasks
    are kept (leave as ``None``/empty set to keep everything). This is handy
    because the free version of LS can only export a whole project at once,
    so this lets you re-import just one annotator's or one task's work.
    """
    tasks = load_json_input(input_path)

    filtered = []
    for item in tasks:
        if allowed_annotators and item.get("annotator") not in allowed_annotators:
            continue
        if allowed_ids and item.get("id") not in allowed_ids:
            continue
        filtered.append(item)

    converted = []
    for task in filtered:
        text = task.get("text", "")
        labels = task.get("label", [])

        results = [
            {
                "id": str(uuid.uuid4())[:10],
                "from_name": from_name,
                "to_name": to_name,
                "type": "labels",
                "value": {
                    "start": lbl["start"],
                    "end": lbl["end"],
                    "text": lbl["text"],
                    "labels": lbl["labels"],
                },
            }
            for lbl in labels
        ]

        annotation = {
            "id": task.get("annotation_id"),
            "completed_by": task.get("annotator"),
            "result": results,
            "was_cancelled": False,
            "ground_truth": False,
            "created_at": task.get("created_at"),
            "updated_at": task.get("updated_at"),
            "lead_time": task.get("lead_time"),
        }

        converted.append({
            "id": task.get("id"),
            "data": {"text": text},
            "annotations": [annotation],
        })

    if output_path is None:
        p = Path(input_path)
        base = p.stem if p.is_file() else p.name
        output_path = Path(input_path if Path(input_path).is_dir() else p.parent) / f"{base}_converted_output.json"
    save_json(converted, output_path)

    print(f"Loaded {len(filtered)} tasks after filtering, converted and saved {len(converted)} to {output_path}")
    return Path(output_path)


# =====================================================================
# 3. LS full export -> spaCy JSON (entities + relations)
# =====================================================================

def ls_task_to_spacy_record(
    task: dict,
    entity_filter: Optional[set] = None,
    relation_filter: Optional[set] = None,
) -> Optional[dict]:
    """
    Convert one Label Studio task (full export) into a spaCy-style record:
    ``{"text", "entities": [...], "relations": [...]}``.
    ``entity_filter`` / ``relation_filter`` optionally restrict which labels
    are kept; spans/relations with other labels are dropped.
    """
    text = task.get("data", {}).get("text", "")
    annotations = task.get("annotations", [])
    if not annotations:
        return None
    results = annotations[0].get("result", []) or []

    entities = []
    id_to_entity = {}

    for r in results:
        if r.get("type") == "labels":
            value = r.get("value", {})
            labels = value.get("labels", [])
            label = labels[0] if labels else None
            if entity_filter and label not in entity_filter:
                continue
            start, end = value.get("start"), value.get("end")
            if start is None or end is None:
                continue
            start, end = normalize_spans(text, start, end)
            span_id = r.get("id")
            ent = {"start": start, "end": end, "label": label,
                   "text": value.get("text", text[start:end]), "id": span_id}
            entities.append(ent)
            if span_id:
                id_to_entity[span_id] = ent

    relations = []
    for r in results:
        if r.get("type") in {"relation", "relations"}:
            rel_labels = r.get("labels", []) or r.get("value", {}).get("labels", [])
            rel_label = rel_labels[0] if rel_labels else None
            if relation_filter and rel_label not in relation_filter:
                continue
            from_id = r.get("from_id") or r.get("value", {}).get("from_id")
            to_id = r.get("to_id") or r.get("value", {}).get("to_id")
            if from_id in id_to_entity and to_id in id_to_entity:
                relations.append({
                    "head": from_id, "child": to_id, "label": rel_label,
                    "direction": r.get("direction", r.get("value", {}).get("direction")),
                })

    return {"text": text, "entities": entities, "relations": relations}


def record_has_relevant_content(
    rec: dict, entity_filter: Optional[set], relation_filter: Optional[set]
) -> bool:
    """
    A record is "relevant" (worth keeping) once label filters are applied:
      - if an entity_filter was requested, at least one entity must remain
      - if a relation_filter was requested, at least one relation must remain
      - if neither filter was requested, any non-empty conversion counts
    """
    if entity_filter and not rec["entities"]:
        return False
    if relation_filter and not rec["relations"]:
        return False
    if not entity_filter and not relation_filter:
        return bool(rec["entities"]) or bool(rec["relations"])
    return True


def export_ls_tasks_to_spacy_files(
    input_path: PathLike,
    allowed_annotators: Optional[set] = None,
    allowed_task_ids: Optional[set] = None,
    allowed_updated_by: Optional[set] = None,
    allowed_entities: Optional[set] = None,
    allowed_relations: Optional[set] = None,
) -> Optional[Path]:
    """
    For each input JSON file (or the single input file), create a folder
    next to it named ``<stem>_spacy[_ents-<...>][_rels-<...>]`` and write
    one JSON file per task: ``{"text", "entities", "relations", "task_id"}``.

    Filtering happens in two stages before anything is written:
      1. task-level filtering (annotator / updated_by / task id)
      2. content-level filtering (entity / relation labels); tasks left
         with no relevant entities/relations after this stage are dropped
    """
    p = Path(input_path)
    file_paths = sorted(p.glob("*.json")) if p.is_dir() else [p]
    last_out_dir = None

    for fp in file_paths:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        tasks = data if isinstance(data, list) else [data]

        tasks = filter_ls_tasks(
            tasks,
            allowed_annotators=allowed_annotators,
            allowed_task_ids=allowed_task_ids,
            allowed_updated_by=allowed_updated_by,
        )

        relevant_records = []
        for idx, task in enumerate(tasks, start=1):
            rec = ls_task_to_spacy_record(task, entity_filter=allowed_entities, relation_filter=allowed_relations)
            if not rec or not record_has_relevant_content(rec, allowed_entities, allowed_relations):
                continue
            rec["task_id"] = get_task_id(task) or idx
            relevant_records.append(rec)

        base = fp.stem
        suffix_parts = []
        if allowed_entities:
            suffix_parts.append("ents-" + "-".join(sorted(str(x).replace(" ", "_") for x in allowed_entities)))
        if allowed_relations:
            suffix_parts.append("rels-" + "-".join(sorted(str(x).replace(" ", "_") for x in allowed_relations)))
        suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""
        out_dir = fp.parent / f"{base}_spacy{suffix}"

        if not relevant_records:
            print(f"Processed '{fp.name}': 0 records left after filtering, skipped creating '{out_dir}'")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        for rec in relevant_records:
            save_json(rec, out_dir / f"{base}_task_{rec['task_id']}.json")

        print(f"Processed '{fp.name}': saved {len(relevant_records)} spaCy records to '{out_dir}' "
              f"(dropped {len(tasks) - len(relevant_records)} tasks with no relevant content)")
        last_out_dir = out_dir

    return last_out_dir


# =====================================================================
# 4. LS full export -> GLiNER JSON
# =====================================================================

def ls_task_to_gliner_record(
    task: dict,
    entity_filter: Optional[set] = None,
    relation_filter: Optional[set] = None,
) -> Optional[dict]:
    """Convert one Label Studio task (full export) into a GLiNER-style record."""
    text = task.get("data", {}).get("text", "")
    ann = task.get("annotations", [])
    if not ann:
        return None
    results = ann[0].get("result", []) or []

    entities = []
    id_to_entity = {}
    for r in results:
        if r.get("type") == "labels":
            value = r.get("value", {})
            labels = value.get("labels", [])
            label = labels[0] if labels else None
            if entity_filter and label not in entity_filter:
                continue
            start, end = value.get("start"), value.get("end")
            if start is None or end is None:
                continue
            start, end = normalize_spans(text, start, end)
            ent = {"id": r.get("id"), "start": start, "end": end, "label": label,
                   "text": value.get("text", text[start:end])}
            entities.append(ent)
            if ent["id"]:
                id_to_entity[ent["id"]] = ent

    relations = []
    for r in results:
        if r.get("type") in {"relation", "relations"}:
            rel_labels = r.get("labels", []) or r.get("value", {}).get("labels", [])
            rel_label = rel_labels[0] if rel_labels else None
            if relation_filter and rel_label not in relation_filter:
                continue
            from_id = r.get("from_id") or r.get("value", {}).get("from_id")
            to_id = r.get("to_id") or r.get("value", {}).get("to_id")
            if from_id in id_to_entity and to_id in id_to_entity:
                relations.append({
                    "from_id": from_id, "to_id": to_id, "label": rel_label,
                    "direction": r.get("direction", r.get("value", {}).get("direction")),
                })

    return {"text": text, "entities": entities, "relations": relations}


def export_ls_tasks_to_gliner_file(
    input_path: PathLike,
    output_path: Optional[PathLike] = None,
    allowed_annotators: Optional[set] = None,
    allowed_task_ids: Optional[set] = None,
    allowed_updated_by: Optional[set] = None,
    allowed_entities: Optional[set] = None,
    allowed_relations: Optional[set] = None,
) -> Path:
    """Load LS full-export task(s), filter, and save as one GLiNER JSON file."""
    tasks = load_json_input(input_path)
    tasks = filter_ls_tasks(
        tasks,
        allowed_annotators=allowed_annotators,
        allowed_task_ids=allowed_task_ids,
        allowed_updated_by=allowed_updated_by,
    )

    records = []
    for t in tasks:
        rec = ls_task_to_gliner_record(t, entity_filter=allowed_entities, relation_filter=allowed_relations)
        if rec:
            records.append(rec)

    if output_path is None:
        p = Path(input_path)
        base = p.stem if p.is_file() else p.name
        output_path = (p.parent if p.is_file() else p) / f"{base}_gliner.json"
    save_json(records, output_path)

    print(f"Saved {len(records)} GLiNER records to {output_path}")
    return Path(output_path)


# =====================================================================
# 5. GLiNER JSON -> spaCy JSON
# =====================================================================

def gliner_items_to_spacy(
    items: List[dict],
    allowed_entities: Optional[set] = None,
    allowed_relations: Optional[set] = None,
) -> List[dict]:
    """Convert a list of GLiNER items into spaCy-style records (entities + relations)."""
    records = []
    for item in items:
        text = item.get("text", "")
        raw_entities = item.get("entities", []) or []
        raw_relations = item.get("relations", []) or []

        entities = []
        for i, ent in enumerate(raw_entities):
            label = ent.get("label")
            if allowed_entities and label not in allowed_entities:
                continue
            start, end = normalize_spans(text, ent.get("start", 0), ent.get("end", 0))
            ent_id = ent.get("id") or ent.get("uid") or f"ent-{i}"
            entities.append({"start": start, "end": end, "label": label,
                              "text": ent.get("text", text[start:end]), "id": ent_id})

        relations = []
        for rel in raw_relations:
            label = rel.get("label")
            if allowed_relations and label not in allowed_relations:
                continue
            relations.append({"head": rel.get("from_id"), "child": rel.get("to_id"),
                               "label": label, "direction": rel.get("direction")})

        records.append({"text": text, "entities": entities, "relations": relations})
    return records


def export_gliner_to_spacy_file(
    input_path: PathLike,
    output_path: Optional[PathLike] = None,
    allowed_entities: Optional[set] = None,
    allowed_relations: Optional[set] = None,
) -> Path:
    """Load GLiNER item(s), convert to spaCy records, and save as one JSON file."""
    items = load_json_input(input_path)
    records = gliner_items_to_spacy(items, allowed_entities=allowed_entities, allowed_relations=allowed_relations)

    if output_path is None:
        p = Path(input_path)
        base = p.stem if p.is_file() else p.name
        output_path = (p.parent if p.is_file() else p) / f"{base}_spacy.json"
    save_json(records, output_path)

    print(f"Saved {len(records)} spaCy records from GLiNER to {output_path}")
    return Path(output_path)


# =====================================================================
# 6. GLiNER JSON -> LS importable JSON (annotations)
# =====================================================================

def gliner_items_to_labelstudio_annotations(
    items: List[dict],
    from_name: str = "label",
    to_name: str = "text",
    result_origin: str = "manual",
) -> List[dict]:
    """
    Convert GLiNER items into LS importable tasks carrying **annotations**
    (i.e. treated as ground truth / already-completed work, not model
    predictions). Use this when re-importing corrected or gold-standard data.
    """
    ls_tasks = []
    for i, item in enumerate(items, start=1):
        text = item.get("text", "")
        results = []

        for ent in item.get("entities", []) or []:
            results.append({
                "id": str(uuid.uuid4())[:10], "from_name": from_name, "to_name": to_name,
                "type": "labels", "origin": result_origin,
                "value": {"start": ent["start"], "end": ent["end"],
                          "text": ent.get("text", text[ent["start"]:ent["end"]]),
                          "labels": [ent["label"]]},
            })

        for rel in item.get("relations", []) or []:
            results.append({
                "id": str(uuid.uuid4())[:10], "type": "relation", "origin": result_origin,
                "from_id": rel.get("from_id"), "to_id": rel.get("to_id"),
                "direction": rel.get("direction", "right"), "labels": [rel["label"]],
            })

        ls_tasks.append({
            "id": i,
            "data": {"text": text},
            "annotations": [{"id": None, "completed_by": None, "result": results,
                              "was_cancelled": False, "ground_truth": False}],
        })
    return ls_tasks


def export_gliner_to_ls_annotations_file(
    input_path: PathLike,
    output_path: Optional[PathLike] = None,
    from_name: str = "label",
    to_name: str = "text",
) -> Path:
    """Load GLiNER item(s) and save as one LS importable JSON file (annotations)."""
    items = load_json_input(input_path)
    ls_tasks = gliner_items_to_labelstudio_annotations(items, from_name=from_name, to_name=to_name)

    if output_path is None:
        p = Path(input_path)
        base = p.stem if p.is_file() else p.name
        output_path = (p.parent if p.is_file() else p) / f"{base}_LS_annotations.json"
    save_json(ls_tasks, output_path)

    print(f"Saved {len(ls_tasks)} Label Studio annotation tasks to {output_path}")
    return Path(output_path)


# =====================================================================
# 7. GLiNER JSON -> LS importable JSON (predictions)
# =====================================================================

def _extract_entities_relations(item: dict) -> Tuple[str, List[dict], List[dict]]:
    """
    Normalize either a flat GLiNER item (``{"text", "entities", "relations"}``)
    or an already LS-shaped item (``{"data": {"text"}, "annotations": [...]}``)
    into ``(text, entities, relations)`` in the flat GLiNER shape.
    """
    if "annotations" in item:
        text = item.get("data", {}).get("text", item.get("text", ""))
        entities, relations = [], []
        anns = item.get("annotations", []) or []
        ann = anns[0] if anns else {}
        for r in ann.get("result", []) or []:
            if r.get("type") == "labels":
                val = r.get("value", {})
                entities.append({"id": r.get("id"), "start": val.get("start"), "end": val.get("end"),
                                  "text": val.get("text", ""), "label": (val.get("labels") or [None])[0]})
            elif r.get("type") in {"relation", "relations"}:
                relations.append({"from_id": r.get("from_id"), "to_id": r.get("to_id"),
                                   "direction": r.get("direction", "right"),
                                   "label": (r.get("labels") or [None])[0]})
        return text, entities, relations

    return item.get("text", ""), item.get("entities", []) or [], item.get("relations", []) or []


def gliner_items_to_labelstudio_predictions(
    items: List[dict],
    model_version: str,
    from_name: str = "label",
    to_name: str = "text",
    result_origin: str = "prediction",
) -> List[dict]:
    """
    Convert GLiNER-style items (or already LS-shaped items) into Label Studio
    **prediction** tasks -- never annotations -- regardless of input shape.
    ``model_version`` is required and is stamped onto every prediction so you
    can tell which model/run produced it once viewed in Label Studio.
    """
    ls_tasks = []
    for i, item in enumerate(items, start=1):
        text, entities, relations = _extract_entities_relations(item)
        results = []

        index_to_ls_id: Dict[int, str] = {}
        original_id_to_ls_id: Dict[str, str] = {}

        for idx, ent in enumerate(entities):
            start, end = ent.get("start"), ent.get("end")
            if start is None or end is None or ent.get("label") is None:
                continue

            orig_id = ent.get("id")
            ls_id = str(orig_id) if orig_id else str(uuid.uuid4())[:10]
            index_to_ls_id[idx] = ls_id
            if orig_id:
                original_id_to_ls_id[str(orig_id)] = ls_id

            results.append({
                "id": ls_id, "from_name": from_name, "to_name": to_name,
                "type": "labels", "origin": result_origin,
                "value": {"start": start, "end": end,
                          "text": ent.get("text", text[start:end]), "labels": [ent["label"]]},
            })

        def _resolve_id(ref):
            if ref is None:
                return None
            if isinstance(ref, int) or (isinstance(ref, str) and ref.isdigit()):
                return index_to_ls_id.get(int(ref))
            return original_id_to_ls_id.get(str(ref), ref)

        for rel in relations:
            if rel.get("label") is None:
                continue
            from_id, to_id = _resolve_id(rel.get("from_id")), _resolve_id(rel.get("to_id"))
            if from_id is None or to_id is None:
                continue
            results.append({
                "id": str(uuid.uuid4())[:10], "type": "relation", "origin": result_origin,
                "from_id": from_id, "to_id": to_id,
                "direction": rel.get("direction", "right"), "labels": [rel["label"]],
            })

        ls_tasks.append({
            "id": i,
            "data": {"text": text},
            "predictions": [{"model_version": model_version, "result": results}],
        })
    return ls_tasks


def export_gliner_to_ls_predictions_file(
    input_path: PathLike,
    model_version: str,
    output_path: Optional[PathLike] = None,
    from_name: str = "label",
    to_name: str = "text",
) -> Path:
    """Load GLiNER item(s) and save as one LS importable JSON file (predictions)."""
    items = load_json_input(input_path)
    ls_tasks = gliner_items_to_labelstudio_predictions(items, model_version=model_version,
                                                        from_name=from_name, to_name=to_name)

    if output_path is None:
        p = Path(input_path)
        base = p.stem if p.is_file() else p.name
        output_path = (p.parent if p.is_file() else p) / f"{base}_LS_predictions.json"
    save_json(ls_tasks, output_path)

    print(f"Saved {len(ls_tasks)} Label Studio prediction tasks to {output_path}")
    return Path(output_path)


# =====================================================================
# U. Utilities
# =====================================================================

def wrap_or_combine_json(input_path: PathLike, output_path: Optional[PathLike] = None) -> Path:
    """
    Make sure a JSON file/folder is a *list* of tasks (the shape every other
    function here expects):
      - a single file holding a bare dict -> wrapped as ``[dict]``
      - a folder of files each holding a dict or list -> combined into one list
        (saved to ``<folder>/combined/combined.json`` by default)
    """
    p = Path(input_path)

    if p.is_file():
        content = p.read_text(encoding="utf-8")
        wrapped = f"[{content}]" if not content.strip().startswith("[") else content
        out = Path(output_path) if output_path else p
        out.write_text(wrapped, encoding="utf-8")
        print(f"Wrapped '{p.name}' as a list and saved to {out}")
        return out

    if p.is_dir():
        out = Path(output_path) if output_path else p / "combined" / "combined.json"
        data = []
        for fp in sorted(p.glob("*.json")):
            with open(fp, "r", encoding="utf-8") as f:
                file_data = json.load(f)
            if isinstance(file_data, list):
                data.extend(file_data)
            elif isinstance(file_data, dict):
                data.append(file_data)
            else:
                print(f"Skipping {fp.name}: unexpected top-level type {type(file_data)}")
        save_json(data, out)
        print(f"Combined {len(data)} tasks from '{p}' into {out}")
        return out

    raise FileNotFoundError(f"'{input_path}' is not a file or directory")


def split_tasks_to_files(input_path: PathLike, output_dir: Optional[PathLike] = None) -> Path:
    """
    Split a JSON file containing a list of task objects into separate JSON
    files, one per task (each task must have a ``task_id`` field). Files are
    named ``<stem>_<task_id>.json`` and placed in a folder named after the
    input file.
    """
    p = Path(input_path)
    if not p.is_file():
        raise FileNotFoundError(f"No such file: {input_path}")

    out_dir = Path(output_dir) if output_dir else p.with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected the JSON file to contain a list of task objects.")

    count = 0
    for task in data:
        if "task_id" not in task:
            print(f"Warning: skipping a task with no 'task_id': {task}")
            continue
        save_json(task, out_dir / f"{p.stem}_{task['task_id']}.json", indent=1)
        count += 1

    print(f"Done. Wrote {count} file(s) to '{out_dir}/'.")
    return out_dir


def merge_predictions_and_annotations(
    predictions_path: PathLike,
    annotations_path: PathLike,
    output_path: PathLike,
) -> Path:
    """
    Merge a Label Studio *predictions* JSON file and a Label Studio
    *annotations* JSON file into one combined importable JSON, matching
    tasks by exact text. Annotation tasks with no matching prediction text
    become their own new task rather than being silently dropped.
    """
    with open(predictions_path, "r", encoding="utf-8") as f:
        prediction_tasks = json.load(f)
    with open(annotations_path, "r", encoding="utf-8") as f:
        annotation_tasks = json.load(f)

    text_to_task: Dict[str, dict] = {}
    for task in prediction_tasks:
        text = task["data"]["text"]
        text_to_task[text] = {"data": task["data"], "predictions": task.get("predictions", []), "annotations": []}

    unmatched = 0
    for task in annotation_tasks:
        text = task["data"]["text"]
        if text in text_to_task:
            text_to_task[text]["annotations"].extend(task.get("annotations", []))
        else:
            unmatched += 1
            text_to_task[text] = {"data": task["data"], "predictions": [], "annotations": task.get("annotations", [])}

    combined_tasks = [
        {"id": i, "data": t["data"], "predictions": t["predictions"], "annotations": t["annotations"]}
        for i, (text, t) in enumerate(text_to_task.items(), start=1)
    ]
    save_json(combined_tasks, output_path)

    print(f"Saved {len(combined_tasks)} combined tasks to {output_path} ({unmatched} annotation-only tasks)")
    return Path(output_path)


def strip_relations(input_path: PathLike, output_path: Optional[PathLike] = None) -> Path:
    """
    Remove all 'relation' annotations (and predictions) from a Label Studio
    JSON file, e.g. to prepare entity-only data for LLM pre-annotation.
    """
    p = Path(input_path)
    with open(p, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    total_removed = 0
    for task in tasks:
        for ann in task.get("annotations", []):
            result = ann.get("result", [])
            kept = [r for r in result if r.get("type") not in {"relation", "relations"}]
            total_removed += len(result) - len(kept)
            ann["result"] = kept
            ann["result_count"] = len(kept)

        for pred in task.get("predictions", []):
            if not isinstance(pred, dict):
                continue
            result = pred.get("result", [])
            kept = [r for r in result if r.get("type") not in {"relation", "relations"}]
            total_removed += len(result) - len(kept)
            pred["result"] = kept

    if output_path is None:
        output_path = p.with_name(p.stem + "_no_relations.json")
    save_json(tasks, output_path)

    print(f"Removed {total_removed} relation annotations across {len(tasks)} tasks")
    print(f"Saved to {output_path}")
    return Path(output_path)
