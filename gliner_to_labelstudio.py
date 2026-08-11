"""
Convert GLiNER2 NER output to Label Studio pre-annotation format.

This script converts GLiNER2 entity extraction results into Label Studio's
pre-annotation JSON format, including confidence scores for each prediction.
"""

import json
import uuid
from typing import Dict, List, Any, Optional
from pathlib import Path


def generate_unique_id() -> str:
    """Generate a unique ID for Label Studio results."""
    return str(uuid.uuid4())[:8]


# ============================================================================
# Schema Loading and Conversion Functions
# ============================================================================

def load_gliner_schema_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load GLiNER2 schema configuration from a JSON file.
    
    Expected JSON structure:
    {
        "schema_name": "...",
        "schema_version": "...",
        "entities": {
            "LABEL": {"description": "...", "threshold": 0.5},
            ...
        },
        "relations": {
            "RELATION": {"description": "...", "threshold": 0.5},
            ...
        }
    }
    
    Args:
        config_path: Path to the JSON schema configuration file
    
    Returns:
        Dictionary with loaded schema configuration
    
    Raises:
        FileNotFoundError: If the config file does not exist
        json.JSONDecodeError: If the JSON is invalid
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Schema config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    return config


def create_gliner_schema_from_config(extractor, config: Dict[str, Any]):
    """
    Create a GLiNER2 schema object from a configuration dictionary.
    
    Args:
        extractor: GLiNER2 extractor instance
        config: Schema configuration dictionary (from load_gliner_schema_config)
    
    Returns:
        GLiNER2 schema object ready for extraction
    
    Example:
        >>> config = load_gliner_schema_config('gliner_schema_template.json')
        >>> schema = create_gliner_schema_from_config(extractor, config)
        >>> results = extractor.extract(text, schema)
    """
    schema = extractor.create_schema()
    
    # Process entities
    if 'entities' in config and config['entities']:
        entities_dict = {}
        for entity_label, entity_config in config['entities'].items():
            if isinstance(entity_config, dict):
                # Has description and threshold
                entities_dict[entity_label] = entity_config
            else:
                # Simple string description or empty
                entities_dict[entity_label] = {
                    "description": entity_config if isinstance(entity_config, str) else "",
                    "threshold": 0.5  # default threshold
                }
        
        schema = schema.entities(entities_dict)
    
    # Process relations
    if 'relations' in config and config['relations']:
        relations_dict = {}
        for relation_label, relation_config in config['relations'].items():
            if isinstance(relation_config, dict):
                # Has description and threshold
                relations_dict[relation_label] = relation_config
            else:
                # Simple string description or empty
                relations_dict[relation_label] = {
                    "description": relation_config if isinstance(relation_config, str) else "",
                    "threshold": 0.5  # default threshold
                }
        
        schema = schema.relations(relations_dict)
    
    return schema


def create_gliner_schema_from_config_file(extractor, config_path: str | Path):
    """
    Convenience function to load schema config and create schema in one call.
    
    Args:
        extractor: GLiNER2 extractor instance
        config_path: Path to the JSON schema configuration file
    
    Returns:
        GLiNER2 schema object ready for extraction
    
    Example:
        >>> schema = create_gliner_schema_from_config_file(extractor, 'gliner_schema_template.json')
        >>> results = extractor.extract(text, schema)
    """
    config = load_gliner_schema_config(config_path)
    return create_gliner_schema_from_config(extractor, config)


def get_schema_metadata(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract metadata from schema configuration.
    
    Args:
        config: Schema configuration dictionary
    
    Returns:
        Dictionary with metadata like schema_name, version, model_version
    """
    return {
        "schema_name": config.get("schema_name", "Unnamed Schema"),
        "schema_version": config.get("schema_version", "1.0"),
        "description": config.get("description", ""),
        "model_version": config.get("model_version", "fastino/gliner2-multi-v1"),
        "num_entities": len(config.get("entities", {})),
        "num_relations": len(config.get("relations", {}))
    }


def generate_default_schema_template(output_path: str | Path) -> None:
    """
    Generate a default schema template JSON file.
    
    Args:
        output_path: Path where to save the template
    """
    template = {
        "schema_name": "My Custom NER Schema",
        "schema_version": "1.0",
        "description": "Custom entity and relation extraction schema",
        "model_version": "fastino/gliner2-multi-v1",
        "entities": {
            "ENTITY_LABEL_1": {
                "description": "Description of this entity type",
                "threshold": 0.5
            },
            "ENTITY_LABEL_2": {
                "description": "Another entity description",
                "threshold": 0.7
            }
        },
        "relations": {
            "RELATION_LABEL_1": {
                "description": "Description of this relationship type",
                "threshold": 0.5
            }
        }
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(template, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Generated default schema template at: {output_path}")


def generate_unique_id() -> str:
    """Generate a unique ID for Label Studio results."""
    return str(uuid.uuid4())[:8]


def gliner_to_labelstudio_predictions(
    gliner_output: Dict[str, Any],
    text: str,
    from_name: str = "label",
    to_name: str = "text",
    include_score: bool = True
) -> Dict[str, Any]:
    """
    Convert GLiNER2 output to Label Studio predictions format.
    Only converts entities - relations are ignored.
    
    Supports both format_results=True (entities as dict keyed by label)
    and format_results=False (entities as list of dicts with a label field).
    
    Args:
        gliner_output: Dictionary with 'entities' key containing GLiNER2 results
        text: The original text that was processed
        from_name: Label Studio control name (from labeling config)
        to_name: Label Studio data field name (from labeling config)
        include_score: Whether to include confidence scores
    
    Returns:
        Dictionary with 'result' array containing entity predictions in Label Studio format
    """
    result: List[Dict[str, Any]] = []
    scores: List[float] = []

    def pick_label(entity: Dict[str, Any], fallback: str = "UNKNOWN") -> str:
        return (
            entity.get("label")
            or entity.get("type")
            or entity.get("entity")
            or entity.get("tag")
            or fallback
        )

    def pick_text(entity: Dict[str, Any]) -> str:
        return (
            entity.get("text")
            or entity.get("span")
            or entity.get("mention")
            or entity.get("value")
            or ""
        )

    def pick_confidence(entity: Dict[str, Any]) -> float:
        return (
            entity.get("confidence")
            or entity.get("score")
            or entity.get("probability")
            or 1.0
        )

    def pick_span(entity: Dict[str, Any]) -> (Optional[int], Optional[int]):
        start = (
            entity.get("start")
            if "start" in entity else entity.get("start_char")
        )
        end = (
            entity.get("end")
            if "end" in entity else entity.get("end_char")
        )
        # Alternate keys sometimes used
        if start is None:
            start = entity.get("begin")
        if end is None:
            end = entity.get("finish")
        return start, end

    # Resolve the raw entities container
    raw_entities: Any = None
    if isinstance(gliner_output, dict):
        if "entities" in gliner_output:
            raw_entities = gliner_output["entities"]
        elif "result" in gliner_output:
            raw_entities = gliner_output["result"]
    elif isinstance(gliner_output, list):
        raw_entities = gliner_output

    # Normalize to a flat list of entity dicts with labels
    normalized: List[Dict[str, Any]] = []
    if isinstance(raw_entities, dict):
        for lbl, ents in raw_entities.items():
            ent_list = ents if isinstance(ents, list) else [ents]
            for ent in ent_list:
                if not isinstance(ent, dict):
                    continue
                ent = dict(ent)
                ent.setdefault("label", lbl)
                normalized.append(ent)
    elif isinstance(raw_entities, list):
        for ent in raw_entities:
            if not isinstance(ent, dict):
                continue

            # Case: list contains a label->entities mapping (common when format_results=True but wrapped in a list)
            if any(isinstance(v, list) for v in ent.values()) and not ("text" in ent or "start" in ent or "end" in ent):
                for lbl, ents in ent.items():
                    ent_list = ents if isinstance(ents, list) else [ents]
                    for sub_ent in ent_list:
                        if not isinstance(sub_ent, dict):
                            continue
                        sub_ent = dict(sub_ent)
                        sub_ent.setdefault("label", lbl)
                        normalized.append(sub_ent)
                continue

            # Otherwise treat as a single entity dict
            ent = dict(ent)
            ent.setdefault("label", pick_label(ent))
            normalized.append(ent)

    for entity in normalized:
        entity_type = pick_label(entity)
        entity_text = pick_text(entity)
        start, end = pick_span(entity)

        if (start is None or end is None) and entity_text:
            start = text.find(entity_text)
            end = start + len(entity_text) if start >= 0 else -1

        if start is None or end is None or start < 0 or end < 0:
            continue

        if not entity_text:
            entity_text = text[start:end]

        confidence = pick_confidence(entity)

        result_item = {
            "id": generate_unique_id(),
            "type": "labels",
            "value": {
                "start": start,
                "end": end,
                "text": entity_text,
                "labels": [entity_type]
            },
            "from_name": from_name,
            "to_name": to_name,
            "origin": "prediction"
        }

        if include_score:
            result_item["score"] = confidence
            scores.append(confidence)

        result.append(result_item)

    return {"result": result}


def create_labelstudio_task(
    text: str,
    gliner_output: Dict[str, Any],
    task_id: Optional[int] = None,
    model_version: str = "gliner2-multi-v1",
    from_name: str = "label",
    to_name: str = "text"
) -> Dict[str, Any]:
    """
    Create a complete Label Studio task with entity predictions from GLiNER2.
    Only converts entities - relations are ignored.
    Supports both format_results=True and format_results=False outputs.
    
    Args:
        text: The source text to annotate
        gliner_output: GLiNER2 extraction results with entities
        task_id: Optional task ID (if None, will be assigned by Label Studio)
        model_version: Version of the GLiNER2 model used
        from_name: Label Studio control name
        to_name: Label Studio data field name
    
    Returns:
        Dictionary representing a complete Label Studio task with entity predictions
    """
    predictions_dict = gliner_to_labelstudio_predictions(
        gliner_output, text, from_name, to_name, include_score=True
    )
    
    result = predictions_dict["result"]
    
    # Calculate average confidence score across all predictions
    avg_score = 0.0
    if result:
        scores = [r.get("score", 0.0) for r in result if "score" in r]
        avg_score = sum(scores) / len(scores) if scores else 0.0
    
    task = {
        "data": {
            "text": text
        },
        "predictions": [
            {
                "model_version": model_version,
                "score": avg_score,
                "result": result
            }
        ]
    }
    
    if task_id is not None:
        task["id"] = task_id
    
    return task


def batch_convert_gliner_to_labelstudio(
    gliner_outputs: List[Dict[str, Any]],
    texts: List[str],
    model_version: str = "gliner2-multi-v1",
    from_name: str = "label",
    to_name: str = "text"
) -> List[Dict[str, Any]]:
    """
    Convert multiple GLiNER2 outputs to Label Studio tasks.
    
    Args:
        gliner_outputs: List of GLiNER2 extraction results
        texts: Corresponding list of source texts
        model_version: Version of the GLiNER2 model used
        from_name: Label Studio control name
        to_name: Label Studio data field name
    
    Returns:
        List of Label Studio tasks ready for import
    """
    if len(gliner_outputs) != len(texts):
        raise ValueError("Number of outputs and texts must match")
    
    tasks = []
    for text, gliner_output in zip(texts, gliner_outputs):
        task = create_labelstudio_task(
            text,
            gliner_output,
            model_version=model_version,
            from_name=from_name,
            to_name=to_name
        )
        tasks.append(task)
    
    return tasks


def convert_gliner_json_file(
    input_file: str,
    output_file: str,
    model_version: str = "gliner2-multi-v1"
) -> None:
    """
    Convert GLiNER2 JSON output file to Label Studio pre-annotation format.
    
    Expects input JSON to have structure:
    {
        "text": "...",
        "entities": { "LABEL": [{...}, ...], ... }
    }
    or (with format_results=False):
    {
        "text": "...",
        "entities": [{"label": "...", "text": "...", ...}, ...]
    }
    
    Args:
        input_file: Path to GLiNER2 output JSON file
        output_file: Path to write Label Studio predictions JSON
        model_version: Version of the GLiNER2 model used
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        gliner_data = json.load(f)
    
    # Handle both single output and array of outputs
    if isinstance(gliner_data, dict) and "text" in gliner_data:
        # Single output
        outputs = [gliner_data]
    elif isinstance(gliner_data, list):
        # Array of outputs
        outputs = gliner_data
    else:
        raise ValueError("Input JSON must contain 'text' key or be an array of outputs")
    
    # Extract texts and entity data
    texts = []
    entity_outputs = []
    
    for output in outputs:
        if "text" in output:
            texts.append(output["text"])
            entity_outputs.append(output)
        else:
            raise ValueError("Each output must contain a 'text' field")
    
    # Convert to Label Studio format
    tasks = batch_convert_gliner_to_labelstudio(
        entity_outputs,
        texts,
        model_version=model_version
    )
    
    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Converted {len(tasks)} tasks to Label Studio format")
    print(f"✓ Output saved to: {output_file}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python gliner_to_labelstudio.py <input_file.json> <output_file.json>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    convert_gliner_json_file(input_path, output_path)
