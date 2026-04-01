from langchain.tools import tool, ToolRuntime
from langchain_core.tools import InjectedToolCallId
from pydantic import BaseModel, Field
from dataclasses import dataclass
from collections import Counter
from langchain.agents.middleware import after_model, AgentState, hook_config, after_agent
from typing import Any, Optional, Annotated
from langchain.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.runtime import Runtime
from langchain_chroma import Chroma
from langgraph.types import Command
import os
import numpy as np

from inference.vg_agent.utils import image_to_base64
from inference.projection import render_point_cloud_with_pytorch3d_with_objects

class FinalAnswer(BaseModel):
    object_id: int = Field(description="The ID of the object in the image that corresponds to the query.")
    explanation: str = Field(description="A brief explanation of why this object was selected based on the query.")


@dataclass
class RoomData:
    objects: dict = Field(description="Mapping of object IDs to object metadata for the current room. Each value should describe one detected object (for example: target, bbox_id, and bbox_3d as [cx, cy, cz, dx, dy, dz]).")
    vectorstore: Optional[Chroma] = Field(description="Chroma vectorstore containing object embeddings and metadata for the current room, which can be used for similarity search based on the query.")
    image_path: Optional[str] = Field(default=None, description="Path to the room image, which can be used for visual reference if needed.")
    scan_pc: Optional[Any] = Field(default=None, description="Point cloud array used for rendering view-dependent images.")
    center: Optional[Any] = Field(default=None, description="Scene center coordinates used for rendering.")
    render_save_dir: Optional[str] = Field(default=None, description="Directory where tool-rendered images are saved.")

@after_agent
def parse_final_answer(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """
    Middleware function that parse the final answer from the agent's response into a structured format using the FinalAnswer schema. This function checks if the agent's last message contains a final answer in the expected format, and if so, it extracts the object ID and explanation from the message content and returns them in a structured format.
    """
    # find the last AIMessage in the conversatione history
    last_message = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage):
            return None
        if isinstance(msg, AIMessage):
            last_message = msg
            break
    if isinstance(last_message, AIMessage):
        content = last_message.content
        if isinstance(content, str):
            import json, re

            # First, try to find a JSON object containing object_id
            match = re.search(r"(\{\s*\"object_id\".*?\})", content, flags=re.DOTALL)
            if match:
                json_str = match.group(1)
                try:
                    parsed = json.loads(json_str)
                    object_id = int(parsed.get("object_id"))
                    explanation = parsed.get("explanation", "")
                    print(f"Final answer extracted from JSON: object_id={object_id}, explanation={explanation}")
                    return {
                        "structured_response": FinalAnswer(object_id=object_id, explanation=explanation)
                    }
                except Exception as e:
                    print(f"Error parsing JSON final answer: {e}\nJSON string was: {json_str}")

            # If no JSON found, try to extract object ID from text patterns
            # Look for patterns like "object 4", "object_id: 4", "ID 4", etc.
            # Use findall to get ALL matches, then take the last one
            id_matches = re.findall(r"object\s+(\d+)|object_id[:\s]+(\d+)|ID\s+(\d+)", content, flags=re.IGNORECASE)
            if id_matches:
                # Get the last match (which is a tuple of groups)
                last_match = id_matches[-1]
                # Get the first non-empty group from the last match
                object_id = int(next((g for g in last_match if g)))
                print(f"Final answer extracted from text: object_id={object_id}")
                return {
                    "structured_response": FinalAnswer(object_id=object_id, explanation=content)
                }
    return None


# ---------------------------------------------------------------------------
# custom middleware to warn when model call budget is nearly exhausted
# ---------------------------------------------------------------------------

@after_model
def warn_call_limit(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """After every model call, inject a human reminder when only a few calls
    remain. Piggybacks on run_model_call_count maintained by
    ModelCallLimitMiddleware — no duplicate counter needed.

    NOTE: warn_call_limit is listed before ModelCallLimitMiddleware in the
    middleware list, so its after_model hook runs first. At this point
    run_model_call_count still holds the value *before* the current call,
    so we add 1 to get the actual call number for this turn.
    """
    # run_model_call_count is managed by ModelCallLimitMiddleware.after_model,
    # which has not run yet → add 1 to get the current call number.
    run_count = state.get("run_model_call_count", 0) + 1

    # Must match the run_limit configured in ModelCallLimitMiddleware.
    run_limit = 10

    remaining = run_limit - run_count
    if 0 <= remaining <= 3:
        reminder = (
            "⚠️ Warning: you have only "
            f"{remaining} model call(s) left. Please finish your reasoning and "
            "provide a final answer in the next turn."
        )
        return {"messages": state.get("messages", []) + [HumanMessage(content=reminder)]}

    return None

@tool
def see_image(runtime: ToolRuntime[RoomData], tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Tool to see the room image. This can be used for query that depend of the view point, e.g "If you are facing the cabinets, it's the one on the right."""
    print("see_image tool called. Checking for image path in context...")
    context = runtime.context
    image_path = context.image_path
    if image_path:
        image_base64 = image_to_base64(image_path)
        image_message = ToolMessage(content=[
                {"type": "text", "text": "Here is the image."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
            ], tool_call_id=tool_call_id)
        print("Created image message")
        return Command(
            update={"messages": [image_message]},
        )
    else:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="see_image tool was called, but no image path is available in the context.",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )


@tool
def see_image_with_object_ids(
    runtime: ToolRuntime[RoomData],
    object_ids: list[int],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Render a new scene image that highlights the provided list of object IDs and return it to the model."""
    context = runtime.context
    objects = context.objects

    if not object_ids:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="see_image_with_object_ids was called with an empty object_ids list.",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    if context.scan_pc is None or context.center is None:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="see_image_with_object_ids requires scan_pc and center in context, but they are missing.",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    valid_ids = []
    targets = []
    for object_id in object_ids:
        obj = objects.get(int(object_id))
        if obj is not None:
            valid_ids.append(int(object_id))
            targets.append(obj)

    if not targets:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"None of the provided object IDs were found: {object_ids}",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    save_dir = context.render_save_dir
    if not save_dir:
        if context.image_path:
            save_dir = os.path.dirname(context.image_path)
        else:
            save_dir = "PCGrounder/logs/tool_renders"

    try:
        render_result = render_point_cloud_with_pytorch3d_with_objects(
            objects=list(objects.values()),
            targets=targets,
            anchors=[],
            center=np.asarray(context.center),
            scan_pc=np.asarray(context.scan_pc),
            save_dir=save_dir,
            image_size=680,
            draw_id=True,
            draw_img=True,
        )

        rendered_image_path = render_result[0] if isinstance(render_result, tuple) else render_result

        image_base64 = image_to_base64(rendered_image_path)
        image_message = ToolMessage(
            content=[
                {
                    "type": "text",
                    "text": f"Rendered image for object IDs: {valid_ids}.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                },
            ],
            tool_call_id=tool_call_id,
        )
        return Command(update={"messages": [image_message]})
    except Exception as error:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Failed to render image for IDs {valid_ids}: {error}",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

@tool
def write_plan(runtime: ToolRuntime[RoomData], plan: str):
    """Write a step-by-step reasoning plan before executing actions. Call this first to outline which tools to use and in what order."""
    # In a real implementation, this could write to a log file or database
    print(f"Agent's reasoning plan:\n{plan}")
    return "Plan written successfully. The plan is:\n" + plan


def get_available_object_labels(objects: dict) -> str:
    """Tool to get a list of available object labels in the current room context.
    This can help the agent understand what objects are present and their attributes.
    
    Example output: "Available object labels in the room: red car, blue chair, green table."
    """
    labels = set()
    for obj_info in objects.values():
        label = obj_info.get("target") or obj_info.get("label") or obj_info.get("type") or obj_info.get("category")
        if label:
            labels.add(label)
    return f"Available object labels in the room: {', '.join(labels)}"


def get_room_description(objects: dict) -> str:
    """Tool to generate a natural language description of the room based on the objects present.
    Example output: "The room contains a red car, a blue chair, and a green table."
    """
    def pluralize(noun: str) -> str:
        if noun.endswith(("s", "x", "z", "ch", "sh")):
            return f"{noun}es"
        if noun.endswith("y") and len(noun) > 1 and noun[-2] not in "aeiou":
            return f"{noun[:-1]}ies"
        return f"{noun}s"

    def format_count(count: int, label: str) -> str:
        return f"{count} {label if count == 1 else pluralize(label)}"
    if not objects:
        return "No objects are available for this room."

    if isinstance(objects, dict):
        entries = list(objects.values())
    elif isinstance(objects, list):
        entries = objects
    else:
        return "Room objects must be a dictionary or a list of object entries."

    labels = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        label = entry.get("target")
        if label:
            labels.append(str(label).strip().lower())

    if not labels:
        return f"The room contains {len(entries)} detected objects, but labels are missing."

    counts = Counter(labels)
    ordered = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    parts = [format_count(count, label) for label, count in ordered]

    if len(parts) == 1:
        summary = parts[0]
    elif len(parts) == 2:
        summary = f"{parts[0]} and {parts[1]}"
    else:
        summary = ", ".join(parts[:-1]) + f", and {parts[-1]}"

    return f"The room contains {len(entries)} objects: {summary}."

@tool
def query_relevant_objects(runtime: ToolRuntime[RoomData], query: str):
    """Semantic search over the vectorstore; returns top-5 objects most relevant to the query with their ID, label, and description. Use this to narrow the search space before applying spatial tools."""
    context = runtime.context
    objects = context.objects
    vectorstore = context.vectorstore

    if vectorstore is None:
        return "No vectorstore available for querying relevant objects."

    # Perform similarity search in the vectorstore based on the query
    relevant_items = vectorstore.similarity_search_with_relevance_scores(query, k=5, score_threshold=0.1)
    
    if not relevant_items:
        return "No relevant objects found for the query."

    parts = ["Relevant objects for the query:"]
    for doc, score in relevant_items:
        metadata = doc.metadata
        label = metadata.get("target", "unknown")
        object_id = metadata.get("bbox_id", "unknown")
        description = doc.page_content
        parts.append(f"Object ID: {object_id}, Label: {label}, Description: {description}, Score: {score}")

    return "\n".join(parts)

@tool
def get_ids_of_objects_with_label(runtime: ToolRuntime[RoomData], label: str):
    """Return all object IDs whose label exactly matches the given string. E.g. label='chair' → IDs: 1, 5, 9."""
    context = runtime.context
    objects = context.objects
    matching_ids = []
    for obj_id, obj_info in objects.items():
        obj_label = obj_info.get("target")
        if obj_label and str(obj_label).strip().lower() == label.strip().lower():
            matching_ids.append(str(obj_id))
    if matching_ids:
        return f"The IDs of objects matching '{label}' are: {', '.join(matching_ids)}."
    else:
        return f"No objects matching '{label}' were found in the room. The available labels are: {', '.join(set(obj_info.get('target') for obj_info in objects.values() if obj_info.get('target')))}."

@tool
def calculate_distance_between_objects(runtime: ToolRuntime[RoomData], ids1: list[int], ids2: list[int]):
    """Return pairwise Euclidean distances (meters) from each object in ids1 to each object in ids2."""
    context = runtime.context
    objects = context.objects
    if not ids1 or not ids2:
        return "Both ids1 and ids2 must be non-empty lists of object IDs."

    def get_center_and_error(object_id: int):
        obj = objects.get(object_id)
        if not obj:
            return None, f"object {object_id} not found"
        bbox = obj.get("bbox_3d")
        if not bbox:
            return None, f"object {object_id} missing 3D bounding box information"
        cx, cy, cz, _, _, _ = bbox
        return (cx, cy, cz), None

    unique_ids = set(ids1 + ids2)
    centers = {}
    errors = {}
    for object_id in unique_ids:
        center, error = get_center_and_error(object_id)
        if error:
            errors[object_id] = error
        else:
            centers[object_id] = center

    parts = ["Pairwise distances (meters):"]
    for id1 in ids1:
        for id2 in ids2:
            if id1 in errors or id2 in errors:
                reason = errors.get(id1) or errors.get(id2)
                parts.append(f"object {id1} ↔ object {id2}: unavailable ({reason}).")
                continue

            cx1, cy1, cz1 = centers[id1]
            cx2, cy2, cz2 = centers[id2]
            distance = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2 + (cz1 - cz2) ** 2) ** 0.5
            parts.append(f"object {id1} ↔ object {id2}: {distance:.2f}.")

    return "\n".join(parts)

    
@tool
def find_farthest_object(runtime: ToolRuntime[RoomData], reference_id: int, target_label: str):
    """Return the ID and distance of the farthest object matching target_label from the reference object."""
    context = runtime.context
    objects = context.objects
    reference_obj = objects.get(reference_id)
    if not reference_obj:
        return f"Reference object ID {reference_id} was not found in the room."

    ref_bbox = reference_obj.get("bbox_3d")
    if not ref_bbox:
        return f"Reference object ID {reference_id} is missing 3D bounding box information."

    cx_ref, cy_ref, cz_ref, _, _, _ = ref_bbox
    farthest_id = None
    farthest_distance = float("-inf")

    for obj_id, obj_info in objects.items():
        if obj_id == reference_id:
            continue
        obj_label = obj_info.get("target")
        if obj_label and str(obj_label).strip().lower() == target_label.strip().lower():
            bbox = obj_info.get("bbox_3d")
            if bbox:
                cx, cy, cz, _, _, _ = bbox
                distance = ((cx - cx_ref) ** 2 + (cy - cy_ref) ** 2 + (cz - cz_ref) ** 2) ** 0.5
                if distance > farthest_distance:
                    farthest_distance = distance
                    farthest_id = obj_id

    if farthest_id is not None:
        return f"The farthest '{target_label}' from object {reference_id} is object {farthest_id}, which is {farthest_distance:.2f} meters away."
    else:
        return f"No objects with label '{target_label}' were found in the room. The available labels are: {', '.join(set(obj_info.get('target') for obj_info in objects.values() if obj_info.get('target')))}."
    
@tool
def list_top_k_nearest_objects(runtime: ToolRuntime[RoomData], reference_id: int, k: int = 3):
    """Return the k nearest objects (any label) to the reference object, sorted by distance ascending."""
    context = runtime.context
    objects = context.objects
    reference_obj = objects.get(reference_id)
    if not reference_obj:
        return f"Reference object ID {reference_id} was not found in the room."

    ref_bbox = reference_obj.get("bbox_3d")
    if not ref_bbox:
        return f"Reference object ID {reference_id} is missing 3D bounding box information."

    cx_ref, cy_ref, cz_ref, _, _, _ = ref_bbox
    distances = []

    for obj_id, obj_info in objects.items():
        if obj_id == reference_id:
            continue
        bbox = obj_info.get("bbox_3d")
        if bbox:
            cx, cy, cz, _, _, _ = bbox
            distance = ((cx - cx_ref) ** 2 + (cy - cy_ref) ** 2 + (cz - cz_ref) ** 2) ** 0.5
            label = obj_info.get("target", "unknown")
            distances.append((obj_id, label, distance))

    distances.sort(key=lambda x: x[2])
    top_k = distances[:k]

    if top_k:
        parts = [f"object {obj_id} ({label}, {distance:.2f} meters)" for obj_id, label, distance in top_k]
        if len(parts) == 1:
            summary = parts[0]
        elif len(parts) == 2:
            summary = f"{parts[0]} and {parts[1]}"
        else:
            summary = ", ".join(parts[:-1]) + f", and {parts[-1]}"
        return f"The top {k} nearest objects to object {reference_id} are: {summary}."
    else:
        return f"No other objects with 3D bounding box information were found in the room."
    
@tool
def list_top_k_nearest_objects_by_label(runtime: ToolRuntime[RoomData], reference_id: int, target_label: str, k: int = 3):
    """Return the k nearest objects matching target_label to the reference object, sorted by distance ascending."""
    context = runtime.context
    objects = context.objects
    reference_obj = objects.get(reference_id)
    if not reference_obj:
        return f"Reference object ID {reference_id} was not found in the room."

    ref_bbox = reference_obj.get("bbox_3d")
    if not ref_bbox:
        return f"Reference object ID {reference_id} is missing 3D bounding box information."

    cx_ref, cy_ref, cz_ref, _, _, _ = ref_bbox
    distances = []

    for obj_id, obj_info in objects.items():
        if obj_id == reference_id:
            continue
        obj_label = obj_info.get("target")
        if obj_label and str(obj_label).strip().lower() == target_label.strip().lower():
            bbox = obj_info.get("bbox_3d")
            if bbox:
                cx, cy, cz, _, _, _ = bbox
                distance = ((cx - cx_ref) ** 2 + (cy - cy_ref) ** 2 + (cz - cz_ref) ** 2) ** 0.5
                distances.append((obj_id, distance))

    distances.sort(key=lambda x: x[1])
    top_k = distances[:k]

    if top_k:
        parts = [f"object {obj_id} ({objects[obj_id].get('target', 'unknown')}, {distance:.2f} meters)" for obj_id, distance in top_k]
        if len(parts) == 1:
            summary = parts[0]
        elif len(parts) == 2:
            summary = f"{parts[0]} and {parts[1]}"
        else:
            summary = ", ".join(parts[:-1]) + f", and {parts[-1]}"
        return f"The top {k} nearest '{target_label}' to object {reference_id} are: {summary}."
    else:
        return f"No objects with label '{target_label}' and 3D bounding box information were found in the room. The available labels are: {', '.join(set(obj_info.get('target') for obj_info in objects.values() if obj_info.get('target')))}."

@tool
def get_object_info(runtime: ToolRuntime[RoomData], object_ids: list[int]):
    """Return the label and 3D bounding box (center [cx,cy,cz] and dimensions [dx,dy,dz]) for one or more objects by ID. Pass a list, e.g. [1, 5, 9]."""
    context = runtime.context
    objects = context.objects
    parts = []
    for object_id in object_ids:
        obj_info = objects.get(object_id)
        if not obj_info:
            parts.append(f"Object ID {object_id} was not found in the room.")
            continue
        label = obj_info.get("target", "unknown")
        bbox = obj_info.get("bbox_3d")
        if bbox:
            cx, cy, cz, dx, dy, dz = bbox
            parts.append(f"Object {object_id} ({label}): center=[{cx:.2f}, {cy:.2f}, {cz:.2f}], dims=[{dx:.2f}, {dy:.2f}, {dz:.2f}].")
        else:
            parts.append(f"Object {object_id} ({label}): missing 3D bounding box.")
    return "\n".join(parts)


@tool
def get_object_info_by_labels(runtime: ToolRuntime[RoomData], labels: list[str]):
    """Return object details for one or more labels. Input example: labels=['chair', 'table'].
    Output includes object id, center (x,y,z), and dimensions (dx,dy,dz).
    Coordinate note: x,y are floor-plane coordinates; z is height."""
    context = runtime.context
    objects = context.objects

    output_string = ""

    if not labels:
        return "No labels were provided. Please pass a non-empty list, e.g. ['chair', 'table']."

    # check if labels are in the list of available labels in the room
    available_labels = set(obj_info.get("target") for obj_info in objects.values() if obj_info.get("target"))
    for label in labels:
        if label.strip().lower() not in (al.strip().lower() for al in available_labels):
            output_string += f"Warning: label '{label}' was not found in the room. Available labels are: {', '.join(available_labels)}.\n"

    # if multiple objects have the same label, return info for all of them
    for obj_id, obj_info in objects.items():
        obj_label = obj_info.get("target")
        if obj_label and any(obj_label.strip().lower() == label.strip().lower() for label in labels):
            bbox = obj_info.get("bbox_3d")
            if bbox:
                cx, cy, cz, dx, dy, dz = bbox
                output_string += f"Object {obj_id} ({obj_label}): center=[{cx:.2f}, {cy:.2f}, {cz:.2f}], dims=[{dx:.2f}, {dy:.2f}, {dz:.2f}].\n"
            else:
                output_string += f"Object {obj_id} ({obj_label}): missing 3D bounding box.\n"
                
    return output_string.strip() if output_string else "No matching objects with 3D bounding box information were found in the room."



@tool
def get_spatial_relationship(runtime: ToolRuntime[RoomData], reference_id: int, target_id: int):
    """Return the spatial relationship (left/right, front/behind, above/below) of reference object relative to target object, with axis deltas in meters."""
    context = runtime.context
    objects = context.objects
    ref = objects.get(reference_id)
    tgt = objects.get(target_id)
    if not ref:
        return f"Reference object ID {reference_id} was not found in the room."
    if not tgt:
        return f"Target object ID {target_id} was not found in the room."

    ref_bbox = ref.get("bbox_3d")
    tgt_bbox = tgt.get("bbox_3d")
    if not ref_bbox or not tgt_bbox:
        return f"One or both objects are missing 3D bounding box information."

    cx_r, cy_r, cz_r, _, _, _ = ref_bbox
    cx_t, cy_t, cz_t, _, _, _ = tgt_bbox

    dx = cx_r - cx_t  # positive → reference is to the right of target
    dy = cy_r - cy_t  # positive → reference is in front of target
    dz = cz_r - cz_t  # positive → reference is above target

    THRESHOLD = 0.2  # meters — below this difference we say "at the same level"

    relations = []
    if abs(dx) > THRESHOLD:
        relations.append("to the RIGHT of" if dx > 0 else "to the LEFT of")
    if abs(dy) > THRESHOLD:
        relations.append("in FRONT of" if dy > 0 else "BEHIND")
    if abs(dz) > THRESHOLD:
        relations.append("ABOVE" if dz > 0 else "BELOW")

    ref_label = ref.get("target", "unknown")
    tgt_label = tgt.get("target", "unknown")

    if not relations:
        rel_str = "at approximately the same position as"
    else:
        rel_str = " and ".join(relations)

    return (
        f"Object {reference_id} ({ref_label}) is {rel_str} object {target_id} ({tgt_label}). "
        f"Δx={dx:+.2f}m, Δy={dy:+.2f}m, Δz={dz:+.2f}m."
    )


@tool
def find_objects_within_radius(runtime: ToolRuntime[RoomData], reference_id: int, radius: float, label: Optional[str] = None):
    """Return all objects within `radius` meters of the reference object. Optionally filter by label."""
    context = runtime.context
    objects = context.objects
    ref = objects.get(reference_id)
    if not ref:
        return f"Reference object ID {reference_id} was not found in the room."

    ref_bbox = ref.get("bbox_3d")
    if not ref_bbox:
        return f"Reference object ID {reference_id} is missing 3D bounding box information."

    cx_r, cy_r, cz_r, _, _, _ = ref_bbox
    results = []

    for obj_id, obj_info in objects.items():
        if obj_id == reference_id:
            continue
        obj_label = obj_info.get("target", "unknown")
        if label and str(obj_label).strip().lower() != label.strip().lower():
            continue
        bbox = obj_info.get("bbox_3d")
        if bbox:
            cx, cy, cz, _, _, _ = bbox
            dist = ((cx - cx_r) ** 2 + (cy - cy_r) ** 2 + (cz - cz_r) ** 2) ** 0.5
            if dist <= radius:
                results.append((obj_id, obj_label, dist))

    if not results:
        label_str = f" matching '{label}'" if label else ""
        return f"No objects{label_str} found within {radius:.1f}m of object {reference_id}."

    results.sort(key=lambda x: x[2])
    ref_label = ref.get("target", "unknown")
    label_str = f" matching '{label}'" if label else ""
    parts = [f"object {oid} ({lbl}, {d:.2f}m)" for oid, lbl, d in results]
    return f"Objects{label_str} within {radius:.1f}m of object {reference_id} ({ref_label}): {', '.join(parts)}."


@tool
def find_nearest_object_to_group(runtime: ToolRuntime[RoomData], reference_ids: list[int], target_label: str):
    """Given a list of reference object IDs and a target label, return the target object that is closest to ANY reference object.
    Use this instead of calling list_top_k_nearest_objects_by_label in a loop over many reference IDs.
    E.g. 'window nearest the front doors': reference_ids=[door IDs], target_label='window'."""
    context = runtime.context
    objects = context.objects

    best_target_id = None
    best_distance = float("inf")
    best_ref_id = None

    for ref_id in reference_ids:
        ref = objects.get(ref_id)
        if not ref:
            continue
        ref_bbox = ref.get("bbox_3d")
        if not ref_bbox:
            continue
        cx_r, cy_r, cz_r, _, _, _ = ref_bbox

        for obj_id, obj_info in objects.items():
            if obj_id in reference_ids:
                continue
            obj_label = obj_info.get("target", "")
            if str(obj_label).strip().lower() != target_label.strip().lower():
                continue
            bbox = obj_info.get("bbox_3d")
            if not bbox:
                continue
            cx, cy, cz, _, _, _ = bbox
            dist = ((cx - cx_r) ** 2 + (cy - cy_r) ** 2 + (cz - cz_r) ** 2) ** 0.5
            if dist < best_distance:
                best_distance = dist
                best_target_id = obj_id
                best_ref_id = ref_id

    if best_target_id is not None:
        target_label_actual = objects[best_target_id].get("target", target_label)
        ref_label_actual = objects[best_ref_id].get("target", "unknown")
        return (
            f"The nearest '{target_label}' to the group {reference_ids} is object {best_target_id} "
            f"({target_label_actual}), which is {best_distance:.2f} meters from object {best_ref_id} ({ref_label_actual})."
        )
    else:
        return f"No objects with label '{target_label}' found in the room."


@tool
def find_nearest_pair_between_labels(runtime: ToolRuntime[RoomData], label_a: str, label_b: str):
    """Find the closest pair of objects between two label groups, returning the IDs and distance.
    E.g. 'window nearest the front doors' → label_a='window', label_b='door'.
    Returns the pair (obj_a_id, obj_b_id) with the minimum distance across all cross-label pairs."""
    context = runtime.context
    objects = context.objects

    group_a = [(oid, info) for oid, info in objects.items()
                if str(info.get("target", "")).strip().lower() == label_a.strip().lower() and info.get("bbox_3d")]
    group_b = [(oid, info) for oid, info in objects.items()
                if str(info.get("target", "")).strip().lower() == label_b.strip().lower() and info.get("bbox_3d")]

    if not group_a:
        return f"No objects with label '{label_a}' found in the room."
    if not group_b:
        return f"No objects with label '{label_b}' found in the room."

    best_dist = float("inf")
    best_a_id = best_b_id = None

    for a_id, a_info in group_a:
        cx_a, cy_a, cz_a, _, _, _ = a_info["bbox_3d"]
        for b_id, b_info in group_b:
            cx_b, cy_b, cz_b, _, _, _ = b_info["bbox_3d"]
            dist = ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2 + (cz_a - cz_b) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_a_id = a_id
                best_b_id = b_id

    return (
        f"The nearest pair between '{label_a}' and '{label_b}': "
        f"object {best_a_id} ({label_a}) ↔ object {best_b_id} ({label_b}), distance={best_dist:.2f}m."
    )


@tool
def get_objects_by_vertical_position(runtime: ToolRuntime[RoomData], label: str, position: str):
    """Return the object(s) with the given label at the specified vertical position: 'highest', 'lowest', or 'all' sorted top-to-bottom."""
    context = runtime.context
    objects = context.objects

    position = position.strip().lower()
    if position not in ("highest", "lowest", "all"):
        return "Invalid position. Use 'highest', 'lowest', or 'all'."

    candidates = []
    for obj_id, obj_info in objects.items():
        obj_label = obj_info.get("target", "")
        if str(obj_label).strip().lower() != label.strip().lower():
            continue
        bbox = obj_info.get("bbox_3d")
        if bbox:
            _, _, cz, _, _, _ = bbox
            candidates.append((obj_id, cz))

    if not candidates:
        return f"No objects with label '{label}' found in the room."

    candidates.sort(key=lambda x: x[1], reverse=True)  # highest z first

    if position == "highest":
        oid, cz = candidates[0]
        return f"The highest '{label}' is object {oid} at z={cz:.2f}m."
    elif position == "lowest":
        oid, cz = candidates[-1]
        return f"The lowest '{label}' is object {oid} at z={cz:.2f}m."
    else:  # all
        parts = [f"object {oid} (z={cz:.2f}m)" for oid, cz in candidates]
        return f"All '{label}' objects sorted top-to-bottom: {', '.join(parts)}."