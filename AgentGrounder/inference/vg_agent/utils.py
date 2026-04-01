"""Utility functions for displaying messages and prompts in Jupyter notebooks."""

import base64
import json
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()


def format_message_content(message):
    """Convert message content to displayable string."""
    parts = []
    tool_calls_processed = False

    def _append_tool_call(name, args, call_id):
        parts.append(f"🔧 Tool Call: {name}")
        parts.append(f"ID: {call_id}")
        parts.append("Args:")
        parts.append(json.dumps(args, indent=2, ensure_ascii=False))

    # Handle main content
    if isinstance(message.content, str):
        text_content = message.content.strip()
        if text_content:
            parts.append(text_content)
    elif isinstance(message.content, list):
        # Handle complex content like tool calls (Anthropic format)
        for item in message.content:
            if not isinstance(item, dict):
                continue

            if item.get("type") == "text":
                text = item.get("text", "").strip()
                if text:
                    parts.append(text)
            elif item.get("type") == "tool_use":
                _append_tool_call(
                    name=item.get("name", "unknown"),
                    args=item.get("input", {}),
                    call_id=item.get("id", "N/A"),
                )
                tool_calls_processed = True
    else:
        parts.append(str(message.content))

    # Handle tool calls attached to the message (OpenAI format) - only if not already processed
    if (
        not tool_calls_processed
        and hasattr(message, "tool_calls")
        and message.tool_calls
    ):
        for tool_call in message.tool_calls:
            if isinstance(tool_call, dict):
                _append_tool_call(
                    name=tool_call.get("name", "unknown"),
                    args=tool_call.get("args", {}),
                    call_id=tool_call.get("id", "N/A"),
                )

    return "\n".join(parts).strip()


def format_messages(
    messages,
    output_file: Optional[str] = None,
    append: bool = True,
    display: bool = True,
    skip_empty: bool = True,
):
    """Format and display a list of messages with Rich formatting.

    Args:
        messages: List of LangChain messages.
        output_file: Optional path to save formatted messages as markdown.
        append: Whether to append to output_file (default: True).
        display: Whether to print to terminal (default: True).
        skip_empty: Whether to skip empty messages (default: True).
    """
    rendered_image_path = None
    if output_file:
        rendered_image_path = "./rendered.png"

    markdown_entries = []

    for index, m in enumerate(messages, start=1):
        msg_type = m.__class__.__name__.replace("Message", "")
        content = format_message_content(m)
        has_rendered_image = False

        if (
            rendered_image_path
            and msg_type == "Tool"
            and "Rendered image for object IDs" in content
        ):
            has_rendered_image = True

        if skip_empty and not content:
            continue

        markdown_entries.append(
            {
                "index": index,
                "type": msg_type,
                "content": content,
                "image_path": rendered_image_path if has_rendered_image else None,
            }
        )

        if display:
            title = f"{index:02d}"

            if msg_type == "Human":
                title = f"🧑 Human • {title}"
                panel_content = content
                border_style = "blue"
            elif msg_type == "Ai":
                title = f"🤖 Assistant • {title}"
                border_style = "green"
            elif msg_type == "Tool":
                title = f"🔧 Tool Output • {title}"
                panel_content = content
                border_style = "yellow"
            else:
                title = f"📝 {msg_type} • {title}"
                border_style = "white"

            if msg_type != "Human" and msg_type != "Tool":
                panel_content = content

            console.print(Panel(panel_content, title=title, border_style=border_style))

    if output_file:
        path = Path(output_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"

        with path.open(mode, encoding="utf-8") as file:
            for entry in markdown_entries:
                file.write(f"## {entry['index']:02d} | {entry['type']}\n\n")
                file.write("```text\n")
                file.write(entry["content"] + "\n")
                file.write("```\n\n")
                if entry["image_path"]:
                    file.write(f"![Rendered image]({entry['image_path']})\n\n")
            file.write("---\n\n")


def format_message(messages):
    """Alias for format_messages for backward compatibility."""
    return format_messages(messages)


def show_prompt(prompt_text: str, title: str = "Prompt", border_style: str = "blue"):
    """Display a prompt with rich formatting and XML tag highlighting.

    Args:
        prompt_text: The prompt string to display
        title: Title for the panel (default: "Prompt")
        border_style: Border color style (default: "blue")
    """
    # Create a formatted display of the prompt
    formatted_text = Text(prompt_text)
    formatted_text.highlight_regex(r"<[^>]+>", style="bold blue")  # Highlight XML tags
    formatted_text.highlight_regex(
        r"##[^#\n]+", style="bold magenta"
    )  # Highlight headers
    formatted_text.highlight_regex(
        r"###[^#\n]+", style="bold cyan"
    )  # Highlight sub-headers

    # Display in a panel for better presentation
    console.print(
        Panel(
            formatted_text,
            title=f"[bold green]{title}[/bold green]",
            border_style=border_style,
            padding=(1, 2),
        )
    )

# more expressive runner
async def stream_agent(agent, query, config=None):
    async for graph_name, stream_mode, event in agent.astream(
        query,
        stream_mode=["updates", "values"], 
        subgraphs=True,
        config=config
    ):
        if stream_mode == "updates":
            print(f'Graph: {graph_name if len(graph_name) > 0 else "root"}')
            
            node, result = list(event.items())[0]
            print(f'Node: {node}')
            
            for key in result.keys():
                if "messages" in key:
                    # print(f"Messages key: {key}")
                    format_messages(result[key])
                    break
        elif stream_mode == "values":
            current_state = event

    return current_state

def load_json(file_path):
    """Load data from a JSON file."""
    with open(file_path, "r") as file:
        return json.load(file)

def load_bboxes(bbox_file) -> dict:
    """Load bounding boxes (GT or predicted)."""
    bboxes = load_json(bbox_file)
    return {int(bbox["bbox_id"]): bbox for bbox in bboxes}


def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return encoded_string.decode('utf-8')