import argparse
import json
import os
from pathlib import Path
import sys

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import pandas as pd

from utils.config_loader import load_configuration
from utils.model_loader import create_embed_model


def get_parser_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Chroma vectorstore from room-level caption files.")
    parser.add_argument("--config_path", default=Path("PCGrounder/configs/scanrefer.yaml"), help="Path to config")
    parser.add_argument(
        "--input-format",
        choices=["json", "csv", "txt"],
        default="json",
        help="Input format for documents in input folder.",
    )
    return parser.parse_args()


def load_text_documents_by_file(input_folder: str) -> list[tuple[str, list[Document]]]:
    if not os.path.isdir(input_folder):
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    txt_files = sorted(
        [
            os.path.join(input_folder, name)
            for name in os.listdir(input_folder)
            if name.lower().endswith(".txt")
        ]
    )

    if not txt_files:
        raise ValueError(f"No .txt files found in input folder: {input_folder}")

    docs_by_file: list[tuple[str, list[Document]]] = []
    total_lines = 0

    for txt_path in txt_files:
        with open(txt_path, "r", encoding="utf-8") as file:
            lines = [line.strip() for line in file if line.strip()]

        if not lines:
            continue

        total_lines += len(lines)
        full_text = "\n".join(lines)
        collection_name = Path(txt_path).stem
        documents = [Document(page_content=full_text, metadata={"source": txt_path})]
        docs_by_file.append((collection_name, documents))

    if not docs_by_file:
        raise ValueError("All .txt files in the input folder are empty.")

    print(
        f"Loaded {len(docs_by_file)} text files from '{input_folder}' with {total_lines} total descriptions"
    )
    return docs_by_file

def load_csv_documents(input_folder: str) -> list[tuple[str, list[Document]]]:
    if not os.path.isdir(input_folder):
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    csv_files = sorted(
        [
            os.path.join(input_folder, name)
            for name in os.listdir(input_folder)
            if name.lower().endswith(".csv")
        ]
    )

    if not csv_files:
        raise ValueError(f"No .csv files found in input folder: {input_folder}")

    docs_by_file: list[tuple[str, list[Document]]] = []
    total_rows = 0
    required_columns = {"bbox_id", "description"}

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"CSV '{csv_path}' is missing required columns: {sorted(missing)}")

        docs: list[Document] = []
        for _, row in df.iterrows():
            docs.append(
                Document(
                    page_content=str(row["description"]),
                    metadata={"source": csv_path, "bbox_id": str(row["bbox_id"])}
                )
            )

        if not docs:
            continue

        total_rows += len(docs)
        collection_name = Path(csv_path).stem
        docs_by_file.append((collection_name, docs))

    if not docs_by_file:
        raise ValueError("All .csv files in the input folder are empty.")

    print(
        f"Loaded {len(docs_by_file)} csv files from '{input_folder}' with {total_rows} total rows"
    )
    return docs_by_file


def load_json_documents(input_folder: str) -> list[tuple[str, list[Document]]]:
    if not os.path.isdir(input_folder):
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    json_files = sorted(
        [
            os.path.join(input_folder, name)
            for name in os.listdir(input_folder)
            if name.lower().endswith(".json")
        ]
    )

    if not json_files:
        raise ValueError(f"No .json files found in input folder: {input_folder}")

    docs_by_file: list[tuple[str, list[Document]]] = []
    total_rows = 0

    for json_path in json_files:
        with open(json_path, "r", encoding="utf-8") as file:
            payload = json.load(file)

        if not isinstance(payload, list):
            raise ValueError(f"JSON '{json_path}' must be a list of objects")

        docs: list[Document] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            # try to get description field otherwise fallback to the target field to get mask3d label
            description = item.get("description") or item.get("target")
            if not description:
                continue

            metadata = {"source": json_path}
            if "bbox_id" in item:
                metadata["bbox_id"] = str(item["bbox_id"])
            if "target" in item:
                metadata["target"] = str(item["target"])

            docs.append(
                Document(
                    page_content=str(description),
                    metadata=metadata,
                )
            )

        if not docs:
            continue

        total_rows += len(docs)
        collection_name = Path(json_path).stem
        docs_by_file.append((collection_name, docs))

    if not docs_by_file:
        raise ValueError("All .json files in the input folder are empty.")

    print(
        f"Loaded {len(docs_by_file)} json files from '{input_folder}' with {total_rows} total rows"
    )
    return docs_by_file

def split_documents(documents: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)


def build_vectorstore(
    documents: list[Document],
    embeddings: OllamaEmbeddings,
    persist_directory: str,
    collection_name: str,
) -> Chroma:
    os.makedirs(persist_directory, exist_ok=True)

    vectorstore = Chroma(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_function=embeddings,
    )

    try:
        existing_count = vectorstore._collection.count()
    except Exception:
        existing_count = 0

    if existing_count > 0:
        print(f"Collection '{collection_name}' already has {existing_count} vectors. Skip indexing.")
        return vectorstore

    vectorstore.add_documents(documents)

    print(f"Created ChromaDB collection '{collection_name}' with {len(documents)} documents.")
    return vectorstore


def load_documents(input_folder: str, input_format: str) -> list[tuple[str, list[Document]]]:
    if input_format == "json":
        return load_json_documents(input_folder)
    if input_format == "csv":
        return load_csv_documents(input_folder)
    return load_text_documents_by_file(input_folder)


def main() -> None:
    args = get_parser_args()
    config = load_configuration(yaml_path=args.config_path)

    data = config.experiment.data
    input_folder = str(data.open_vocab_captions_dir)
    persist_dir = str(data.vectorstore_dir)

    print(f"input_folder: {input_folder}")
    print(f"output_folder: {persist_dir}")

    embeddings = create_embed_model(config.embedding_model)
    docs_by_file = load_documents(input_folder, args.input_format)

    for collection_name, docs in docs_by_file:
        pages_split = split_documents(docs)
        build_vectorstore(
            documents=pages_split,
            embeddings=embeddings,
            persist_directory=persist_dir,
            collection_name=collection_name,
        )
        print(f"Finished processing collection '{collection_name}' with {len(pages_split)} split documents.")


if __name__ == "__main__":
    main()
