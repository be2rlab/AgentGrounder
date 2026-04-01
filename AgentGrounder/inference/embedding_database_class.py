import os
import sys
import argparse
from typing import List, Optional

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from pathlib import Path

from utils.model_loader import create_embed_model, ModelConfig
from utils.config_loader import load_configuration


class EmbeddingDatabase:
    """Manage an embedding collection and query it using an Ollama client."""

    def __init__(self, embedder: OllamaEmbeddings, vectorstore_dir: Path, collection_name: str):
        self.embedder = embedder
        self.vectorstore = self._load_room(vectorstore_dir, collection_name)

    def _load_room(self, vectorstore_dir: Path, collection_name: str) -> Chroma:
        """Load room-level Chroma collection based on room filename."""

        if not os.path.exists(vectorstore_dir):
            raise FileNotFoundError(
                f"Chroma directory not found: {vectorstore_dir}. "
                "Run prepare_data/build_vectorstore.py first."
            )

        vectorstore = Chroma(
            persist_directory=vectorstore_dir,
            collection_name=collection_name,
            embedding_function=self.embedder,
        )

        try:
            existing_count = vectorstore._collection.count()
        except Exception:
            existing_count = 0

        if existing_count <= 0:
            raise ValueError(
                f"No vectors found in ChromaDB collection '{collection_name}'. "
                "Run prepare_data/build_vectorstore.py to create/populate the vectorstore."
            )

        return vectorstore

    def get_top_k_similar_items(self, text: str, top_k: int = 1, score_threshold: Optional[float] = None) -> List[dict]:
        """Generate embedding for `text` and return top-k matching metadata with scores."""
        
        top_k = min(top_k, self.vectorstore._collection.count())
        
        if score_threshold is not None:
            docs_with_scores = self.vectorstore.similarity_search_with_relevance_scores(
                text,
                k=top_k,
            )
            docs = [doc for doc, score in docs_with_scores if score >= score_threshold]
        else:
            docs = self.vectorstore.similarity_search(text, k=top_k)

        results: List[dict] = []
        for doc in docs:
            target = doc.metadata.get("target")
            description = doc.page_content
            bbox_id = doc.metadata.get("bbox_id")
            results.append(
                {
                    "target": target,
                    "description": description,
                    "bbox_id" : bbox_id
                }
            )

        return results


def create_embedding_database(config: ModelConfig, vectorstore_dir: Path, collection_name: str) -> EmbeddingDatabase:
    return EmbeddingDatabase(
        embedder=create_embed_model(config),
        vectorstore_dir=vectorstore_dir,
        collection_name=collection_name,
    )


def get_parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default=Path('PCGrounder/configs/scanrefer.yaml'), help="Path to config")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_parser_args()
    config = load_configuration(yaml_path=args.config_path)

    room = "scene0011_00"
    data = config.experiment.data

    text = "brown wooden cabinets"

    embedding_db = create_embedding_database(
        config.embedding_model,
        vectorstore_dir=config.experiment.data.vectorstore_dir,
        collection_name=room
    )

    # results = embedding_db.get_top_k_similar_items(text, 5)
    results = embedding_db.get_top_k_similar_items(text, 40, score_threshold=0.1)

    print(f"Query: {text}")
    print("Top matching bbox IDs:")
    for res in results:
        print(res)
