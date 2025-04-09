import os
import json
import torch
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from qdrant_client.http.models import PointStruct
from qdrant_client.models import VectorParams, Distance
from qdrant_client.local.qdrant_local import QdrantLocal

from gme_inference import GmeQwen2VL

class AltRetriever(BaseRetriever):
    your_choice_path: str
    db_path: str
    k: int = 5
    model_name: str = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"
    collection_name: str = ""
    gme: any = None
    emb_size: int = 0
    client: any = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        with open(self.your_choice_path, encoding='utf-8') as f:
            your_choice = json.load(f)

        self.collection_name = f'{your_choice["distr"]} {your_choice["v"]}'

        self.gme = GmeQwen2VL(self.model_name)
        self.emb_size = self.gme.base.model.config.hidden_size

        self.client = QdrantLocal(self.db_path)
        if not self.client.collection_exists(collection_name=self.collection_name):
            raise ValueError(f"Коллекция батчей по документации '{full_name}' не существует.")

    def _get_vector(self, query: str) -> List[float]:
        embedding = self.gme.get_fused_embeddings(
            texts=[query], images=None, show_progress_bar=False
        ).squeeze(0).tolist()
        return embedding

    def _retrieve_neighbors(self, results) -> List:
        neighbor_ids = set()
        for result in results:
            center_id = result.id
            neighbor_ids.update([center_id - 1, center_id + 1])

        existing_ids = {res.id for res in results}
        neighbor_ids -= existing_ids

        if neighbor_ids:
            return self.client.retrieve(
                collection_name=self.collection_name,
                ids=list(neighbor_ids),
                with_payload=True,
            )
        return []

    def _combine_results(self, results) -> List[str]:
        prev_id = None
        concat = []
        scores = []
        begin = None

        sorted_results = sorted(results, key=lambda x: x.id)

        for i in sorted_results:
            if i.id - 1 != prev_id:
                concat.append("")
                begin = i.payload["begin"]
                scores.append(0)
            if hasattr(i, 'score') and i.score > scores[-1]:
                scores[-1] = i.score
            concat[-1] += i.payload["text"][begin - i.payload["begin"]:]
            begin = i.payload["end"]
            prev_id = i.id

        return [text.strip() for text in concat]

    def _get_relevant_documents(self, query: str) -> List[Document]:
        query_vector = self._get_vector(query)
        top_k_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=self.k,
            with_payload=True,
        )
        neighbors = self._retrieve_neighbors(top_k_results)
        all_results = top_k_results + neighbors

        documents = []
        combined_texts = self._combine_results(all_results)

        for text in combined_texts:
            documents.append(Document(page_content=text))

        return documents
