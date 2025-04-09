import torch
import json
import os
from qdrant_client.http.models import PointStruct
from qdrant_client.http.models import HnswConfigDiff
from qdrant_client.models import VectorParams, Distance
from qdrant_client.local.qdrant_local import QdrantLocal

from gme_inference import GmeQwen2VL

# Используем только проц для эмбеддингов!
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    possible_choices_file_path = 'generated_configs/possible_choices.json'
    your_choice_file_path = 'generated_configs/your_choice.json'
    your_choice = None
    with open(your_choice_file_path, encoding='utf-8') as fh:
        your_choice = json.load(fh)
    possible_choices = None
    with open(possible_choices_file_path, encoding='utf-8') as fh:
        possible_choices = json.load(fh)
    url = None
    full_name = f'{your_choice["distr"]} {your_choice["v"]}'

    print("Подключение визуально-языкового эмбеддера...")
    gme = GmeQwen2VL("Alibaba-NLP/gme-Qwen2-VL-2B-Instruct")
    emb_size = gme.base.model.config.hidden_size
    print("Подключение визуально-языкового эмбеддера - OK")
    print("Подключение к qdrant...")
    client = QdrantLocal("db/")
    if client.collection_exists(collection_name=full_name) == False:
        print("Подключение к qdrant - FAIL")
        print(f"Коллекция чанков по документации '{full_name}' не существует.")
        return
    print("Подключение к qdrant - OK")
    print("Цикл запрос-ответ запущен.")
    user_input = None
    k = 5
    while True:
        user_input = input("Запрос: ")
        if user_input == 'q':
            break

        texts, images = [user_input], None
        vector = gme.get_fused_embeddings(texts=texts, images=images, show_progress_bar=False).squeeze(0).tolist()
        # Используем только проц для эмбеддингов!
        # torch.cuda.empty_cache()

        top_k_results = client.search(
            collection_name=full_name,
            query_vector=vector,
            limit=k,
            with_payload=True
        )

        neighbor_ids = set()
        for result in top_k_results:
            center_id = result.id
            neighbor_ids.update([center_id - 1, center_id + 1])

        existing_ids = {res.id for res in top_k_results}
        neighbor_ids -= existing_ids

        if neighbor_ids:
            neighbor_records = client.retrieve(
                collection_name=full_name,
                ids=list(neighbor_ids),
                with_payload=True,
            )
        else:
            neighbor_records = []

        combined_results = top_k_results + neighbor_records

        prev_id = None
        concat = []
        results = sorted(combined_results, key=lambda x: x.id)
        scores = []
        begin = None
        for i in results:
            if i.id - 1 != prev_id:
                concat.append("")
                begin = i.payload["begin"]
                scores.append(0)
            if hasattr(i, 'score') and i.score > scores[-1]:
                scores[-1] = i.score
            concat[-1] += i.payload["text"][begin - i.payload["begin"]:]
            begin = i.payload["end"]
            prev_id = i.id
        
        # Скоры кусков и сами куски
        print(scores)
        for i in concat:
            print(f"\n{i.strip()}\n")

if __name__ == "__main__":
    main()
