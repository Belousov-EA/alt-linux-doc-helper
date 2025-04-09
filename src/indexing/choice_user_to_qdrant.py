import requests
import torch
import json
import tqdm
import os
import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from markdownify import markdownify as md
from qdrant_client.http.models import PointStruct
from qdrant_client.http.models import HnswConfigDiff
from qdrant_client.models import VectorParams, Distance
from qdrant_client.local.qdrant_local import QdrantLocal

from gme_inference import GmeQwen2VL

os.environ["TOKENIZERS_PARALLELISM"] = "false"

START_DIVIDE = 2048

class Chunk(object):
    def __init__(self, s, img_url=None, img_pos=None, begin=None, end=None):
        self.s = s
        self.i = 0
        self.img = img_url
        self.img_pos = img_pos
        if (begin == None and end == None):
            self.begin = 0
            self.end = len(s)
        else:
            self.begin = begin
            self.end = end

    def if_img_for_emb_view(self):
        if self.img == None:
            return [self.s.strip()], None
        right_before = self.s[self.img_pos:].find('](') + self.img_pos + 2
        right_after = self.img_pos + len(self.img) - 1
        return [(self.s[:right_before] + "<image>" + self.s[right_after:]).strip()], [self.s[right_before:right_after]]

    def split_by_png(self):
        regex = r'!\[[^\]]*\]\((https?://[^\s)]+?\.png)\)'
        matches = list(re.finditer(regex, self.s))
        if not matches:
            return [self]
        
        chunks = []
        start = 0
        pref_repl_end = 0
        link_start = matches[0].start()
        link_end = matches[0].end()
        img_url = None
        img_pos = None
        
        for i in range(len(matches) - 1):
            end = matches[i + 1].start()
            content = " " * (pref_repl_end - start) + self.s[pref_repl_end:end]
            img_url = self.s[link_start:link_end]
            img_pos = link_start - start
            chunks.append(Chunk(content, img_url, img_pos, self.begin + start, self.begin + end))
            img_url = None
            img_pos = None
            start = matches[i].start()
            pref_repl_end = matches[i].end()
            link_start = matches[i+1].start()
            link_end = matches[i+1].end()
        
        content = self.s[start:]
        img_url = self.s[link_start:link_end]
        img_pos = link_start - start
        chunks.append(Chunk(content, img_url, img_pos, self.begin + start, self.end))

        return chunks

    def split_by_const(self, max_len):
        end = len(self.s)
        if max_len >= end:
            return [self]
        even = [0]
        odd = self.s[:(max_len//2)].rfind("\n")
        if odd == -1:
            raise Exception(f"Не повезло, есть строка > {max_len}")
        odd = [odd]
        fin = 0
        not_done = True
        while not_done:
            fin = min(end, even[-1]+max_len)
            if fin == end:
                even.append(end)
                not_done = False
            else:
                div = even[-1] + self.s[even[-1]:fin].rfind("\n")
                if div == -1:
                    raise Exception(f"Не повезло, есть строка > {max_len}")
                even.append(div)
            fin = min(end, odd[-1]+max_len)
            if fin == end:
                odd.append(end)
                not_done = False
            else:
                div = odd[-1] + self.s[odd[-1]:fin].rfind("\n")
                if div == -1:
                    raise Exception(f"Не повезло, есть строка > {max_len}")
                odd.append(div)
        if even[-1] == odd[-1] == end:
            if even[-2] < odd[-2]:
                odd.pop()
            else:
                even.pop()
        result = [None]*(len(even)+len(odd))
        result[::2] = even
        result[1::2] = odd
        res = []
        for i in range(len(result)-2):
            start = result[i]
            end = result[i+2]
            img_url = None
            img_pos = None
            if self.img != None and start <= self.img_pos < end:
                img_url = self.img
                img_pos = self.img_pos - start
            res.append(Chunk(self.s[start:end], img_url, img_pos, self.begin + start, self.begin + end))
        return res

def flatten(xss):
    return [x for xs in xss for x in xs]

def to_chunks(html_content):
    result = Chunk(md(html_content)).split_by_png()

    for i in range(len(result)):
        if len(result[i].s) > START_DIVIDE:
            result[i] = result[i].split_by_const(START_DIVIDE)
        else:
            result[i] = [result[i]]
    res = flatten(result)

    result = []
    for i in res:
        if not result or i.end > result[-1].end:
            if result and i.begin <= result[-1].begin:
                result[-1] = i
            else:
                result.append(i)
    result_len = len(result)
    print(f"Создано {result_len} чанков.")
    min_i = 0
    max_i = 0
    min_len = len(result[0].s)
    max_len = len(result[0].s)
    print("№\tНачало\tКонец\tНач.и.\tИзображение")
    for i in range(result_len):
        result[i].i = i
        i_len = len(result[i].s)
        if i_len < min_len:
            min_len = i_len
            min_i = i
        elif i_len > max_len:
            max_len = i_len
            max_i = i
        if i != 0 and result[i].begin > result[i-1].end:
            print("ВНИМАНИЕ! Следующий чанк отделён от предыдущего:")
        print(f"{i}:\t{result[i].begin}\t{result[i].end}\t{result[i].img_pos}\t{result[i].img}")
    print(f"Самый короткий чанк №{min_i}: {min_len}")
    print(f"Самый длинный чанк №{max_i}: {max_len}")
    return result

def download_and_convert(url):
    response = requests.get(url)
    response.encoding = 'utf-8'
    if response.status_code != 200:
        print(f"Failed to fetch page: {response.status_code}")
        return None
    
    html_content = response.text
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    for tag in soup.find_all(['a', 'link', 'script', 'img']):
        attr = 'href' if tag.name in ['a', 'link'] else 'src'
        if tag.has_attr(attr):
            tag[attr] = urljoin(url, tag[attr])
    
    return str(soup).replace(url + '#', "")

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
    print(f"Приступаю к индексации документации '{full_name}'...")
    try:
        url = possible_choices[your_choice["distr"]][your_choice["v"]]
    except Exception as e:
        print(f"Неверно задан дистрибутив и/или версия! Необходимо заполнить '{your_choice_file_path}' в соответствии с '{possible_choices_file_path}'.")
        print(f"Не нашёл {str(e)}.")
        return
    if len(url) != 1:
        print(f"Извините, документация '{full_name}' пока не поддерживается.")
        return
    print(f"Адрес: '{url[0]}'.")
    modified_html = download_and_convert(url[0])
    chunks = None
    if modified_html:
        chunks = to_chunks(modified_html)
    else:
        print("Ошибка доступа к сайту документации!")
        return
    print("Подключение визуально-языкового эмбеддера...")
    gme = GmeQwen2VL("Alibaba-NLP/gme-Qwen2-VL-2B-Instruct")
    emb_size = gme.base.model.config.hidden_size
    print("Подключение визуально-языкового эмбеддера - OK")
    print("Подключение к qdrant...")
    client = QdrantLocal("db/")
    if client.collection_exists(collection_name=full_name) == False:
        client.create_collection(
            collection_name=full_name,
            vectors_config=VectorParams(
                size=emb_size,
                distance=Distance.COSINE,
                on_disk=True,
                hnsw_config=HnswConfigDiff(ef_construct=100, m=16, on_disk=True)),
            on_disk_payload=True
        )
    else:
        print("Подключение к qdrant - FAIL")
        print(f"Коллекция чанков по документации '{full_name}' уже существует, индексация отменена.")
        return
    print("Подключение к qdrant - OK")
    print("Индексация...")
    points = []
    for chunk in tqdm.tqdm(chunks, desc="Прогресс индексации чанков"):
        texts, images = chunk.if_img_for_emb_view()
        vector = gme.get_fused_embeddings(texts=texts, images=images, show_progress_bar=False).squeeze(0).tolist()
        torch.cuda.empty_cache()

        payload = {
            "text": chunk.s,
            "begin": chunk.begin,
            "end": chunk.end,
        }

        #if chunk.img is not None:
        #    payload["img_url"] = chunk.img
        #    payload["img_pos"] = chunk.img_pos
        if images != None:
            payload["img_url"] = images[0]

        points.append(PointStruct(id=chunk.i, vector=vector, payload=payload))

    client.upsert(
        collection_name=full_name,
        points=points
    )
    print("Индексация - OK")

if __name__ == "__main__":
    main()
