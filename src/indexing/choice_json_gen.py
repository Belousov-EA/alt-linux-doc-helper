import requests
from bs4 import BeautifulSoup
import base64
import re
import time
import json

BASE_URL = "https://docs.altlinux.org"
MENU = "/ru-RU/products_menu.html"

def get_soup(url):
    response = requests.get(url, timeout=5)
    response.encoding = 'utf-8'
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")

def get_target(url, source, target=None):
    soup = get_soup(url)
    menu_list = soup.find("ul", id=source)
    choices = {
        li.a.text: f"{BASE_URL}{li.a['href']}"
        for li in menu_list.find_all("li")
    }
    choices = {k.replace(u'\xa0', u' '): v for k, v in choices.items()}
    if target == None:
        return choices
    if not target in choices:
        raise Exception(f'Sorry, no "{target}" in {choices.keys()}')
    return choices[target]

def get_large_htmls(url, base_url):
    soup = get_soup(url)
    links = [a["href"] for a in soup.find_all("a", class_="html-single")]
    return [base_url + link for link in links]

def extract_choice(url):
    match = re.search(r"/ru-RU/([^/]+)/index\.html", url)
    return match.group(1) if match else None

def get_large_htmls_for_all(base_url, menu_path):
    result = get_target(base_url + menu_path, 'product_menu_list')
    for key, value in result.items():
        choice = extract_choice(value)
        MENU_VERSION = f"/ru-RU/{choice}/versions_menu.html"
        item_result = get_target(base_url + MENU_VERSION, 'version_menu_list')
        for item_key, item_vallue in item_result.items():
            try:
                item_result[item_key] = get_large_htmls(item_vallue, base_url)
            except Exception as e:
                print(e)
        result[key] = item_result
    return result

def main():
    possible_choices_file_path = 'generated_configs/possible_choices.json'
    your_choice_file_path = 'generated_configs/your_choice.json'
    with open(possible_choices_file_path, 'w', encoding='utf-8') as f:
        json.dump(get_large_htmls_for_all(BASE_URL, MENU), f, ensure_ascii=False, indent=4)
    print(f"Файл возможных выборов '{possible_choices_file_path}' сгенерирован.")
    try:
        with open(your_choice_file_path, 'x', encoding='utf-8') as file:
            json.dump({"distr": "Название дистрибутива", "v": "Версия дистрибутива"}, file, ensure_ascii=False, indent=4)
            print(f"Файл пресета пользовательского выбора '{your_choice_file_path}' сгенерирован.")
    except FileExistsError:
        print(f"Файл '{your_choice_file_path}' уже существует, не генерирую пресет.")

if __name__ == "__main__":
    main()
