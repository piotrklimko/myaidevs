import os
from dotenv import load_dotenv
load_dotenv()

import requests
import csv
from datetime import date
import json
from openai import OpenAI

# --- KLUCZE ---
HUB_API_KEY = os.environ["HUB_API_KEY"]
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

# --- KROK 1: Pobierz CSV ---
csv_url = f"https://hub.ag3nts.org/data/{HUB_API_KEY}/people.csv"
response = requests.get(csv_url)
response.raise_for_status()

lines = response.text.splitlines()
reader = csv.DictReader(lines)
people = list(reader)

print(f"Pobrano {len(people)} osób")

# --- KROK 2: Filtrowanie ---
current_year = date.today().year  # 2026

filtered = []
for p in people:
    # Płeć
    if p['gender'] != 'M':
        continue
    # Miasto urodzenia
    if p['birthPlace'] != 'Grudziądz':
        continue
    # Wiek 20-40 lat w 2026
    birth_year = int(p['birthDate'][:4])
    age = current_year - birth_year
    if not (20 <= age <= 40):
        continue
    filtered.append(p)

print(f"Po filtrowaniu: {len(filtered)} osób")
for p in filtered:
    print(f"  {p['name']} {p['surname']}, ur. {p['birthDate']}, praca: {p['job'][:60]}...")

# --- KROK 3: Tagowanie przez LLM ---
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

TAGS = ["IT", "transport", "edukacja", "medycyna", "praca z ludźmi", "praca z pojazdami", "praca fizyczna"]

# Budujemy numerowaną listę opisów stanowisk
jobs_list = "\n".join([f"{i+1}. {p['job']}" for i, p in enumerate(filtered)])

prompt = f"""Masz listę opisów stanowisk pracy. Przypisz do każdego odpowiednie tagi z listy.

Dostępne tagi:
- IT: programowanie, systemy informatyczne, sieci, bazy danych
- transport: przewóz towarów, logistyka, kierowcy, spedycja, zarządzanie ruchem towarowym
- edukacja: nauczanie, szkolenia, badania naukowe
- medycyna: leczenie, opieka zdrowotna, farmacja, mikrobiologia
- praca z ludźmi: obsługa klienta, HR, opieka społeczna
- praca z pojazdami: naprawa, instalacja w pojazdach, transport
- praca fizyczna: budowa, instalacje, prace manualne

Lista stanowisk:
{jobs_list}

Odpowiedz TYLKO w formacie JSON, bez żadnego tekstu przed ani po:
[
  {{"id": 1, "tags": ["tag1", "tag2"]}},
  {{"id": 2, "tags": ["tag1"]}},
  ...
]"""

response = client.chat.completions.create(
    model="anthropic/claude-3-5-haiku",
    messages=[{"role": "user", "content": prompt}],
    response_format={"type": "json_object"},
)

raw = response.choices[0].message.content
print("Odpowiedź LLM:", raw[:200], "...")

# --- KROK 4: Parsuj tagi i połącz z danymi ---
tagged = json.loads(raw)
tag_map = {item["id"]: item["tags"] for item in tagged}

# Dodaj tagi do każdej osoby i filtruj tylko transport
result = []
for i, p in enumerate(filtered):
    tags = tag_map.get(i + 1, [])
    if "transport" in tags:
        result.append({
            "name": p["name"],
            "surname": p["surname"],
            "gender": p["gender"],
            "born": int(p["birthDate"][:4]),
            "city": p["birthPlace"],
            "tags": tags
        })

print(f"\nOsoby z tagiem 'transport': {len(result)}")
for r in result:
    print(f"  {r['name']} {r['surname']} - tagi: {r['tags']}")

# --- KROK 5: Wyślij odpowiedź ---
payload = {
    "apikey": HUB_API_KEY,
    "task": "people",
    "answer": result
}

hub_response = requests.post("https://hub.ag3nts.org/verify", json=payload)
print("\nOdpowiedź hubu:", hub_response.text)