import os
from dotenv import load_dotenv
load_dotenv()

import requests
import csv
import json
from datetime import date
from openai import OpenAI

HUB_API_KEY = os.environ["HUB_API_KEY"]
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

# Łączymy się przez OpenRouter — agregator API wielu modeli.
# Używamy SDK OpenAI (base_url wskazuje na OpenRouter zamiast api.openai.com),
# bo OpenRouter implementuje ten sam interfejs REST. Dzięki temu możemy
# przełączać modele (OpenAI, Anthropic, Mistral...) bez zmiany kodu.
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# --- KROK 1: Pobierz CSV ---
# Dane wejściowe to plik CSV z serwera HUB — to "nieustrukturyzowane" dane,
# które będziemy przetwarzać kodem (filtrowanie) i modelem (tagowanie).
# Klucz API identyfikuje nas wobec HUB-a i wskazuje, jaki zbiór danych pobrać.
csv_url = f"https://hub.ag3nts.org/data/{HUB_API_KEY}/people.csv"
response = requests.get(csv_url)
lines = response.text.splitlines()
reader = csv.DictReader(lines)
people = list(reader)

# --- KROK 2: Filtrowanie deterministyczne ---
# Zanim wyślemy dane do modelu, robimy co możemy kodem — filtrowanie wg
# znanych, twardych reguł (płeć, miasto, wiek) jest szybkie, tanie i pewne.
# LLM angażujemy tylko do zadań, których kod nie potrafi rozwiązać łatwo
# (np. semantyczna klasyfikacja opisu stanowiska).
current_year = date.today().year
filtered = []
for p in people:
    if p['gender'] != 'M':
        continue
    if p['birthPlace'] != 'Grudziądz':
        continue
    birth_year = int(p['birthDate'][:4])
    if not (20 <= current_year - birth_year <= 40):
        continue
    filtered.append(p)

print(f"Po filtrowaniu: {len(filtered)} osób")

# --- KROK 3: Tagowanie ze Structured Output ---
# Tutaj wchodzi LLM. Zamiast prosić o odpowiedź w JSON i liczyć na to,
# że model zastosuje właściwy format (co jest niedeterministyczne), używamy
# Structured Outputs — mechanizmu, który *gwarantuje* zgodność odpowiedzi
# z podanym JSON Schema.
#
# Jak to działa (lekcja S01E01):
# - Podajemy schemat w polu response_format.json_schema.schema
# - Model musi zwrócić JSON pasujący do tego schematu — nie może go naruszyć
# - Pole "strict": True wyłącza jakiekolwiek odstępstwa od schematu
# - Pole "enum" w tagach ogranicza dozwolone wartości do naszej listy —
#   model nie może "wymyślić" nowego tagu
#
# To jeden z kluczowych sposobów na wbudowanie LLM w logikę aplikacji:
# deterministyczny kod otrzymuje przewidywalną strukturę danych z modelu
# i może na niej operować bez dodatkowego parsowania / walidacji.
jobs_list = "\n".join([f"{i+1}. {p['job']}" for i, p in enumerate(filtered)])

VALID_TAGS = ["IT", "transport", "edukacja", "medycyna", "praca z ludźmi", "praca z pojazdami", "praca fizyczna"]

# Używamy gpt-4o-mini, bo Structured Outputs (json_schema + strict) jest funkcją
# OpenAI — modele Anthropic przez OpenRouter nie obsługują tego trybu.
# Dla modeli Claude używalibyśmy response_format={"type": "json_object"}
# i instrukcji w prompcie (jak w task_01.py).
response = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=[
        {
            # Prompt systemowy to instrukcja dla modelu — definiuje jego rolę
            # i dostępne tagi wraz z opisem. Im dokładniejsze opisy, tym
            # trafniejsza klasyfikacja (context engineering).
            "role": "system",
            "content": """Przypisujesz tagi do opisów stanowisk pracy. Użyj podanych tagów:
- IT: programowanie, systemy informatyczne, sieci, bazy danych
- transport: przewóz towarów, logistyka, kierowcy, spedycja, zarządzanie ruchem towarowym
- edukacja: nauczanie, szkolenia, badania naukowe
- medycyna: leczenie, opieka zdrowotna, farmacja, mikrobiologia
- praca z ludźmi: obsługa klienta, HR, opieka społeczna
- praca z pojazdami: naprawa, instalacja w pojazdach, transport
- praca fizyczna: budowa, instalacje, prace manualne"""
        },
        {
            # Wiadomość użytkownika zawiera dane do przetworzenia — całą listę
            # stanowisk w jednym zapytaniu (batch). To oszczędniejsze niż
            # osobne zapytanie per osoba: mniej tokenów na nagłówki/schema,
            # niższe koszty, krótszy czas odpowiedzi.
            "role": "user",
            "content": f"Przypisz tagi do każdego stanowiska:\n{jobs_list}"
        }
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "tagging_result",
            "strict": True,          # wymusza ścisłe przestrzeganie schematu
            "schema": {
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {
                                    "type": "integer",
                                    "description": "Numer stanowiska z listy"
                                },
                                "tags": {
                                    "type": "array",
                                    "items": {
                                        "type": "string",
                                        # enum = zamknięta lista dozwolonych wartości
                                        # Model nie może zwrócić żadnego innego tagu
                                        "enum": VALID_TAGS
                                    },
                                    "description": "Lista tagów pasujących do stanowiska"
                                }
                            },
                            "required": ["id", "tags"],
                            "additionalProperties": False   # brak dodatkowych pól
                        }
                    }
                },
                "required": ["results"],
                "additionalProperties": False
            }
        }
    }
)

# --- KROK 4: Parsuj i filtruj transport ---
# Dzięki Structured Outputs json.loads() nie może się tu wysypać —
# model musiał zwrócić poprawny JSON zgodny ze schematem.
# Budujemy słownik {id -> [tagi]} i łączymy go z danymi osobowymi.
data = json.loads(response.choices[0].message.content)
tag_map = {item["id"]: item["tags"] for item in data["results"]}

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

print(f"Osoby z tagiem 'transport': {len(result)}")
for r in result:
    print(f"  {r['name']} {r['surname']} - tagi: {r['tags']}")

# --- KROK 5: Wyślij ---
# Końcowy etap to weryfikacja przez HUB — wysyłamy ustrukturyzowaną odpowiedź
# (lista obiektów JSON). HUB sprawdza poprawność i zwraca wynik zadania.
payload = {
    "apikey": HUB_API_KEY,
    "task": "people",
    "answer": result
}
resp = requests.post("https://hub.ag3nts.org/verify", json=payload)
print(f"\nOdpowiedź hubu: {resp.text}")
