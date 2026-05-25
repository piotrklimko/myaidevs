import os
from dotenv import load_dotenv
load_dotenv()

import requests
import json
import math
from openai import OpenAI

HUB_API_KEY = os.environ["HUB_API_KEY"]
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

suspects = [
    {"name": "Cezary",   "surname": "Żurek",    "birthYear": 1987},
    {"name": "Jacek",    "surname": "Nowak",     "birthYear": 1991},
    {"name": "Oskar",    "surname": "Sieradzki", "birthYear": 1993},
    {"name": "Wojciech", "surname": "Bielik",    "birthYear": 1986},
    {"name": "Wacław",   "surname": "Jasiński",  "birthYear": 1986},
]

plant_coords = {
    "Zabrze":               {"lat": 50.3249, "lon": 18.7857, "code": "PWR3847PL"},
    "Piotrków Trybunalski": {"lat": 51.4058, "lon": 19.7028, "code": "PWR5921PL"},
    "Grudziądz":            {"lat": 53.4837, "lon": 18.7536, "code": "PWR7264PL"},
    "Tczew":                {"lat": 54.0921, "lon": 18.7763, "code": "PWR1593PL"},
    "Radom":                {"lat": 51.4027, "lon": 21.1471, "code": "PWR8406PL"},
    "Chelmno":              {"lat": 53.3496, "lon": 18.4252, "code": "PWR2758PL"},
    "Żarnowiec":            {"lat": 54.5853, "lon": 18.1749, "code": "PWR6132PL"},
}

# ============================================================
# NARZĘDZIA — prawdziwe funkcje wykonywane przez kod aplikacji
# ============================================================
#
# Kluczowa zasada lekcji S01E02: LLM fizycznie NIE może wchodzić
# w interakcję ze światem — może tylko generować tekst.
# Function Calling to konwencja, w której:
#   1. Model zamiast odpowiedzi tekstowej zwraca JSON z nazwą funkcji i argumentami
#   2. Kod aplikacji (my) odbiera ten JSON i FAKTYCZNIE wywołuje funkcję
#   3. Wynik trafia z powrotem do kontekstu modelu
#
# Poniższe funkcje to "prawdziwy świat" — HTTP, obliczenia, API.
# Model ich nie uruchamia. Model tylko o nie prosi.

def get_locations(name: str, surname: str) -> list:
    """Pobiera listę lokalizacji osoby z API HUB."""
    resp = requests.post(
        "https://hub.ag3nts.org/api/location",
        json={"apikey": HUB_API_KEY, "name": name, "surname": surname}
    )
    return resp.json()

def find_closest_plant(name: str, surname: str) -> dict:
    """Sprawdza, która elektrownia jest najbliżej jakiejkolwiek lokalizacji osoby.

    To jest przykład dobrego projektowania narzędzia dla LLM (lekcja S01E02):
    - Nie udostępniamy osobno get_locations() i haversine() — model musiałby
      rozumieć matematykę i wielokrotnie wywoływać narzędzia. Zamiast tego
      łączymy kilka kroków w JEDNO narzędzie — model pyta "kto jest najbliżej?"
      i dostaje gotową odpowiedź. Minimalizuje to liczbę kroków agenta.
    - Obliczenie odległości (Haversine) to logika deterministyczna — nie ma
      sensu, żeby model samodzielnie liczył odległości. Robimy to w kodzie.
    """
    locations = get_locations(name, surname)
    best_dist = float('inf')
    best_plant = None

    for loc in locations:
        for city, plant in plant_coords.items():
            # Wzór Haversine — odległość między dwoma punktami na sferze (Ziemia).
            # Wynik w kilometrach. LLM nie mógłby tego obliczyć deterministycznie.
            R = 6371
            dlat = math.radians(plant["lat"] - loc["latitude"])
            dlon = math.radians(plant["lon"] - loc["longitude"])
            a = math.sin(dlat/2)**2 + math.cos(math.radians(loc["latitude"])) * math.cos(math.radians(plant["lat"])) * math.sin(dlon/2)**2
            dist = R * 2 * math.asin(math.sqrt(a))
            if dist < best_dist:
                best_dist = dist
                best_plant = {"city": city, "code": plant["code"], "distance_km": round(dist, 2)}

    return best_plant

def get_access_level(name: str, surname: str, birth_year: int) -> int:
    """Pobiera poziom dostępu osoby do obiektów strategicznych."""
    resp = requests.post(
        "https://hub.ag3nts.org/api/accesslevel",
        json={"apikey": HUB_API_KEY, "name": name, "surname": surname, "birthYear": birth_year}
    )
    return resp.json().get("accessLevel")

def submit_answer(name: str, surname: str, access_level: int, power_plant_code: str) -> str:
    """Wysyła finalną odpowiedź do HUB i zwraca jego odpowiedź.

    Uwaga projektowa: submit_answer jest narzędziem dostępnym dla modelu,
    ale w produkcyjnym systemie akcje nieodwracalne (wysłanie, publikacja)
    powinny wymagać potwierdzenia od użytkownika na poziomie UI — nie polegać
    wyłącznie na decyzji modelu. Tu pomijamy to dla uproszczenia.
    """
    resp = requests.post(
        "https://hub.ag3nts.org/verify",
        json={
            "apikey": HUB_API_KEY,
            "task": "findhim",
            "answer": {
                "name": name,
                "surname": surname,
                "accessLevel": access_level,
                "powerPlant": power_plant_code
            }
        }
    )
    return resp.text


# ============================================================
# SCHEMATY NARZĘDZI — to widzi model LLM w kontekście zapytania
# ============================================================
#
# Schematy trafiają do modelu jako dodatkowy blok tokenów przy KAŻDYM zapytaniu,
# nawet gdy narzędzia nie są używane. To ma koszt — dlatego lekcja mówi,
# żeby agent posiadał nie więcej niż 10-15 narzędzi naraz.
#
# Schemat to "dokumentacja dla modelu bez dostępu do dokumentacji".
# Model widzi: nazwę, opis i parametry. Na tej podstawie decyduje:
#   - czy w ogóle użyć narzędzia,
#   - które narzędzie wybrać spośród dostępnych,
#   - jakie argumenty przekazać.
#
# Dlatego opisy muszą być precyzyjne i zwięzłe (wysoki signal-to-noise).
# Np. "find_closest_plant" jest lepszą nazwą niż "check" — mała szansa
# na kolizję z innym narzędziem, jasno sugeruje co robi.

tools = [
    {
        "type": "function",
        "function": {
            "name": "find_closest_plant",
            # Opis wyjaśnia CO robi narzędzie i KIEDY je użyć.
            # Model nie ma dostępu do kodu — ten opis to jedyna "dokumentacja".
            "description": "Dla podanej osoby sprawdza wszystkie jej lokalizacje i zwraca elektrownię, która jest najbliżej którejkolwiek z nich",
            "parameters": {
                "type": "object",
                "properties": {
                    "name":    {"type": "string", "description": "Imię osoby"},
                    "surname": {"type": "string", "description": "Nazwisko osoby"},
                },
                "required": ["name", "surname"]
                # birthYear celowo NIE jest tu parametrem — obliczenia odległości
                # go nie potrzebują. Pomijamy zbędne parametry, żeby nie mylić modelu.
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_access_level",
            "description": "Pobiera poziom dostępu osoby do obiektów strategicznych",
            "parameters": {
                "type": "object",
                "properties": {
                    "name":       {"type": "string",  "description": "Imię osoby"},
                    "surname":    {"type": "string",  "description": "Nazwisko osoby"},
                    # birth_year jest wymagany przez API accesslevel — model musi go podać.
                    # Dane o roku urodzenia podejrzanych są w prompcie użytkownika,
                    # więc model ma do nich dostęp w kontekście.
                    "birth_year": {"type": "integer", "description": "Rok urodzenia"},
                },
                "required": ["name", "surname", "birth_year"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            # Opis mówi kiedy użyć tego narzędzia ("gdy już znasz...").
            # To ważne — model musi wiedzieć, że to krok finalny, a nie pośredni.
            "description": "Wysyła finalną odpowiedź gdy już znasz kandydata, jego poziom dostępu i kod elektrowni",
            "parameters": {
                "type": "object",
                "properties": {
                    "name":             {"type": "string",  "description": "Imię podejrzanego"},
                    "surname":          {"type": "string",  "description": "Nazwisko podejrzanego"},
                    "access_level":     {"type": "integer", "description": "Poziom dostępu"},
                    "power_plant_code": {"type": "string",  "description": "Kod elektrowni np. PWR1234PL"},
                },
                "required": ["name", "surname", "access_level", "power_plant_code"]
            }
        }
    },
]

# Mapa nazwa_narzędzia -> callable.
# Gdy model zwróci nazwę narzędzia jako string, musimy wiedzieć, którą funkcję
# Pythona uruchomić. To jest "dispatch" — most między światem modelu a kodem.
tool_map = {
    "find_closest_plant": lambda args: find_closest_plant(args["name"], args["surname"]),
    "get_access_level":   lambda args: get_access_level(args["name"], args["surname"], args["birth_year"]),
    "submit_answer":      lambda args: submit_answer(args["name"], args["surname"], args["access_level"], args["power_plant_code"]),
}


# ============================================================
# PĘTLA AGENTA
# ============================================================
#
# Agent = LLM uruchomiony w pętli, który sam decyduje o kolejnych krokach.
# W przeciwieństwie do task_02.py (zwykły skrypt), tu NIE kodujemy na sztywno
# kolejności kroków. Model sam ustala: kogo sprawdzić najpierw, kiedy
# pobrać access level, kiedy wysłać odpowiedź.
#
# Schemat każdej iteracji pętli:
#   1. Wyślij cały kontekst (messages + schematy narzędzi) do LLM
#   2a. Jeśli finish_reason == "tool_calls" → model chce użyć narzędzia:
#       - wyciągnij nazwę i argumenty z odpowiedzi
#       - wywołaj prawdziwą funkcję przez tool_map
#       - dodaj wynik do messages (role: "tool")
#       - wróć do kroku 1 (model nie widział jeszcze wyniku)
#   2b. Jeśli finish_reason == "stop" → model uznał zadanie za zakończone:
#       - wypisz odpowiedź i przerwij pętlę
#
# Każda iteracja to OSOBNE zapytanie do API — model nie "pamięta" poprzednich
# kroków. Pamięć to wyłącznie lista messages, którą my budujemy i wysyłamy
# w całości przy każdym zapytaniu (API jest bezstanowe — lekcja S01E01).

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

messages = [
    {
        "role": "system",
        # Prompt systemowy definiuje rolę i strategię agenta.
        # Zwrot "metodycznie" i opis kolejności kroków pomaga modelowi
        # nie skakać od razu do submit_answer zanim przejrzy wszystkich.
        "content": "Jesteś agentem śledczym. Twoim zadaniem jest znalezienie podejrzanego który był najbliżej elektrowni atomowej. Używaj narzędzi metodycznie: najpierw sprawdź każdą osobę z listy, potem pobierz poziom dostępu kandydata, na końcu wyślij odpowiedź."
    },
    {
        "role": "user",
        # Wiadomość użytkownika zawiera dane wejściowe (listę podejrzanych z S01E01)
        # oraz instrukcję co zrobić. Model ma tu wszystko, czego potrzebuje:
        # dane + cel + dostępne narzędzia (w schematach).
        "content": f"""Sprawdź następujące osoby i znajdź tę, która była najbliżej elektrowni atomowej:

{json.dumps(suspects, ensure_ascii=False, indent=2)}

Dla każdej osoby użyj narzędzia find_closest_plant. Gdy znajdziesz osobę z najmniejszą odległością, pobierz jej poziom dostępu i wyślij odpowiedź."""
    }
]

print("=== START AGENTA ===\n")

# Limit iteracji to zabezpieczenie przed nieskończoną pętlą — gdyby model
# nie doszedł do finish_reason == "stop" (np. z powodu halucynacji lub błędu
# w narzędziu). W produkcji zawsze ustawiaj taki limit.
for step in range(20):
    response = client.chat.completions.create(
        model="anthropic/claude-3-5-haiku",
        messages=messages,
        tools=tools,       # schematy narzędzi dołączane do KAŻDEGO zapytania
    )

    msg = response.choices[0].message
    finish_reason = response.choices[0].finish_reason

    # Odpowiedź modelu zawsze trafia do historii — niezależnie od tego,
    # czy to wywołanie narzędzia, czy finalna odpowiedź. Dzięki temu przy
    # następnym zapytaniu model "pamięta" co już zrobił.
    messages.append(msg)

    if finish_reason == "tool_calls":
        # Model zdecydował się użyć narzędzia.
        # msg.tool_calls to lista — model może zażądać kilku narzędzi naraz
        # (parallel function calling). Tu iterujemy przez wszystkie.
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            # Argumenty przychodzą jako string JSON — musimy je sparsować.
            args = json.loads(tool_call.function.arguments)

            print(f"[krok {step+1}] Agent wywołuje: {name}({args})")
            result = tool_map[name](args)   # tu FAKTYCZNIE wywołujemy funkcję
            print(f"           Wynik: {result}\n")

            # Wynik narzędzia trafia do historii jako rola "tool".
            # tool_call_id łączy wynik z konkretnym wywołaniem — model może
            # zlecić kilka narzędzi równolegle i musi wiedzieć, który wynik
            # do którego wywołania należy.
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, ensure_ascii=False)
            })
        # Po dodaniu wyników wracamy do początku pętli — model dostanie
        # zaktualizowany kontekst i zdecyduje o kolejnym kroku.

    elif finish_reason == "stop":
        # Model nie zażądał żadnego narzędzia — uznał, że zadanie jest skończone
        # i zwrócił odpowiedź tekstową. To naturalny koniec pętli agenta.
        print("=== AGENT ZAKOŃCZYŁ ===")
        print(f"Odpowiedź: {msg.content}")
        break
