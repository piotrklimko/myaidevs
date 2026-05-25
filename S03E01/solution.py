"""
S03E01 — Analiza anomalii w odczytach czujników elektrowni

STRATEGIA (temat lekcji: ewaluacja i obserwowanie):
────────────────────────────────────────────────────
Mamy 9999 plików JSON z odczytami sensorów. Anomalie dzielą się na 4 typy:

  Typ 1: Wartości poza dozwolonym zakresem          → PROGRAMISTYCZNIE
  Typ 2: Sensor zwraca pola których nie powinien    → PROGRAMISTYCZNIE
  Typ 3: Operator mówi OK, ale dane są złe          → LLM (klasyfikacja notatki)
  Typ 4: Operator mówi "błąd", ale dane są OK       → LLM (klasyfikacja notatki)

OPTYMALIZACJA KOSZTÓW (kluczowy temat lekcji):
──────────────────────────────────────────────
• Typy 1 i 2 to czysta logika — nie potrzebujemy LLM, to zero kosztu.
• Typy 3 i 4 wymagają LLM, ale są tylko 2032 UNIKALNE notatki spośród 9999 plików.
  Cachujemy klasyfikację notatki → każdy tekst klasyfikujemy raz, niezależnie od
  tego ile razy pojawia się w danych. Oszczędzamy ~80% kosztów LLM.
• LLM klasyfikuje binarnie: "ok" / "problem" — minimalny output = tani output.
• Wysyłamy notatki w batchach (wiele naraz w jednym prompcie) → mniej zapytań API.
"""

import json
import os
from dotenv import load_dotenv
load_dotenv()
import re
import time
from collections import defaultdict
from openai import OpenAI

# ─── Konfiguracja ──────────────────────────────────────────────────────────────

HUB_API_KEY = os.environ["HUB_API_KEY"]
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
CENTRAL_URL = "https://hub.ag3nts.org/verify"

# Używamy szybkiego i taniego modelu — klasyfikacja binarna nie wymaga dużego modelu
# haiku-4.5 jest wielokrotnie tańszy od sonnet/opus, a zadanie jest proste
MODEL = "anthropic/claude-haiku-4.5"

SENSORS_DIR = "sensors"

# ─── Zakresy poprawnych wartości dla aktywnych sensorów ───────────────────────
#
# Każde pole ma zakres [min, max] włącznie. Wartości z sensor_type decydują,
# które pola powinny być NIEZEROWE — reszta powinna wynosić dokładnie 0.
#
FIELD_RANGES = {
    "temperature_K":       (553,  873),   # Kelviny
    "pressure_bar":        (60,   160),   # bary
    "water_level_meters":  (5.0,  15.0),  # metry
    "voltage_supply_v":    (229.0, 231.0),# wolty
    "humidity_percent":    (40.0,  80.0), # procenty
}

# Mapowanie: token z sensor_type → pole w JSON
# np. "temperature" → "temperature_K"
SENSOR_TO_FIELD = {
    "temperature": "temperature_K",
    "pressure":    "pressure_bar",
    "water":       "water_level_meters",
    "voltage":     "voltage_supply_v",
    "humidity":    "humidity_percent",
}

# Wszystkie możliwe pola pomiarowe (do sprawdzenia, czy nieaktywne = 0)
ALL_MEASUREMENT_FIELDS = list(SENSOR_TO_FIELD.values())


# ═══════════════════════════════════════════════════════════════════════════════
# ETAP 1: ANALIZA PROGRAMISTYCZNA
# ═══════════════════════════════════════════════════════════════════════════════

def parse_active_sensors(sensor_type: str) -> set[str]:
    """
    Parsuje pole sensor_type i zwraca zbiór AKTYWNYCH komponentów.

    Przykład: "voltage/water" → {"voltage", "water"}
    Przykład: "pressure/temperature/voltage" → {"pressure", "temperature", "voltage"}
    """
    return set(sensor_type.lower().split("/"))


def check_data_programmatically(file_id: str, data: dict) -> list[str]:
    """
    Sprawdza plik pod kątem anomalii danych (Typy 1 i 2) bez użycia LLM.

    Zwraca listę opisów problemów (pusta lista = brak anomalii w danych).

    Typ 1: wartość aktywnego sensora poza zakresem lub równa 0
    Typ 2: nieaktywny sensor zwraca wartość != 0
    """
    problems = []
    sensor_type = data.get("sensor_type", "")
    active = parse_active_sensors(sensor_type)

    for component, field in SENSOR_TO_FIELD.items():
        value = data.get(field, 0)
        is_active = component in active

        if is_active:
            # Aktywny sensor: wartość musi być w zakresie i niezerowa
            lo, hi = FIELD_RANGES[field]
            if value == 0:
                problems.append(
                    f"[TYP2] Aktywny sensor '{component}' ma wartość 0 "
                    f"(pole {field}) — powinien zwracać dane"
                )
            elif not (lo <= value <= hi):
                problems.append(
                    f"[TYP1] {field}={value} poza zakresem [{lo}, {hi}] "
                    f"dla aktywnego sensora '{component}'"
                )
        else:
            # Nieaktywny sensor: wartość MUSI być 0
            if value != 0:
                problems.append(
                    f"[TYP2] Nieaktywny sensor '{component}' zwraca {field}={value} "
                    f"(powinno być 0 dla sensor_type='{sensor_type}')"
                )

    return problems


# ═══════════════════════════════════════════════════════════════════════════════
# ETAP 2: KLASYFIKACJA NOTATEK PRZEZ LLM
# ═══════════════════════════════════════════════════════════════════════════════

def classify_notes_with_llm(unique_notes: list[str], client: OpenAI) -> dict[str, str]:
    """
    Klasyfikuje listę unikalnych notatek operatora jako 'ok' lub 'problem'.

    Kluczowa optymalizacja:
    - Wysyłamy UNIKALNE notatki (2032), nie wszystkie 9999 pliki
    - Wysyłamy je w batchach po BATCH_SIZE naraz → mniej zapytań API
    - Model zwraca tylko krótkie etykiety → minimalny output = niski koszt

    Zwraca słownik: {tekst_notatki → "ok" | "problem"}
    """

    BATCH_SIZE = 80  # Ile notatek w jednym prompcie — balans między rozmiarem a precyzją

    system_prompt = """You are a binary classifier for operator notes from a power plant sensor system.

For each numbered note, classify whether the operator claims:
- "ok" — everything is fine, readings are normal, within expected range, stable, no issues
- "problem" — something is wrong, anomaly detected, readings are out of range, errors found

Respond ONLY with lines in format: <number>:<label>
Example:
1:ok
2:problem
3:ok

No explanations. No other text."""

    results = {}

    # Przetwarzamy notatki w batchach
    for batch_start in range(0, len(unique_notes), BATCH_SIZE):
        batch = unique_notes[batch_start : batch_start + BATCH_SIZE]

        # Numerujemy notatki w batchyu od 1
        numbered = "\n".join(
            f"{i+1}: {note}" for i, note in enumerate(batch)
        )

        user_message = f"Classify these {len(batch)} operator notes:\n\n{numbered}"

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            temperature=0,  # Deterministyczna klasyfikacja — nie chcemy losowości
            max_tokens=len(batch) * 12,  # ~"99:problem\n" = 12 tokenów na notatkę
        )

        raw = response.choices[0].message.content.strip()

        # Parsujemy odpowiedź: "1:ok", "2:problem" itp.
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            m = re.match(r"^(\d+)\s*:\s*(ok|problem)$", line, re.IGNORECASE)
            if m:
                idx = int(m.group(1)) - 1  # z powrotem na 0-based
                label = m.group(2).lower()
                if 0 <= idx < len(batch):
                    note_text = batch[idx]
                    results[note_text] = label

        # Małe opóźnienie między batchami — grzeczność wobec API
        if batch_start + BATCH_SIZE < len(unique_notes):
            time.sleep(0.3)

        print(f"  Batch {batch_start//BATCH_SIZE + 1}/{(len(unique_notes)-1)//BATCH_SIZE + 1} "
              f"({batch_start + len(batch)}/{len(unique_notes)} notatek) — OK")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# ETAP 3: WYSYŁANIE ODPOWIEDZI DO CENTRALI
# ═══════════════════════════════════════════════════════════════════════════════

def send_answer(anomaly_ids: list[str], client_http) -> dict:
    """
    Wysyła listę anomalii do centrali w wymaganym formacie JSON.

    Format odpowiedzi akceptowany przez centralę (wg dokumentacji):
    - stringi z zerami wiodącymi: ["0001", "0002"]
    - liczby bez zera: [1, 2]
    - nazwy plików: ["0001.json"]
    - dane mieszane

    Używamy formatu string z zerami wiodącymi, bo to najprostszy format.
    """
    import urllib.request

    payload = {
        "apikey": HUB_API_KEY,
        "task": "evaluation",
        "answer": {
            "recheck": anomaly_ids
        }
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        CENTRAL_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

    # ── Wczytaj wszystkie pliki ────────────────────────────────────────────────
    print("Wczytuję pliki...")
    files = {}  # file_id → dict z danymi JSON
    for fname in sorted(os.listdir(SENSORS_DIR)):
        if not fname.endswith(".json"):
            continue
        file_id = fname.replace(".json", "")  # np. "0001"
        with open(os.path.join(SENSORS_DIR, fname)) as f:
            files[file_id] = json.load(f)
    print(f"Wczytano {len(files)} plików\n")

    # ── ETAP 1: Analiza programistyczna ───────────────────────────────────────
    print("=" * 60)
    print("ETAP 1: Analiza programistyczna (zakresy i pola sensorów)")
    print("=" * 60)

    data_anomaly_files = {}   # file_id → lista opisów problemów (tylko dane)
    data_status = {}          # file_id → True (dane OK) / False (dane złe)

    for file_id, data in files.items():
        problems = check_data_programmatically(file_id, data)
        data_status[file_id] = (len(problems) == 0)
        if problems:
            data_anomaly_files[file_id] = problems

    print(f"Pliki z anomaliami w danych: {len(data_anomaly_files)}")
    print(f"Pliki z poprawnymi danymi:   {len(files) - len(data_anomaly_files)}\n")

    # ── ETAP 2: Klasyfikacja notatek przez LLM ────────────────────────────────
    print("=" * 60)
    print("ETAP 2: Klasyfikacja notatek operatora przez LLM")
    print("=" * 60)

    # Cache klasyfikacji na dysku — jeśli już raz zapłaciliśmy za LLM,
    # nie płacimy drugi raz. Plik cache.json zapisuje wyniki między uruchomieniami.
    CACHE_FILE = "notes_cache.json"

    if os.path.exists(CACHE_FILE):
        print(f"Wczytuję cache z {CACHE_FILE}...")
        with open(CACHE_FILE) as f:
            note_classification = json.load(f)
        print(f"Załadowano {len(note_classification)} sklasyfikowanych notatek z cache\n")
    else:
        # Zbierz UNIKALNE notatki — to serce optymalizacji kosztowej
        # Każda notatka pojawia się średnio ~5x, więc zamiast 9999 zapytań
        # robimy klasyfikację 2032 unikalnych tekstów
        unique_notes = list({data["operator_notes"] for data in files.values()})
        print(f"Unikalnych notatek do klasyfikacji: {len(unique_notes)} (z {len(files)} plików)")
        print("Wysyłam do LLM w batchach...\n")

        note_classification = classify_notes_with_llm(unique_notes, client)

        # Zapisz cache na dysk
        with open(CACHE_FILE, "w") as f:
            json.dump(note_classification, f, ensure_ascii=False, indent=2)
        print(f"\nZapisano cache do {CACHE_FILE}")

    # Sprawdź ile sklasyfikowano
    ok_count = sum(1 for v in note_classification.values() if v == "ok")
    problem_count = sum(1 for v in note_classification.values() if v == "problem")
    print(f"\nSklasyfikowano {len(note_classification)}/{len(unique_notes)} notatek:")
    print(f"  'ok'      : {ok_count}")
    print(f"  'problem' : {problem_count}\n")

    # ── ETAP 3: Wykrywanie rozbieżności nota ↔ dane ───────────────────────────
    print("=" * 60)
    print("ETAP 3: Wykrywanie rozbieżności notatka ↔ dane")
    print("=" * 60)

    note_anomaly_files = {}  # file_id → opis rozbieżności

    for file_id, data in files.items():
        note = data["operator_notes"]
        label = note_classification.get(note)

        if label is None:
            # LLM nie sklasyfikował tej notatki (błąd parsowania) — pomijamy
            continue

        data_ok = data_status[file_id]

        if label == "ok" and not data_ok:
            # TYP 3: Operator mówi "wszystko OK", ale dane mają błędy
            note_anomaly_files[file_id] = (
                f"[TYP3] Operator twierdzi OK, ale dane są niepoprawne"
            )
        elif label == "problem" and data_ok:
            # TYP 4: Operator raportuje problem, ale dane są w porządku
            note_anomaly_files[file_id] = (
                f"[TYP4] Operator twierdzi błąd, ale dane są prawidłowe"
            )

    print(f"Rozbieżności notatka ↔ dane: {len(note_anomaly_files)}")
    print(f"  Typ 3 (ok claim + złe dane): "
          f"{sum(1 for v in note_anomaly_files.values() if 'TYP3' in v)}")
    print(f"  Typ 4 (problem claim + dobre dane): "
          f"{sum(1 for v in note_anomaly_files.values() if 'TYP4' in v)}\n")

    # ── ETAP 4: Sumowanie wszystkich anomalii ─────────────────────────────────
    print("=" * 60)
    print("ETAP 4: Sumowanie wyników")
    print("=" * 60)

    # Suma zbiorów: anomalie danych + anomalie notatek
    # Plik może mieć oba typy jednocześnie (złe dane I zła notatka)
    all_anomaly_ids = sorted(set(data_anomaly_files.keys()) | set(note_anomaly_files.keys()))

    print(f"Anomalie w danych:            {len(data_anomaly_files)}")
    print(f"Anomalie w notatkach:         {len(note_anomaly_files)}")
    print(f"Suma unikalna (do wysłania):  {len(all_anomaly_ids)}\n")

    # Pokaż kilka przykładów
    print("Przykłady anomalii (pierwsze 10):")
    for fid in all_anomaly_ids[:10]:
        data_issues = data_anomaly_files.get(fid, [])
        note_issue = note_anomaly_files.get(fid, "")
        issues = data_issues + ([note_issue] if note_issue else [])
        print(f"  {fid}.json: {issues[0]}")

    # ── ETAP 5: Wysyłanie do centrali ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ETAP 5: Wysyłanie odpowiedzi do centrali")
    print("=" * 60)

    response = send_answer(all_anomaly_ids, None)
    print(f"Odpowiedź centrali: {response}")


if __name__ == "__main__":
    main()
