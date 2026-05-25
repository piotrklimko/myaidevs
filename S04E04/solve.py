"""
================================================================================
S04E04 — "filesystem" — Agentowe porządkowanie bazy wiedzy
================================================================================

KONTEKST LEKCJI:
Lekcja S04E04 dotyczy projektowania własnej bazy wiedzy dla AI. Kluczowe wnioski:

1. STRUKTURA BAZY WIEDZY — dobrze zaprojektowana baza wiedzy dla agentów AI
   powinna mieć jasną strukturę katalogów (np. Me/World/Craft/Ops/System),
   szablony notatek i zasady organizacji. Agent sam nie wymyśli dobrej struktury
   — potrzebuje naszych wytycznych.

2. NOTATKI DLA AI ≠ NOTATKI DLA LUDZI — notatki muszą być zrozumiałe
   bez dodatkowego kontekstu. Linki, referencje, powiązania muszą być jawne,
   bo agent nie ma "pamięci" o tym, co było wcześniej.

3. PODZIAŁ ODPOWIEDZIALNOŚCI — człowiek odpowiada za treść i zasady,
   AI odpowiada za organizację (formatowanie, linkowanie, walidacja, indeksowanie).

4. MARKDOWN JAKO FORMAT — prosty, przeszukiwalny, transformowalny format
   idealny dla agentów. Pliki .md to zwykły tekst, który LLM naturalnie rozumie.

5. SYSTEM WIELOAGENTOWY — różne agenty mogą współpracować nad bazą wiedzy:
   jeden analizuje, drugi waliduje, trzeci organizuje pliki.

ZADANIE:
Uporządkowanie chaotycznych notatek Natana (handlowca) w wirtualny filesystem
z trzema katalogami: /miasta (potrzeby), /osoby (handlowcy), /towary (oferty).

ARCHITEKTURA ROZWIĄZANIA (podejście agentowe):
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌─────────┐
│ Pobierz dane│────▶│ Agent #1:    │────▶│ Agent #2:    │────▶│ Batch   │
│ (ZIP)       │     │ Ekstrakcja   │     │ Walidacja    │     │ API     │
└─────────────┘     │ (LLM)       │     │ (LLM)       │     │ calls   │
                    └──────────────┘     └──────────────┘     └─────────┘

Agent #1 (Ekstraktor): Analizuje surowe notatki i wydobywa strukturę danych.
Agent #2 (Walidator):  Sprawdza poprawność ekstrakcji wobec oryginalnych notatek.
Oba agenty działają na tym samym modelu LLM, ale mają różne instrukcje systemowe.
To klasyczny wzorzec "generator + critic" zwiększający jakość wyników.
================================================================================
"""

import json
import os
from dotenv import load_dotenv
load_dotenv()
import zipfile
import requests
from openai import OpenAI

# ── Konfiguracja ─────────────────────────────────────────────────────
# Klucze API i endpointy. W projekcie edukacyjnym hardkodowane bezpośrednio.
HUB_API_KEY = os.environ["HUB_API_KEY"]
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
VERIFY_URL = "https://hub.ag3nts.org/verify"
NOTES_URL = "https://hub.ag3nts.org/dane/natan_notes.zip"

# OpenRouter jako proxy do modeli LLM — pozwala korzystać z różnych modeli
# (Claude, GPT, Gemini itp.) przez jeden interfejs kompatybilny z OpenAI SDK.
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

WORK_DIR = os.path.dirname(os.path.abspath(__file__))


# ══════════════════════════════════════════════════════════════════════
# KROK 1: POBIERANIE DANYCH
# ══════════════════════════════════════════════════════════════════════
# W kontekście bazy wiedzy — to odpowiednik "ingestii" danych.
# Notatki Natana to surowe, chaotyczne dane z różnych źródeł:
# - ogłoszenia.txt  → zapotrzebowanie miast (co potrzebują i ile)
# - rozmowy.txt     → dziennik rozmów z handlowcami (kto za co odpowiada)
# - transakcje.txt  → historia transakcji (kto komu co sprzedał)
# - README.md       → opis co jest w każdym pliku
#
# Kluczowa obserwacja z lekcji: dane z różnych źródeł trzeba połączyć,
# bo żadne z nich samodzielnie nie daje pełnego obrazu. To jest istota
# Context Engineering — umiejętne zestawienie fragmentów kontekstu.
# ══════════════════════════════════════════════════════════════════════
def download_notes():
    """Pobiera i rozpakowuje archiwum ZIP z notatkami Natana."""
    zip_path = os.path.join(WORK_DIR, "natan_notes.zip")
    if not os.path.exists(zip_path):
        print("[1] Pobieram notatki Natana...")
        r = requests.get(NOTES_URL)
        with open(zip_path, "wb") as f:
            f.write(r.content)
    notes_dir = os.path.join(WORK_DIR, "notes")
    os.makedirs(notes_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(notes_dir)
    # Wczytaj wszystkie pliki tekstowe do słownika {nazwa: treść}
    files = {}
    for fname in os.listdir(notes_dir):
        fpath = os.path.join(notes_dir, fname)
        if os.path.isfile(fpath):
            with open(fpath, "r", encoding="utf-8") as f:
                files[fname] = f.read()
    return files


# ══════════════════════════════════════════════════════════════════════
# KROK 2: AGENT EKSTRAKCJI (Agent #1)
# ══════════════════════════════════════════════════════════════════════
# Wzorzec: jeden model analizuje surowe dane i generuje ustrukturyzowany JSON.
#
# Kluczowe elementy promptu (lekcja o prompt engineering):
# 1. JASNA ROLA — "jesteś agentem analizującym notatki handlowe"
# 2. PRECYZYJNE ZADANIE — trzy konkretne typy danych do wydobycia
# 3. ZASADY — normalizacja znaków, format nazw, logika transakcji
# 4. FORMAT WYJŚCIA — dokładna struktura JSON
#
# Dlaczego temperature=0? Bo chcemy deterministycznych, powtarzalnych
# wyników. Ekstrakcja danych to zadanie wymagające precyzji, nie kreatywności.
#
# Uwaga na pułapkę z transakcjami: "MiastoA -> towar -> MiastoB"
# oznacza, że MiastoA SPRZEDAJE towar. To nie jest oczywiste i wymaga
# jawnego podania w instrucji — inaczej LLM może pomylić kierunek.
#
# Kolejna pułapka: imiona i nazwiska rozrzucone po różnych notatkach.
# Np. "Kisiel" pojawia się w jednym zdaniu, a "Rafał" w innym —
# oba dotyczą Brudzewa, więc to ta sama osoba: Rafał Kisiel.
# To wymaga wnioskowania z kontekstu, co jest mocną stroną LLM.
# ══════════════════════════════════════════════════════════════════════
def analyze_notes(notes: dict) -> dict:
    """Agent #1: Analizuje surowe notatki i wydobywa strukturę danych."""

    # Łączymy wszystkie notatki w jeden tekst z wyraźnymi separatorami.
    # To ważne — LLM musi wiedzieć, z jakiego pliku pochodzi dana informacja,
    # bo logika ekstrakcji zależy od źródła (np. zapotrzebowanie z ogłoszeń,
    # ale osoby z rozmów).
    notes_text = "\n\n".join(
        f"=== {name} ===\n{content}" for name, content in notes.items()
    )

    system_prompt = """Jesteś agentem analizującym notatki handlowe. Twoim zadaniem jest wyekstrahowanie trzech typów danych:

1. MIASTA - jakie towary potrzebuje każde miasto i w jakiej ilości (bez jednostek)
2. OSOBY - kto odpowiada za handel w każdym mieście (imię i nazwisko)
3. TOWARY - które towary są na sprzedaż i przez które miasto (na podstawie transakcji - miasto SPRZEDAJĄCE)

WAŻNE ZASADY:
- Nie używaj polskich znaków (ą→a, ć→c, ę→e, ł→l, ń→n, ó→o, ś→s, ź/ż→z) w nazwach i w JSON
- Nazwy towarów w mianowniku liczby pojedynczej (np. "koparka" nie "koparki", "mlotek" nie "mlotki")
- W transakcjach format to: "MiastoA -> towar -> MiastoB" - MiastoA SPRZEDAJE towar do MiastoB
- Jeśli ten sam towar sprzedaje wiele miast, podaj wszystkie miasta
- Jeśli imię i nazwisko osoby pojawiają się w różnych miejscach rozmów, połącz je (np. "Kisiel" + "Rafał" z kontekstu Brudzewa = "Rafal Kisiel")

Odpowiedz WYŁĄCZNIE poprawnym JSON w formacie:
{
  "miasta": {
    "nazwa_miasta": {"towar1": ilosc, "towar2": ilosc}
  },
  "osoby": {
    "nazwa_miasta": "Imie Nazwisko"
  },
  "towary": {
    "nazwa_towaru": ["miasto1", "miasto2"]
  }
}"""

    print("[2] Wysyłam notatki do LLM w celu analizy...")
    response = client.chat.completions.create(
        model="anthropic/claude-haiku-4.5",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": notes_text},
        ],
        temperature=0,  # Deterministyczna odpowiedź — ważne dla ekstrakcji
    )

    raw = response.choices[0].message.content
    print(f"[2] Odpowiedź LLM:\n{raw}\n")

    # Parsowanie odpowiedzi — LLM może owinąć JSON w blok ```json ... ```
    # To częsty wzorzec: model formatuje odpowiedź w Markdown nawet gdy
    # prosimy o "czysty JSON". Trzeba to obsłużyć.
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0]
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0]

    return json.loads(raw.strip())


# ══════════════════════════════════════════════════════════════════════
# KROK 3: AGENT WALIDACJI (Agent #2) — wzorzec "Generator + Critic"
# ══════════════════════════════════════════════════════════════════════
# To kluczowy wzorzec z lekcji o systemach wieloagentowych:
# - Agent #1 generuje dane (może popełnić błędy)
# - Agent #2 dostaje te same źródła + wynik Agenta #1 i waliduje
#
# Dlaczego to działa lepiej niż jeden agent?
# 1. Inny "punkt widzenia" — walidator patrzy krytycznie na wynik
# 2. Podwójna weryfikacja — dwa niezależne "przeczytania" źródła
# 3. Specjalizacja — każdy agent ma węższe, bardziej precyzyjne zadanie
#
# UWAGA NA PUŁAPKĘ: walidator też może popełniać błędy!
# W pierwszym uruchomieniu walidator pomylił kupujących ze sprzedającymi.
# Dlatego prompt walidatora musi być BARDZO precyzyjny w kwestii logiki
# transakcji (punkt 4 z "KRYTYCZNE").
#
# To ważna lekcja: system wieloagentowy nie jest magicznie lepszy —
# wymaga tak samo dobrego prompt engineeringu jak pojedynczy agent.
# ══════════════════════════════════════════════════════════════════════
def validate_data(data: dict, notes: dict) -> dict:
    """Agent #2: Waliduje i poprawia wynik Agenta #1."""

    notes_text = "\n\n".join(
        f"=== {name} ===\n{content}" for name, content in notes.items()
    )

    system_prompt = """Jesteś agentem walidującym. Otrzymujesz oryginalne notatki oraz wyekstrahowane dane JSON.
Sprawdź:
1. Czy wszystkie 8 miast z notatek jest uwzględnionych
2. Czy ilości towarów są poprawne (z ogłoszeń)
3. Czy osoby są poprawnie przypisane do miast (z rozmów)
4. KRYTYCZNE: Czy towary na sprzedaż wynikają WYŁĄCZNIE z LEWEJ strony transakcji (sprzedający).
   Format transakcji: "MiastoA -> towar -> MiastoB" oznacza że MiastoA SPRZEDAJE.
   MiastoB jest KUPUJĄCYM i NIE powinno być dodawane do listy sprzedających tego towaru!
5. Czy nazwy nie zawierają polskich znaków
6. Czy nazwy towarów są w mianowniku liczby pojedynczej
7. Czy nazwy miast w kluczach JSON zaczynają się MAŁĄ literą (lowercase)

WAŻNE: Wszystkie nazwy miast muszą być pisane małą literą (np. "opalino" nie "Opalino").

Jeśli znajdziesz błędy, popraw je. Zwróć WYŁĄCZNIE poprawiony JSON (ten sam format) bez żadnego dodatkowego tekstu.

Format:
{
  "miasta": {"nazwa_miasta": {"towar": ilosc}},
  "osoby": {"nazwa_miasta": "Imie Nazwisko"},
  "towary": {"towar": ["miasto1"]}
}"""

    print("[3] Walidacja danych przez drugiego agenta...")
    response = client.chat.completions.create(
        model="anthropic/claude-haiku-4.5",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                # Przekazujemy ORYGINALNE notatki + wynik Agenta #1.
                # Walidator ma oba konteksty i może porównać.
                "content": f"NOTATKI:\n{notes_text}\n\nDANE DO WALIDACJI:\n{json.dumps(data, ensure_ascii=False, indent=2)}",
            },
        ],
        temperature=0,
    )

    raw = response.choices[0].message.content
    print(f"[3] Odpowiedź walidatora:\n{raw}\n")

    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0]
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0]

    return json.loads(raw.strip())


# ══════════════════════════════════════════════════════════════════════
# KROK 4: BUDOWANIE OPERACJI NA FILESYSTEM
# ══════════════════════════════════════════════════════════════════════
# Zamieniamy ustrukturyzowane dane JSON na sekwencję operacji API.
#
# API filesystem wymaga ścieżek w formacie ^[a-z0-9_]+$ (bez polskich
# znaków, bez wielkich liter). Funkcja normalize() zapewnia tę konwersję.
#
# Używamy trybu BATCH — jedna duża lista operacji zamiast wielu requestów.
# To optymalizacja: jedno zapytanie HTTP zamiast ~33 osobnych.
# API przetwarza operacje SEKWENCYJNIE, więc kolejność ma znaczenie:
# 1. Reset (wyczyść)
# 2. Utwórz katalogi (muszą istnieć przed plikami)
# 3. Utwórz pliki miast (muszą istnieć przed linkami w osoby/towary)
# 4. Utwórz pliki osób (linkują do miast)
# 5. Utwórz pliki towarów (linkują do miast)
#
# WAŻNE: API waliduje, że "markdown links must point to existing files"
# — dlatego pliki miast MUSZĄ być tworzone PRZED osobami i towarami!
# ══════════════════════════════════════════════════════════════════════
def normalize(name: str) -> str:
    """Zamienia na lowercase i usuwa polskie znaki.

    API filesystem wymaga nazw pasujących do ^[a-z0-9_]+$.
    Przykłady: "Darzlubie" → "darzlubie", "wołowina" → "wolowina"
    """
    name = name.lower()
    for pl, en in [
        ("ą", "a"), ("ć", "c"), ("ę", "e"), ("ł", "l"),
        ("ń", "n"), ("ó", "o"), ("ś", "s"), ("ź", "z"), ("ż", "z"),
    ]:
        name = name.replace(pl, en)
    return name


def build_batch(data: dict) -> list:
    """Przekształca dane JSON w listę operacji batch dla API filesystem.

    Wynikowa struktura:
    /miasta/
        opalino     → {"chleb": 45, "woda": 120, "mlotek": 6}
        domatowo    → {"makaron": 60, "woda": 150, "lopata": 8}
        ...
    /osoby/
        iga_kapecka → "Iga Kapecka\n[opalino](/miasta/opalino)"
        ...
    /towary/
        chleb       → "[domatowo](/miasta/domatowo)\n[celbowo](/miasta/celbowo)..."
        ...
    """
    ops = []

    # Reset — czyści cały filesystem. Ważne przy powtórnych uruchomieniach,
    # żeby nie zostały "śmieci" z poprzednich prób.
    ops.append({"action": "reset"})

    # Katalogi — muszą być utworzone PRZED plikami
    for d in ["miasta", "osoby", "towary"]:
        ops.append({"action": "createDirectory", "path": f"/{d}"})

    # PLIKI MIAST — zawierają JSON z zapotrzebowaniem
    # Źródło danych: ogłoszenia.txt (przeanalizowane przez LLM)
    for city, needs in data["miasta"].items():
        city_norm = normalize(city)
        ops.append(
            {
                "action": "createFile",
                "path": f"/miasta/{city_norm}",
                "content": json.dumps(needs, ensure_ascii=False),
            }
        )

    # PLIKI OSÓB — zawierają imię + link Markdown do miasta
    # Źródło danych: rozmowy.txt (kto dzwonił z jakiego miasta)
    # Link w formacie Markdown: [nazwa_miasta](/miasta/nazwa_miasta)
    # API waliduje, że cel linka istnieje — dlatego miasta tworzmy wcześniej!
    for city, person in data["osoby"].items():
        city_norm = normalize(city)
        filename = normalize(person.replace(" ", "_"))
        content = f"{person}\n[{city}](/miasta/{city_norm})"
        ops.append(
            {
                "action": "createFile",
                "path": f"/osoby/{filename}",
                "content": content,
            }
        )

    # PLIKI TOWARÓW — zawierają linki do miast SPRZEDAJĄCYCH dany towar
    # Źródło danych: transakcje.txt (format: "MiastoA -> towar -> MiastoB")
    # Jeden towar może być sprzedawany przez wiele miast → wiele linków
    for item, cities in data["towary"].items():
        item_norm = normalize(item)
        links = "\n".join(f"[{c}](/miasta/{normalize(c)})" for c in cities)
        ops.append(
            {
                "action": "createFile",
                "path": f"/towary/{item_norm}",
                "content": links,
            }
        )

    return ops


# ══════════════════════════════════════════════════════════════════════
# KROK 5: KOMUNIKACJA Z API
# ══════════════════════════════════════════════════════════════════════
# API /verify obsługuje zarówno pojedyncze operacje jak i batch.
# Przy batch: answer jest listą operacji (nie obiektem).
# API zwraca kody statusu:
#   100 — batch wykonany
#    20 — plik utworzony
#    10 — katalog utworzony
#    47 — filesystem zresetowany
#     0 — "done" sukces (z flagą!)
#  <0   — błąd (np. -940 invalid path)
# ══════════════════════════════════════════════════════════════════════
def send_to_api(payload):
    """Wysyła zapytanie do API filesystem /verify."""
    body = {"apikey": HUB_API_KEY, "task": "filesystem", "answer": payload}
    r = requests.post(VERIFY_URL, json=body)
    return r.json()


# ══════════════════════════════════════════════════════════════════════
# MAIN — ORKIESTRACJA PIPELINE'U
# ══════════════════════════════════════════════════════════════════════
# Cały flow to klasyczny pipeline agentowy:
# 1. Ingest  → pobierz surowe dane
# 2. Extract → Agent #1 wydobywa strukturę (LLM)
# 3. Validate → Agent #2 weryfikuje poprawność (LLM)
# 4. Transform → zamień dane na operacje API
# 5. Execute → wyślij do systemu plików
# 6. Verify → potwierdź wynik
#
# To uproszczona wersja wzorca z lekcji, gdzie agenci operują na
# bazie wiedzy Markdown. W pełnej wersji (przykład daily-news z lekcji)
# agenci mają dostęp do szablonów notatek, mapy treści (MoC) i mogą
# samodzielnie decydować o lokalizacji i strukturze nowych wpisów.
# ══════════════════════════════════════════════════════════════════════
def main():
    # 1. Pobierz notatki — surowe dane z archiwum ZIP
    notes = download_notes()
    print(f"[1] Pobrano pliki: {list(notes.keys())}\n")

    # 2. Agent #1: ekstrakcja danych z notatek przez LLM
    data = analyze_notes(notes)

    # 3. Agent #2: walidacja i korekta (wzorzec Generator + Critic)
    data = validate_data(data, notes)

    print(f"[4] Finalna struktura danych:")
    print(json.dumps(data, ensure_ascii=False, indent=2))

    # 4. Transformacja danych → operacje batch
    ops = build_batch(data)
    print(f"\n[5] Przygotowano {len(ops)} operacji. Wysyłam batch...")

    # 5. Wykonanie — jeden request z całym filesystem
    result = send_to_api(ops)
    print(f"[5] Wynik batch: {json.dumps(result, ensure_ascii=False, indent=2)}")

    # 6. Podgląd utworzonego filesystem (debugging)
    print("\n[6] Sprawdzam listę plików...")
    list_result = send_to_api({"action": "listFiles", "path": "/"})
    print(json.dumps(list_result, ensure_ascii=False, indent=2))

    # 7. Finalna weryfikacja — wysyłamy "done" do Centrali
    print("\n[7] Wysyłam 'done' do weryfikacji...")
    done_result = send_to_api({"action": "done"})
    print(f"[7] Wynik: {json.dumps(done_result, ensure_ascii=False, indent=2)}")


if __name__ == "__main__":
    main()
