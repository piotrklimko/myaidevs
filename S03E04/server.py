import os
from dotenv import load_dotenv
load_dotenv()

"""
S03E04 — Negotiations: Serwer narzędziowy dla agenta AI
========================================================

KONTEKST LEKCJI:
Lekcja S03E04 dotyczy budowania narzędzi dla agentów AI na podstawie danych testowych.
Kluczowe zasady z lekcji:
1. Narzędzia powinny być DOPASOWANE do potrzeb (nie generyczne)
2. Odpowiedzi powinny zawierać wskazówki nawigacyjne (co agent może zrobić dalej)
3. Obsługa błędów powinna jasno wyjaśniać co się stało i co można zrobić
4. Format odpowiedzi powinien być spójny i przewidywalny

ARCHITEKTURA:
Serwer udostępnia JEDNO narzędzie (endpoint /api/search), które:
1. Przyjmuje opis przedmiotu w języku naturalnym (pole "params")
2. DETERMINISTYCZNIE filtruje bazę przedmiotów (keyword matching + fuzzy search)
3. Jeśli dopasowanie niejednoznaczne → LLM (Haiku) wybiera najlepszy match
4. Zwraca listę miast, w których przedmiot jest dostępny

DLACZEGO JEDNO NARZĘDZIE?
- Agent ma max 10 kroków i szuka 3 przedmiotów → 3 kroki na wyszukanie = wystarczy
- Jedno narzędzie = prostsza logika agenta, mniej szans na błąd
- Zgodnie z lekcją: łączymy akcje (search + lookup) w jedno narzędzie

DLACZEGO HYBRID (deterministyczny + LLM)?
- Keyword matching jest SZYBKI i TANI — obsłuży 90% zapytań
- LLM jest potrzebny tylko gdy agent pyta w języku naturalnym
  np. "potrzebuję kabla 10 metrów" → trzeba dopasować do "Kabel miedziany 10 m"
- Dzięki temu nie płacimy za LLM przy każdym zapytaniu
- Lekcja podkreśla: używaj LLM tam gdzie ma sens, resztę rób deterministycznie
"""

import csv
import re
import unicodedata
from pathlib import Path
from flask import Flask, request, jsonify
from openai import OpenAI
import json

# =============================================================================
# KONFIGURACJA
# =============================================================================

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
LLM_MODEL = "anthropic/claude-haiku-4.5"
PORT = 18356

# Klient OpenRouter — używamy interfejsu OpenAI z innym base_url
# (standard w ekosystemie AI: OpenAI-compatible API)
llm = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

app = Flask(__name__)

# =============================================================================
# ŁADOWANIE DANYCH Z CSV
# =============================================================================
# Dane ładujemy RAZ przy starcie serwera (nie przy każdym requeście).
# To klasyczny wzorzec "load once, serve many" — sensowny bo dane się nie zmieniają.

DATA_DIR = Path(__file__).parent

def load_csv(filename, key_col, val_col):
    """Ładuje CSV i zwraca listę krotek (klucz, wartość)."""
    rows = []
    with open(DATA_DIR / filename, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row[key_col].strip(), row[val_col].strip()))
    return rows


# items: lista (name, code) — wszystkie przedmioty
ITEMS = load_csv("items.csv", "name", "code")

# cities: mapowanie code → name (do tłumaczenia kodów miast na nazwy)
CITY_BY_CODE = {}
for name, code in load_csv("cities.csv", "name", "code"):
    CITY_BY_CODE[code] = name

# connections: mapowanie itemCode → set of cityCodes
# (jeden przedmiot może być w wielu miastach)
CONNECTIONS = {}
for item_code, city_code in load_csv("connections.csv", "itemCode", "cityCode"):
    CONNECTIONS.setdefault(item_code, set()).add(city_code)

print(f"[INIT] Załadowano: {len(ITEMS)} przedmiotów, {len(CITY_BY_CODE)} miast, "
      f"{sum(len(v) for v in CONNECTIONS.values())} powiązań")


# =============================================================================
# NORMALIZACJA TEKSTU
# =============================================================================
# Agent wysyła zapytania w języku naturalnym, a nazwy w CSV są bez polskich znaków.
# Normalizujemy OBA teksty do tej samej formy, żeby porównanie było fair.

def normalize(text: str) -> str:
    """
    Normalizuje tekst do porównywania:
    1. Lowercase
    2. Usunięcie znaków diakrytycznych (ą→a, ć→c, etc.)
    3. Usunięcie znaków specjalnych (zostają litery, cyfry, spacje)

    Dlaczego? Bo agent może napisać "łącznik" a w CSV jest "Lacznik",
    albo "10kΩ" a w CSV "10 kOhm". Normalizacja wyrównuje te różnice.
    """
    text = text.lower()
    # Zamiana polskich znaków diakrytycznych na ASCII
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    # Zostaw tylko litery, cyfry, spacje
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    # Kompresja wielokrotnych spacji
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Przygotowujemy znormalizowane nazwy przedmiotów RAZ (cache).
# Każdy element: (znormalizowana_nazwa, oryginalna_nazwa, kod)
ITEMS_NORMALIZED = [
    (normalize(name), name, code) for name, code in ITEMS
]


# =============================================================================
# WYSZUKIWANIE PRZEDMIOTÓW — WARSTWA DETERMINISTYCZNA
# =============================================================================

# Stopwords — słowa, które nie niosą informacji o przedmiocie.
# Filtrujemy je, żeby "potrzebuję turbiny wiatrowej" → ["turbiny", "wiatrowej"]
STOPWORDS = {
    "potrzebuje", "potrzebujemy", "szukam", "szukamy", "chce", "chcemy",
    "prosze", "czy", "jest", "sa", "moze", "gdzie", "znajdz", "znajde",
    "kupic", "dostac", "miec", "jakis", "jaki", "jaka", "jakie",
    "do", "na", "w", "z", "i", "a", "o", "od", "po", "za", "ze",
    "sie", "to", "ten", "ta", "te", "tym", "tego", "tej", "tych",
    "nie", "tak", "ale", "lub", "albo", "oraz", "ktory", "ktora", "ktore",
    "mi", "nam", "mnie", "nas", "go", "jej", "ich", "mu",
    "bardzo", "tylko", "jeszcze", "juz", "tutaj", "tam",
    "bym", "byc", "jest", "bedzie", "byl", "byla", "bylo",
    "potrzebny", "potrzebna", "potrzebne",
    "one", "the", "need", "want", "find", "search", "look", "for",
    "item", "items", "product", "products",
}


def stem_pl(word: str) -> str:
    """
    Bardzo uproszczony stemmer dla języka polskiego.

    DLACZEGO NIE PEŁNY STEMMER?
    - Pełne stemmery (np. Stempel) wymagają dodatkowych zależności
    - Dla naszego zastosowania wystarczy obcinanie popularnych końcówek
    - Nazwy przedmiotów w CSV są w mianowniku — agent może użyć dowolnego przypadku
    - Obcinamy końcówki fleksyjne, żeby "turbiny" → "turbin" pasowało do "turbina" → "turbin"

    Nie jest idealny, ale pokrywa ~90% przypadków deklinacji rzeczowników i przymiotników.
    """
    # Sortujemy końcówki od najdłuższych (żeby "owej" złapać przed "ej")
    suffixes = [
        "owego", "owych", "owej", "owym", "owych",
        "owej", "owym", "owa", "owe", "owy",
        "nych", "nego", "nej", "nym", "nych",
        "ami", "owi", "ach", "iem", "iem",
        "ny", "na", "ne", "ni",
        "ej", "em", "ie", "om",
        "ow", "ek", "ka", "ki", "ko", "ku",
        "ce", "cy", "cz",
        "y", "i", "e", "a", "u", "o",
    ]
    if len(word) <= 3:
        return word
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def keyword_search(query: str, top_n: int = 10) -> list[tuple[str, str, float]]:
    """
    Wyszukiwanie deterministyczne po słowach kluczowych z obsługą polskiej fleksji.

    ALGORYTM:
    1. Normalizuj zapytanie i podziel na tokeny (słowa)
    2. Odfiltruj stopwords
    3. Dla każdego tokena oblicz "stem" (uproszczony rdzeń wyrazu)
    4. Dla każdego przedmiotu sprawdź:
       a) Czy pełny token jest podciągiem nazwy (najsilniejsze dopasowanie)
       b) Czy stem tokena pasuje do stemu jakiegoś słowa w nazwie (fuzzy match)
    5. Zwróć top N posortowanych po score

    DLACZEGO TAK PROSTO?
    - Proste rozwiązanie działa w 90%+ przypadków dla tego typu danych
    - Nazwy przedmiotów są opisowe ("Turbina wiatrowa 400W 48V")
    - Agent zazwyczaj użyje kluczowych słów z nazwy
    - Stemming obsługuje polską odmianę (turbiny→turbin ≈ turbina→turbin)

    Zwraca: [(oryginalna_nazwa, kod, score), ...]
    """
    norm_query = normalize(query)
    tokens = [t for t in norm_query.split() if t not in STOPWORDS and len(t) > 1]

    if not tokens:
        return []

    # Stemujemy tokeny zapytania
    query_stems = [stem_pl(t) for t in tokens]

    scored = []
    for norm_name, orig_name, code in ITEMS_NORMALIZED:
        name_words = norm_name.split()
        name_stems = [stem_pl(w) for w in name_words]

        score = 0
        for token, qstem in zip(tokens, query_stems):
            # Dopasowanie 1: pełny token jako podciąg nazwy (najsilniejsze)
            if token in norm_name:
                score += 2
            # Dopasowanie 2: stem tokena pasuje do stemu słowa w nazwie
            elif any(qstem == nstem for nstem in name_stems):
                score += 1.5
            # Dopasowanie 3: stem tokena jest podciągiem stemu słowa (lub odwrotnie)
            elif any(qstem in nstem or nstem in qstem
                     for nstem in name_stems if len(nstem) >= 3 and len(qstem) >= 3):
                score += 1

        if score > 0:
            # Bonus za frazowe dopasowanie (wszystkie tokeny obok siebie)
            phrase = " ".join(tokens)
            if phrase in norm_name:
                score += len(tokens)
            scored.append((orig_name, code, score))

    # Sortuj: najpierw po score malejąco, potem po nazwie (stabilność)
    scored.sort(key=lambda x: (-x[2], x[0]))
    return scored[:top_n]


# =============================================================================
# WYSZUKIWANIE PRZEDMIOTÓW — WARSTWA LLM (fallback)
# =============================================================================

def llm_match(query: str, candidates: list[tuple[str, str]]) -> tuple[str, str] | None:
    """
    Używa LLM do wybrania najlepszego dopasowania z listy kandydatów.

    KIEDY JEST WYWOŁYWANY:
    - Gdy keyword search zwrócił wiele wyników z tym samym score
    - Gdy zapytanie jest bardzo opisowe i trudne do dopasowania słowami kluczowymi

    DLACZEGO HAIKU?
    - To proste zadanie klasyfikacji (wybierz 1 z N) — nie potrzeba dużego modelu
    - Haiku jest tani i szybki (~0.25$/1M tokenów)
    - Lekcja podkreśla: dobieraj model do złożoności zadania

    PROMPT ENGINEERING:
    - Dajemy LLM jasne instrukcje: zwróć KOD najlepszego dopasowania
    - Podajemy kontekst: to baza przedmiotów handlarzy w postapokaliptycznym świecie
    - Format odpowiedzi: sam kod, nic więcej → łatwy parsing
    """
    if not candidates:
        return None

    # Formatujemy listę kandydatów dla LLM
    items_text = "\n".join(f"- {name} (kod: {code})" for name, code in candidates)

    response = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Jesteś ekspertem od dopasowywania przedmiotów. "
                    "Użytkownik opisuje przedmiot w języku naturalnym. "
                    "Wybierz JEDEN najlepiej pasujący przedmiot z listy. "
                    "Odpowiedz TYLKO kodem przedmiotu, nic więcej."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Zapytanie: {query}\n\n"
                    f"Dostępne przedmioty:\n{items_text}\n\n"
                    f"Który przedmiot najlepiej pasuje? Podaj sam kod."
                ),
            },
        ],
        temperature=0,  # Deterministyczna odpowiedź — chcemy powtarzalności
        max_tokens=20,  # Kod ma ~6 znaków, 20 to bezpieczny margines
    )

    chosen_code = response.choices[0].message.content.strip()

    # Sprawdzamy czy zwrócony kod istnieje w kandydatach
    for name, code in candidates:
        if code == chosen_code:
            return (name, code)

    # Fallback: jeśli LLM zwrócił coś dziwnego, bierzemy pierwszy wynik
    return candidates[0] if candidates else None


# =============================================================================
# LOOKUP MIAST DLA PRZEDMIOTU
# =============================================================================

def get_cities_for_item(item_code: str) -> list[str]:
    """
    Zwraca posortowaną listę nazw miast, w których dostępny jest dany przedmiot.

    Logika:
    1. Szukamy kodu przedmiotu w connections (itemCode → set of cityCodes)
    2. Tłumaczymy kody miast na nazwy ludzkie
    3. Sortujemy alfabetycznie (spójna kolejność = łatwiejsze porównywanie)
    """
    city_codes = CONNECTIONS.get(item_code, set())
    city_names = []
    for cc in city_codes:
        name = CITY_BY_CODE.get(cc)
        if name:
            city_names.append(name)
    return sorted(city_names)


# =============================================================================
# GŁÓWNA FUNKCJA WYSZUKIWANIA
# =============================================================================

def search_item_cities(query: str) -> str:
    """
    Pełen pipeline: zapytanie w języku naturalnym → lista miast.

    FLOW (zgodny z zasadą lekcji: deterministycznie gdzie się da, LLM gdzie trzeba):

    1. Keyword search — DETERMINISTYCZNY, szybki, darmowy
       → Jeśli znalazł 1 wynik z najwyższym score → używamy go
       → Jeśli znalazł wiele z tym samym top score → LLM disambiguuje
       → Jeśli nie znalazł nic → LLM przeszukuje WSZYSTKIE przedmioty (ostatnia deska)

    2. Lookup miast — DETERMINISTYCZNY (proste przeszukanie connections)

    3. Formatowanie odpowiedzi — zgodnie z lekcją:
       - Jasna informacja o wyniku
       - Wskazówki nawigacyjne (co agent może zrobić dalej)
       - Limit 500 bajtów
    """
    print(f"[QUERY] '{query}'")

    # --- KROK 1: Wyszukiwanie deterministyczne ---
    results = keyword_search(query, top_n=10)

    matches = []  # lista (name, code) do zwrócenia

    if results:
        top_score = results[0][2]
        # Filtrujemy tylko wyniki z najwyższym score
        top_results = [(name, code) for name, code, score in results if score == top_score]

        if len(top_results) <= 3:
            # Mało kandydatów — zwracamy WSZYSTKIE z ich miastami.
            # Dzięki temu agent sam może wybrać, a nie tracimy info.
            # Np. "turbina wiatrowa" → 2 warianty (24V i 48V) z różnymi miastami.
            matches = top_results
            print(f"[MATCH] Deterministyczny: {len(matches)} dopasowań")
        else:
            # Za dużo wyników z tym samym score → LLM wybiera najlepszy
            print(f"[AMBIG] {len(top_results)} kandydatów, pytam LLM...")
            match = llm_match(query, top_results)
            if match:
                matches = [match]
                print(f"[MATCH] LLM wybrał: {match[0]} ({match[1]})")
            else:
                return "Nie znaleziono pasującego przedmiotu. Spróbuj opisać go inaczej."
    else:
        # --- KROK 1b: Brak wyników keyword search → LLM jako ostatnia deska ---
        # Wysyłamy WSZYSTKIE przedmioty — drogo, ale rzadko się zdarza.
        print("[FALLBACK] Brak wyników keyword search, pytam LLM o pełną listę...")
        all_items = [(name, code) for name, code in ITEMS]
        match = llm_match(query, all_items)
        if match:
            matches = [match]
            print(f"[MATCH] LLM fallback: {match[0]} ({match[1]})")
        else:
            return "Nie znaleziono pasującego przedmiotu. Spróbuj opisać go inaczej."

    # --- KROK 2: Lookup miast dla każdego dopasowania ---
    # Zbieramy wyniki w format "Nazwa przedmiotu: Miasto1, Miasto2"
    # Jeśli jest kilka dopasowań, agent widzi je wszystkie i może zawęzić.
    parts = []
    for name, code in matches:
        cities = get_cities_for_item(code)
        if cities:
            parts.append(f"{name}: {', '.join(cities)}")
        else:
            parts.append(f"{name}: brak w sprzedazy")

    if not parts:
        return "Nie znaleziono miast z tym przedmiotem. Spróbuj inny opis."

    # --- KROK 3: Formatowanie odpowiedzi ---
    # Limit 500 bajtów! Łączymy wyniki separatorem " | "
    response = " | ".join(parts)

    # Sprawdzamy limit 500 bajtów (UTF-8)
    if len(response.encode("utf-8")) > 500:
        # Obcinamy do limitu — skracamy ostatnią część
        while len(response.encode("utf-8")) > 495 and len(parts) > 1:
            parts.pop()
            response = " | ".join(parts)
        # Jeśli nadal za długo, obcinamy miasta
        if len(response.encode("utf-8")) > 500:
            response = response.encode("utf-8")[:497].decode("utf-8", errors="ignore") + "..."

    print(f"[RESPONSE] {response}")
    return response


# =============================================================================
# ENDPOINT API
# =============================================================================

@app.route("/api/search", methods=["POST"])
def api_search():
    """
    Endpoint narzędziowy dla agenta.

    FORMAT WEJŚCIA (zdefiniowany przez centralę):
    POST {"params": "opis przedmiotu w języku naturalnym"}

    FORMAT WYJŚCIA (wymagany przez centralę):
    {"output": "odpowiedź dla agenta"}

    WAŻNE OGRANICZENIA:
    - Odpowiedź musi mieć 4-500 bajtów
    - Agent ma max 10 kroków
    - Musimy zwrócić jasną, użyteczną odpowiedź

    ZGODNIE Z LEKCJĄ:
    - Spójna struktura odpowiedzi
    - Obsługa błędów z wyjaśnieniem co poszło nie tak
    - Wskazówki co agent może zrobić dalej
    """
    try:
        data = request.get_json(force=True)
        params = data.get("params", "")

        if not params or not params.strip():
            return jsonify({"output": "Podaj opis przedmiotu, np. 'turbina wiatrowa 400W'"})

        result = search_item_cities(params.strip())
        return jsonify({"output": result})

    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({"output": f"Błąd serwera: {str(e)[:100]}. Spróbuj ponownie."})


# =============================================================================
# URUCHOMIENIE SERWERA
# =============================================================================

if __name__ == "__main__":
    # Serwer nasłuchuje na 0.0.0.0:18356
    # 0.0.0.0 = akceptuj połączenia z zewnątrz (nie tylko localhost)
    # Port 18356 = przydzielony przez Azyl (nginx mapuje subdomenę → port)
    print(f"[START] Serwer narzędziowy na porcie {PORT}")
    print(f"[START] Endpoint: POST /api/search")
    print(f"[START] Przedmiotów w bazie: {len(ITEMS)}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
