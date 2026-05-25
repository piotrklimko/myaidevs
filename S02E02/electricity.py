import os
from dotenv import load_dotenv
load_dotenv()

import base64
import io
import json
import requests
from PIL import Image
from openai import OpenAI

# ============================================================
# S02E02 — Zewnętrzny kontekst narzędzi i dokumentów
# ============================================================
#
# Lekcja S02E02 dotyczy łączenia LLM z zewnętrznymi źródłami danych:
# dokumentami, obrazami, bazami wiedzy. Kluczowe tematy:
#
# 1. FORMAT DANYCH MA ZNACZENIE — dane w formie obrazów są trudniejsze
#    w wyszukiwaniu i interpretacji niż tekst. Lekcja mówi wprost:
#    "LLM nie widzi obrazka" — trzeba go przekonwertować do formatu,
#    w którym model może nad nim rozumować (opis tekstowy, symboliczny,
#    albo Base64 z multimodalnością z S01E04).
#
# 2. DEKOMPOZYCJA — zamiast wysyłać cały obraz planszy 3x3 do modelu
#    (ryzyko: za dużo detali, model traci uwagę — lekcja S02E01),
#    rozbijamy go na 9 komórek i analizujemy osobno. To ten sam wzorzec
#    co "chunking" dokumentów tekstowych — mniejsze fragmenty dają
#    precyzyjniejsze wyniki.
#
# 3. SUBAGENT/NARZĘDZIE VISION — lekcja sugeruje: "opisanie obrazka
#    wydeleguj do odpowiedniego narzędzia lub subagenta". Tu analiza
#    wizualna jest wydzielona do analyze_cell_pair() — oddzielne
#    zapytanie do modelu vision, jak analyze_image() z S01E04.
#
# UWAGA ARCHITEKTONICZNA:
# To rozwiązanie używa modelu vision do porównania orientacji kabli.
# Jest to podejście "agentowe", ale niestabilne — modele vision słabo
# rozpoznają orientację małych elementów graficznych. Alternatywne,
# deterministyczne podejście: pixel-scan krawędzi komórek kodem PIL
# (sprawdź, gdzie ciemne piksele dotykają krawędzi → wyznacz kierunki
# kabli → programistycznie oblicz różnicę rotacji). Lekcja S01E05
# i S02E02 obie mówią: "jeśli zadanie może zostać zrealizowane kodem,
# powinniśmy z tego skorzystać". Porównanie kabli to problem wizualny,
# ale deterministyczny — nie wymaga "rozumowania" modelu.
#
# Mimo to rozwiązanie z vision zadziałało (po kilku próbach) i jest
# dobrą ilustracją multimodalności z zewnętrznym kontekstem.

HUB_API_KEY = os.environ["HUB_API_KEY"]
HUB_URL = "https://hub.ag3nts.org/verify"
BOARD_URL = f"https://hub.ag3nts.org/data/{HUB_API_KEY}/electricity.png"
SOLVED_URL = "https://hub.ag3nts.org/i/solved_electricity.png"
RESET_URL = f"{BOARD_URL}?reset=1"

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

# Gemini Flash — wybrany ze względu na szybkość i cenę przy zadaniach vision.
# Lekcja S02E02: "nie wszystkie modele vision będą dobrze radziły sobie
# z tym zadaniem — przetestuj które modele zwracają najlepsze wyniki".
# Flash jest szybki i tani, ale mniej precyzyjny niż np. Gemini 3 Flash Preview.
VISION_MODEL = "google/gemini-2.0-flash-001"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# Pozycje komórek w siatce 3x3. Format "WierszxKolumna" — zgodny z API HUB-a.
POSITIONS = [
    "1x1", "1x2", "1x3",
    "2x1", "2x2", "2x3",
    "3x1", "3x2", "3x3",
]


# ============================================================
# POBIERANIE I PRZYGOTOWANIE OBRAZÓW
# ============================================================

def fetch_image(url: str) -> Image.Image:
    """Pobiera obrazek z URL i konwertuje na obiekt PIL RGB.

    Konwersja na RGB normalizuje format — usuwamy kanał alpha i różnice
    kolorystyczne, żeby crop i analiza działały spójnie.
    """
    r = requests.get(url)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


def image_to_b64(img: Image.Image, fmt="PNG") -> str:
    """Konwertuje obraz PIL na string Base64 — format wymagany przez API vision.

    Ten sam wzorzec co analyze_image() z S01E04: model nie może "zobaczyć"
    pliku na dysku. Obraz musi trafić jako Base64 w polu image_url
    wiadomości JSON. Alternatywa: publiczny URL, ale tu operujemy na
    wyciętych fragmentach, które nie mają URL.
    """
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def crop_cells(img: Image.Image) -> dict:
    """Rozcina planszę 3x3 na 9 osobnych komórek.

    To jest DEKOMPOZYCJA obrazu — odpowiednik "chunkingu" z lekcji S02E02,
    ale dla danych wizualnych zamiast tekstowych. Zamiast rzucać cały obraz
    planszy w kontekst modelu (który ma problem z utrzymaniem uwagi na
    9 małych elementach naraz — "instruction dropout" z lekcji S02E02),
    wycinamy każdą komórkę osobno i analizujemy ją w izolacji.

    Podział jest prosty: obraz dzielimy na 3 kolumny × 3 wiersze.
    Zakłada, że plansza zajmuje cały obraz bez obramowania.
    """
    w, h = img.size
    print(f"   Rozmiar obrazu: {w}x{h}")

    cells = {}
    for row in range(3):
        for col in range(3):
            cell_w = w // 3
            cell_h = h // 3
            x1 = col * cell_w
            y1 = row * cell_h
            x2 = x1 + cell_w
            y2 = y1 + cell_h
            pos = f"{row+1}x{col+1}"
            cells[pos] = img.crop((x1, y1, x2, y2))
    return cells


# ============================================================
# ANALIZA VISION — model porównuje parę komórek
# ============================================================

def analyze_cell_pair(current_cell: Image.Image, target_cell: Image.Image, pos: str) -> int:
    """Wysyła parę komórek (current + target) do modelu vision i pyta o rotację.

    To jest realizacja wzorca "subagent/narzędzie vision" z lekcji S02E02:
    główna logika deleguje interpretację obrazu do osobnego zapytania
    do wyspecjalizowanego modelu. Nie mieszamy vision z główną pętlą agenta.

    Prompt jest zaprojektowany tak, żeby model:
    1. Zidentyfikował kierunki kabli w OBU obrazkach (T/R/B/L)
    2. Obliczył różnicę jako liczbę obrotów 90° CW (0-3)
    3. Zwrócił ustrukturyzowaną odpowiedź JSON (Structured Output z S01E01)

    Zwracamy REASONING — dzięki temu można zweryfikować, czy model
    poprawnie "zobaczył" kable. To tip z lekcji S02E02: wyniki narzędzi
    powinny zawierać wskazówki, co model zrobił i dlaczego.
    """
    current_b64 = image_to_b64(current_cell)
    target_b64 = image_to_b64(target_cell)

    prompt = """You are analyzing two cable connector puzzle pieces.

Image 1: CURRENT state of cell position """ + pos + """
Image 2: TARGET state of the same cell

Each cell contains a cable/wire connector piece. A rotation is 90 degrees clockwise.

Analyze the cable connections (which edges - Top/Right/Bottom/Left - have cable exits) in both images.

Determine: how many 90-degree CLOCKWISE rotations does the CURRENT piece need to match the TARGET?
Answer must be 0, 1, 2, or 3.

Respond ONLY with a JSON object:
{"rotations": <number>, "current_connections": ["T","R","B","L" subset], "target_connections": ["T","R","B","L" subset], "reasoning": "brief explanation"}"""

    # Wiadomość multimodalna: dwa obrazy + tekst w jednym zapytaniu.
    # Kolejność ma znaczenie — lekcja S02E01 o cache: statyczny tekst
    # na początku, dynamiczne dane (obrazy) potem. Ale tu cache nie gra
    # roli, bo każde zapytanie ma unikalne obrazy.
    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Image 1 (CURRENT):"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{current_b64}"}},
                {"type": "text", "text": "Image 2 (TARGET):"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{target_b64}"}},
                {"type": "text", "text": prompt}
            ]
        }],
        max_tokens=300,
    )
    text = response.choices[0].message.content.strip()

    # Parsowanie JSON z odpowiedzi modelu — model czasem dodaje markdown
    # wokół JSON-a, więc szukamy pierwszego { i ostatniego }.
    # To kompromis: nie używamy Structured Output (Gemini go nie wspiera
    # przez OpenRouter tak jak GPT), ale prosty heurystyczny parser.
    start = text.find('{')
    end = text.rfind('}') + 1
    if start == -1:
        print(f"   WARNING: No JSON in response for {pos}: {text}")
        return 0
    data = json.loads(text[start:end])
    return data.get("rotations", 0), data


# ============================================================
# MARTWY KOD — analiza całej planszy naraz (nieużywany)
# ============================================================
#
# analyze_whole_board() to alternatywne podejście: zamiast 9 osobnych
# zapytań vision, jedno zapytanie z obydwoma pełnymi planszami.
# Szybsze, ale MNIEJ dokładne — model vision ma problem z rozróżnieniem
# 9 małych komórek na jednym obrazie. To dokładnie problem opisany
# w lekcji S02E02 jako "instruction dropout" przy zbyt dużym kontekście.
#
# Funkcja została tu zachowana dla porównania podejść, ale main() używa
# analyze_cell_pair() (dekompozycja per-komórka = wyższa precyzja).

def analyze_whole_board(current_b64: str, solved_b64: str) -> dict:
    """Analyze entire board with both images at once - faster but less accurate."""
    prompt = """You are solving a cable connector puzzle on a 3x3 grid.

Image 1: CURRENT board state
Image 2: TARGET (solved) board state

For each of the 9 cells (positions 1x1 to 3x3), determine how many 90-degree CLOCKWISE rotations
the current piece needs to match the target.

Grid positions:
1x1 | 1x2 | 1x3
2x1 | 2x2 | 2x3
3x1 | 3x2 | 3x3

For each cell, the number of cable connections (1,2,3,4) MUST be the same in both images - only orientation differs.

Respond ONLY with JSON:
{
  "1x1": {"rotations": 0, "reasoning": "already matches"},
  "1x2": {"rotations": 1, "reasoning": "..."},
  "1x3": {"rotations": 2, "reasoning": "..."},
  "2x1": {"rotations": 0, "reasoning": "..."},
  "2x2": {"rotations": 3, "reasoning": "..."},
  "2x3": {"rotations": 1, "reasoning": "..."},
  "3x1": {"rotations": 0, "reasoning": "..."},
  "3x2": {"rotations": 2, "reasoning": "..."},
  "3x3": {"rotations": 1, "reasoning": "..."}
}"""

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Image 1 (CURRENT board):"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{current_b64}"}},
                {"type": "text", "text": "Image 2 (TARGET board):"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{solved_b64}"}},
                {"type": "text", "text": prompt}
            ]
        }],
        max_tokens=800,
    )
    text = response.choices[0].message.content.strip()
    print(f"\nOdpowiedź modelu (całościowa analiza):\n{text}\n")
    start = text.find('{')
    end = text.rfind('}') + 1
    if start == -1:
        raise ValueError(f"No JSON in response: {text}")
    data = json.loads(text[start:end])
    return {pos: data[pos]["rotations"] for pos in POSITIONS if pos in data}


def rotate_cell_api(position: str) -> dict:
    """Wysyła jeden obrót (90° CW) do HUB-a.

    Jedno zapytanie = jeden obrót jednego pola. To jest "akcja" w rozumieniu
    lekcji S02E02 — agent wywołuje narzędzie, które zmienia stan zewnętrznego
    systemu. Każdy obrót kosztuje jedno żądanie API — dlatego ważna jest
    precyzja analizy vision (błędny obrót = kolejne 3 obroty, żeby "wrócić").
    """
    r = requests.post(HUB_URL, json={
        "apikey": HUB_API_KEY,
        "task": "electricity",
        "answer": {"rotate": position}
    })
    r.raise_for_status()
    return r.json()


# ============================================================
# GŁÓWNA LOGIKA — workflow (nie agent)
# ============================================================
#
# To jest WORKFLOW, nie agent — kroki są zakodowane na sztywno:
# 1. Pobierz oba obrazy (current + solved)
# 2. Rozbij na komórki (dekompozycja)
# 3. Porównaj parami przez vision (9 zapytań)
# 4. Wyślij obroty do API
#
# Lekcja S01E04 rozróżnia agent vs workflow: tu kolejność kroków jest
# z góry znana, nie ma potrzeby "odkrywania" co robić dalej.
# Jedyny element niedeterministyczny to krok 3 (vision) — reszta to kod.
#
# Wskazówki z lekcji sugerują podejście agentowe (Function Calling),
# ale workflow tu wystarczył, bo problem jest dobrze zdefiniowany.

def main():
    print("=== ELECTRICITY PUZZLE SOLVER ===\n")

    # --- Krok 1: Pobierz oba obrazy ---
    # Aktualny stan planszy (zmienia się po obrotach) + wzorzec docelowy
    # (stały, publiczny URL). To jest "zewnętrzny kontekst" z lekcji S02E02 —
    # dane, z których model musi "zrozumieć" stan i zaplanować akcje.
    print("1. Pobieranie obrazków planszy...")
    current_img = fetch_image(BOARD_URL)
    solved_img = fetch_image(SOLVED_URL)
    print(f"   Aktualna: {current_img.size}, Docelowa: {solved_img.size}")

    # Zapis do plików — artefakty debugowe (można pominąć w produkcji).
    current_img.save("S02E02/current_board.png")
    solved_img.save("S02E02/solved_board.png")
    print("   Zapisano: current_board.png, solved_board.png")

    # --- Krok 2: Dekompozycja obrazu na komórki ---
    # Odpowiednik "chunkingu" dokumentów z S02E02: zamiast całego obrazu,
    # model vision dostanie małe, izolowane fragmenty = wyższa precyzja.
    print("\n2. Przycinanie komórek...")
    current_cells = crop_cells(current_img)
    solved_cells = crop_cells(solved_img)

    # Zapis wyciętych komórek — debugging wizualny.
    for pos in POSITIONS:
        current_cells[pos].save(f"S02E02/cell_current_{pos}.png")
        solved_cells[pos].save(f"S02E02/cell_solved_{pos}.png")
    print("   Zapisano pliki komórek do S02E02/")

    # --- Krok 3: Analiza per-komórka przez model vision ---
    # 9 oddzielnych zapytań do Gemini Flash — po jednym na komórkę.
    # Każde zawiera DWA obrazy (current + target) i prosi o liczbę obrotów.
    # To jest najdroższy i najwolniejszy krok — tu vision może się pomylić.
    print(f"\n3. Analiza per-komórka ({VISION_MODEL})...")
    rotations = {}

    for pos in POSITIONS:
        print(f"   Analizuję {pos}...", end=" ")
        result, details = analyze_cell_pair(current_cells[pos], solved_cells[pos], pos)
        rotations[pos] = result
        curr = details.get("current_connections", "?")
        tgt = details.get("target_connections", "?")
        reason = details.get("reasoning", "")
        print(f"-> {result} obrotów | curr={curr} tgt={tgt} | {reason}")

    # --- Krok 4: Podsumowanie planu ---
    print("\n4. Plan obrotów:")
    total = 0
    for pos in POSITIONS:
        n = rotations[pos]
        total += n
        if n > 0:
            print(f"   {pos}: {n}x obrót w prawo")
        else:
            print(f"   {pos}: OK (bez zmian)")

    print(f"\n   Łącznie zapytań API: {total}")

    if total == 0:
        print("   Plansza już wygląda prawidłowo - wysyłam zapytanie weryfikacyjne...")

    # --- Krok 5: Wykonanie obrotów ---
    # Wysyłamy obroty sekwencyjnie — każdy POST to jeden obrót 90° CW.
    # Gdy plansza osiągnie stan docelowy (może po DOWOLNYM obrocie, nie
    # tylko ostatnim), HUB zwróci flagę {FLG:...} — dlatego sprawdzamy
    # po każdym zapytaniu.
    print(f"\n5. Wysyłanie obrotów...")
    flag = None
    for pos in POSITIONS:
        n = rotations[pos]
        for i in range(n):
            print(f"   POST rotate {pos} ({i+1}/{n})...", end=" ")
            result = rotate_cell_api(pos)
            print(f"-> {result}")
            if "FLG" in str(result):
                flag = result
                break
        if flag:
            break

    if flag:
        print(f"\n*** FLAGA! ***\n{flag}")
    else:
        # Brak flagi = vision się pomylił w co najmniej jednej komórce.
        # Pobranie aktualnego stanu pozwala zweryfikować wizualnie.
        # W praktyce: reset + ponowna próba (może z innym modelem).
        print("\n6. Weryfikacja - sprawdzam aktualny stan...")
        new_img = fetch_image(BOARD_URL)
        new_img.save("S02E02/final_board.png")
        print("   Zapisano: final_board.png")
        print("   (Jeśli plansza nie zgadza się ze wzorcem, rozważ reset i ponowną próbę)")


if __name__ == "__main__":
    main()
