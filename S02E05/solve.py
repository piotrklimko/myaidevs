import os
from dotenv import load_dotenv
load_dotenv()

"""
S02E05 — drone
===============
Lekcja: PROJEKTOWANIE AGENTÓW — ich instrukcji systemowych, narzędzi, wiedzy i kontekstu.

=== KONTEKST LEKCJI ===

Lekcja omawia 6 kluczowych obszarów projektowania agenta AI:
1. USTAWIENIA  — nazwa, opis, narzędzia, uprawnienia, model (konfiguracja techniczna)
2. PROFIL      — "osobowość", cechy wpływające na jakość pracy (nie tylko kosmetyka!)
3. ZASADY      — reguły komunikacji, radzenia sobie z problemami, zachowania awaryjne
4. LIMITY      — aktualność wiedzy, dynamiczne uprawnienia, świadomość czasu
5. STYL        — sposób wypowiedzi dostosowany do środowiska (tekst vs głos)
6. SESJA       — zmienne zależne od użytkownika i bieżącej interakcji

Instrukcja agenta dzieli się na sekcje:
- <identity>  — motyw przewodni: charakter + styl + zachowanie
- <protocol>  — zasady działania, zarządzanie kontekstem, pamięcią, eskalacja
- <voice>     — ton wypowiedzi, few-shot examples, antywzorce
- <tools>     — opis narzędzi i dostępnych agentów (dynamicznie generowany)

=== ZADANIE FABULARNE ===

Przejmujemy kontrolę nad dronem bojowym DRN-BMB7, aby zbombardować TAMĘ
(nie elektrownię!). W systemie deklarujemy cel jako elektrownię PWR6132PL,
ale faktyczne koordynaty ustawiamy na sektor z tamą. System automatycznie
odznaczy misję jako wykonaną, elektrownia przetrwa, a woda z jeziora
Żarnowieckiego trafi do systemu chłodzenia reaktora.

=== TECHNIKI ZASTOSOWANE ===

1. ANALIZA WIZUALNA (vision model jako "narzędzie percepcji" agenta)
   - Mapa terenu jest podzielona siatką na sektory
   - Używamy GPT-5.4 (lekcja zaleca go jako najlepszy do liczenia siatki)
   - Obraz kodujemy jako base64 (URL-e nie zawsze przechodzą przez OpenRouter)
   - To ilustracja: model vision = sensor agenta, transformujący obraz → dane

2. PODEJŚCIE REAKTYWNE (reactive approach)
   - Nie rozgryzamy całej dokumentacji przed pierwszą próbą
   - API zwraca precyzyjne komunikaty błędów → iteracyjne dopasowywanie
   - Jeśli vision model źle odczyta koordynaty, próbujemy sąsiednich sektorów
   - Wzorzec: percepcja → działanie → refleksja → korekta → powtórzenie

3. MINIMALIZM W DOBORZE NARZĘDZI
   - Dokumentacja API drona celowo zawiera nadmiar funkcji i pułapki
   - Agent skupia się na MINIMUM potrzebnym do misji (8 instrukcji z ~15 dostępnych)
   - Ignorujemy: kalibrację, diagnostykę, personalizację, LED-y
   - Lekcja: "10-15 narzędzi to maks" to uproszczenie — liczy się kontekst i cel

4. DOKUMENTACJA PEŁNA PUŁAPEK
   - Metoda set() jest PRZECIĄŻONA — rozpoznaje parametr po formacie:
     set(x,y) = sektor, set(engineON) = silnik, set(50m) = pułap, set(destroy) = cel
   - Parametry w NAWIASACH — łatwo przeoczyć (przykłady w docs to pokazują)
   - set(return) jest WYMAGANY — bez niego API odmówi startu drona
   - Kolidujące nazwy celowo utrudniają zrozumienie API
"""

import base64
import re
import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Konfiguracja — klucze i endpointy
# ---------------------------------------------------------------------------
HUB_API_KEY = os.environ["HUB_API_KEY"]
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

VERIFY_URL = "https://hub.ag3nts.org/verify"
MAP_URL = f"https://hub.ag3nts.org/data/{HUB_API_KEY}/drone.png"

# Klient OpenRouter — jedno API do wielu modeli (OpenAI, Anthropic, Google...)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)


# ---------------------------------------------------------------------------
# KROK 1: Analiza mapy z modelem vision
# ---------------------------------------------------------------------------
# Mapa jest podzielona siatką na sektory (kolumny × wiersze, indeksowane od 1).
# Tama ma podkreślony kolor wody (intensywny niebieski).
#
# Używamy GPT-5.4 — lekcja wskazuje, że jest lepszy niż GPT-4o w liczeniu siatki.
# Obraz przesyłamy jako base64, bo URL-e przez OpenRouter bywają niestabilne.
#
# UWAGA: Modele vision potrafią dawać NIESPÓJNE wyniki między wywołaniami
# (raz 5×3 siatka, raz 3×4). Dlatego stosujemy podejście reaktywne:
# próbujemy wskazany sektor, a jeśli nie trafimy — sąsiednie.

def analyze_map() -> tuple[int, int]:
    """
    Pobiera mapę, analizuje modelem vision, zwraca (kolumna, wiersz) tamy.
    Próbuje dwa podejścia: base64 i URL. Jeśli oba zawiodą — zwraca None.
    """
    vision_prompt = (
        "This is a satellite/aerial map of the Żarnowiec power plant area "
        "divided by a grid into sectors.\n\n"
        "TASK: Find the DAM (tama). The dam separates the power plant from "
        "the Żarnowiec Lake. Water near the dam has intensified blue color.\n\n"
        "1. Count grid lines carefully — columns (left→right) and rows (top→bottom).\n"
        "2. Top-left sector = (1,1).\n"
        "3. Describe each sector briefly.\n"
        "4. Which EXACT sector contains the dam?\n\n"
        "You MUST answer with: DAM_COL=X, DAM_ROW=Y"
    )

    # Próba 1: base64 z GPT-5.4 (zalecany przez lekcję)
    # Próba 2: URL z GPT-4o (fallback — inny model, inna metoda przesyłu)
    attempts = [
        ("openai/gpt-5.4", "base64"),
        ("openai/gpt-4o", "url"),
        ("openai/gpt-4o", "base64"),
    ]

    img_b64 = None
    for model, method in attempts:
        try:
            print(f"[VISION] Próba: {model} ({method})...")

            if method == "base64":
                if img_b64 is None:
                    print(f"[VISION] Pobieram mapę: {MAP_URL}")
                    img_data = requests.get(MAP_URL).content
                    img_b64 = base64.b64encode(img_data).decode()
                    print(f"[VISION] Rozmiar obrazu: {len(img_data)} bajtów")
                image_content = {"url": f"data:image/png;base64,{img_b64}"}
            else:
                image_content = {"url": MAP_URL}

            response = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": vision_prompt},
                        {"type": "image_url", "image_url": image_content},
                    ],
                }],
                max_tokens=1500,
            )

            answer = response.choices[0].message.content
            print(f"[VISION] Odpowiedź:\n{answer}\n")

            col_match = re.search(r"DAM_COL\s*=\s*(\d+)", answer)
            row_match = re.search(r"DAM_ROW\s*=\s*(\d+)", answer)

            if col_match and row_match:
                col, row = int(col_match.group(1)), int(row_match.group(1))
                print(f"[VISION] Sektor tamy: kolumna={col}, wiersz={row}")
                return col, row

            print("[VISION] Nie znaleziono koordynatów, próbuję dalej...")

        except Exception as e:
            print(f"[VISION] Błąd: {e}, próbuję dalej...")

    # Fallback — jeśli żaden model nie dał koordynatów
    print("[VISION] Wszystkie próby zawiodły, nie udało się odczytać mapy.")
    return None, None


# ---------------------------------------------------------------------------
# KROK 2: Budowa instrukcji drona
# ---------------------------------------------------------------------------
# Dokumentacja DRN-BMB7 zawiera ~15 metod — używamy tylko 8.
# Kluczowe pułapki:
# - set() przeciążony — format parametru decyduje o znaczeniu
# - Parametry w NAWIASACH: set(engineON), nie "set engineON"
# - set(return) WYMAGANY — bez niego dron nie wystartuje
# - flyToLocation na KOŃCU — wymaga wcześniejszej konfiguracji

def build_instructions(dam_col: int, dam_row: int) -> list[str]:
    """
    Składa minimalną sekwencję instrukcji dla drona.

    Fabularny trick:
    - setDestinationObject(PWR6132PL) → oficjalny cel: ELEKTROWNIA
    - set(col,row) → faktyczny cel: TAMA
    System raportuje "elektrownia zniszczona", a bomba leci na tamę.
    """
    return [
        "setDestinationObject(PWR6132PL)",  # oficjalny cel (kamuflaj)
        f"set({dam_col},{dam_row})",         # faktyczny sektor: TAMA
        "set(engineON)",                     # silnik włączony
        "set(100%)",                         # pełna moc
        "set(50m)",                          # pułap 50m
        "set(destroy)",                      # cel misji: zniszczenie
        "set(return)",                       # powrót (WYMAGANY!)
        "flyToLocation",                     # start lotu
    ]


# ---------------------------------------------------------------------------
# KROK 3: Wysyłka do API
# ---------------------------------------------------------------------------

def send_instructions(instructions: list[str]) -> dict:
    """Wysyła instrukcje do /verify, zwraca odpowiedź API."""
    payload = {
        "apikey": HUB_API_KEY,
        "task": "drone",
        "answer": {"instructions": instructions},
    }
    resp = requests.post(VERIFY_URL, json=payload)
    return resp.json()


# ---------------------------------------------------------------------------
# KROK 4: Generowanie wariantów koordynatów (sąsiednie sektory)
# ---------------------------------------------------------------------------
# Vision model bywa niespójny — raz poda (5,3), raz (2,4), raz (4,5).
# Podejście reaktywne: jeśli wskazany sektor "nie trafia w tamę",
# próbujemy sąsiednich sektorów (Manhattan distance ≤ 2).
# To odpowiednik zasady "fail fast, learn fast" — iteracyjna korekta.

def generate_coord_variants(col: int, row: int, max_distance: int = 4) -> list[tuple[int, int]]:
    """
    Generuje listę koordynatów do wypróbowania: najpierw wskazany sektor,
    potem sąsiednie (w kolejności rosnącej odległości Manhattan).

    Modele vision bywają niespójne — mogą podać siatkę 5×3 zamiast 3×4.
    Dlatego skanujemy szerzej (do dystansu 4), co pokrywa całą rozsądną siatkę.
    """
    variants = [(col, row)]  # najpierw oryginał

    for distance in range(1, max_distance + 1):
        for dc in range(-distance, distance + 1):
            for dr in range(-distance, distance + 1):
                if abs(dc) + abs(dr) != distance:
                    continue
                nc, nr = col + dc, row + dr
                if 1 <= nc <= 6 and 1 <= nr <= 6 and (nc, nr) not in variants:
                    variants.append((nc, nr))

    return variants


# ---------------------------------------------------------------------------
# MAIN — Orkiestracja
# ---------------------------------------------------------------------------
# Cały przepływ to uproszczony agent:
#   1. PERCEPCJA   → model vision analizuje mapę → koordynaty tamy
#   2. PLANOWANIE  → budowa instrukcji (dokumentacja API + koordynaty)
#   3. DZIAŁANIE   → wysyłka instrukcji
#   4. REFLEKSJA   → analiza odpowiedzi (sukces / "nie trafisz")
#   5. KOREKTA     → próba sąsiedniego sektora
#   6. POWTÓRZENIE → aż do flagi FLG
#
# Agent ma: NARZĘDZIA (API drona, vision), WIEDZĘ (dokumentacja),
# KONTEKST (kod PWR6132PL, mapa) i ZASADY (deklaruj elektrownię, celuj w tamę).

def main():
    print("=" * 60)
    print("S02E05 — drone: Bombardowanie tamy w Żarnowcu")
    print("=" * 60)
    print()

    # --- Faza 1: Percepcja — analiza wizualna mapy ---
    dam_col, dam_row = analyze_map()

    if dam_col is None:
        # Vision zawiodło — skanujemy całą siatkę (max 6×6 = 36 sektorów)
        print("[PLAN] Vision nie dał wyników — skanowanie całej siatki")
        variants = [(c, r) for r in range(1, 7) for c in range(1, 7)]
    else:
        variants = generate_coord_variants(dam_col, dam_row)

    print(f"[PLAN] Wygenerowano {len(variants)} wariantów koordynatów do sprawdzenia\n")

    for i, (c, r) in enumerate(variants, 1):
        instructions = build_instructions(c, r)
        print(f"--- Próba {i}/{len(variants)}: sektor ({c},{r}) ---")
        for j, instr in enumerate(instructions, 1):
            print(f"  {j}. {instr}")

        result = send_instructions(instructions)
        result_str = str(result)
        print(f"  Odpowiedź: {result}\n")

        if "FLG" in result_str:
            print("=" * 60)
            print(f"SUKCES! Flaga: {result_str}")
            print(f"Tama znaleziona w sektorze ({c},{r})")
            print("=" * 60)
            return

        # Jeśli błąd inny niż "nie trafisz" → problem z instrukcjami, nie z pozycją
        msg = result.get("message", "")
        if "hit the dam" not in msg and "somewhere nearby" not in msg:
            print(f"[WARN] Nieoczekiwany błąd: {msg}")
            # Nie przerywamy — może kolejny sektor zadziała

    print("[FAIL] Żaden wariant nie zadziałał.")


if __name__ == "__main__":
    main()
