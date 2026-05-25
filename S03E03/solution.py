import os
from dotenv import load_dotenv
load_dotenv()

"""
S03E03 — Nawigacja robota przez reaktor (Reactor)

KONTEKST LEKCJI: Kontekstowy feedback wspierający skuteczność agentów
═══════════════════════════════════════════════════════════════════════
Lekcja omawia mechaniki feedbacku, hooków i proaktywnego działania agentów.
W zadaniu mamy do czynienia z **pętlą percepcja → decyzja → akcja** — robot
musi reagować na zmieniające się otoczenie (ruchome bloki reaktora) na bieżąco.

DLACZEGO BEZ LLM (podejście deterministyczne)?
───────────────────────────────────────────────
Lekcja uczy nas: "gdzie możliwe jest uproszczenie realizacji zadań bądź nawet
całkowite pominięcie LLM, powinniśmy to rozważyć."

Tu logika decyzyjna jest prosta i w pełni deterministyczna:
  1. Jeśli mogę iść w prawo (kolumna docelowa bezpieczna) → idź
  2. Jeśli nie, ale mogę czekać (bieżąca kolumna bezpieczna) → czekaj
  3. Jeśli też nie (blok zbliża się do mojej kolumny) → cofnij się w lewo

LLM nie wnosi tu żadnej wartości — mapa jest strukturalna, reguły jasne,
a halucynacja mogłaby zabić robota. To lekcja o stosowaniu WŁAŚCIWYCH narzędzi
do danego problemu.

MECHANIKA GRY:
──────────────
- Plansza 7 kolumn × 5 wierszy (kolumny 1-7, wiersze 1-5)
- Robot porusza się po wierszu 5 (najniższy), od kolumny 1 do 7
- Bloki reaktora zajmują 2 pola (top_row, bottom_row) w jednej kolumnie
- Bloki poruszają się góra/dół cyklicznie — odbijają się od krawędzi
- KLUCZOWE: bloki poruszają się RÓWNOCZEŚNIE z wydaniem komendy
  → muszę SYMULOWAĆ stan planszy po ruchu zanim wydam komendę

FEEDBACK LOOP (nawiązanie do lekcji):
─────────────────────────────────────
Każda odpowiedź API to "feedback z otoczenia" — informacja zwrotna o stanie
planszy po wykonanej akcji. Agent (nasz skrypt) reaguje na ten feedback,
podejmując kolejną decyzję. To esencja pętli agentowej:
  percepcja (odczyt mapy) → rozumowanie (symulacja) → akcja (komenda)

W kontekście lekcji to odpowiednik:
- onStepFinish → odczytujemy nowy stan planszy
- onStepStart → podejmujemy decyzję o kolejnej akcji
- "informacja zwrotna z otoczenia" → pozycje bloków + ich kierunki ruchu
"""

import requests
import json

# ─── Konfiguracja ──────────────────────────────────────────────────────────────

HUB_API_KEY = os.environ["HUB_API_KEY"]
VERIFY_URL = "https://hub.ag3nts.org/verify"

# Limity planszy
MIN_ROW = 1
MAX_ROW = 5  # Robot zawsze na wierszu 5
MIN_COL = 1
MAX_COL = 7  # Cel
MAX_STEPS = 100  # Bezpiecznik — żeby nie wpaść w nieskończoną pętlę


# ─── Symulacja ruchu bloków ───────────────────────────────────────────────────
#
# To jest KLUCZOWY element rozwiązania. API zwraca stan planszy PO wykonaniu
# ostatniej komendy. Pole "direction" mówi nam, w którą stronę blok poruszy
# się przy NASTĘPNEJ komendzie. Musimy więc zasymulować ten ruch, żeby wiedzieć
# czy dana kolumna będzie bezpieczna ZANIM wyślemy komendę.
#
# Mechanika ruchu bloku:
# - direction "down" → top_row += 1, bottom_row += 1
# - direction "up"   → top_row -= 1, bottom_row -= 1
# - Po dotarciu do krawędzi (bottom_row == 5 lub top_row == 1) kierunek się odwraca
#
def simulate_blocks(blocks):
    """
    Symuluje jeden krok ruchu WSZYSTKICH bloków.

    Zwraca listę bloków z NOWYMI pozycjami (po ruchu).
    Nie modyfikuje oryginału — tworzymy kopie.

    Jest to odpowiednik "lookahead" — patrzymy w przyszłość zanim podejmiemy
    decyzję. W kontekście lekcji to element "context engineering" — agent
    wzbogaca swój kontekst o symulowaną przyszłość, nie tylko bieżący stan.
    """
    new_blocks = []
    for b in blocks:
        top = b["top_row"]
        bot = b["bottom_row"]
        d = b["direction"]

        if d == "down":
            top += 1
            bot += 1
        else:  # up
            top -= 1
            bot -= 1

        # Odwrócenie kierunku przy granicy
        if bot >= MAX_ROW:
            new_dir = "up"
        elif top <= MIN_ROW:
            new_dir = "down"
        else:
            new_dir = d

        new_blocks.append({
            "col": b["col"],
            "top_row": top,
            "bottom_row": bot,
            "direction": new_dir,
        })
    return new_blocks


# ─── Sprawdzanie bezpieczeństwa kolumny ──────────────────────────────────────
#
# Robot porusza się WYŁĄCZNIE po wierszu 5 (najniższy). Kolumna jest niebezpieczna
# jeśli JAKIKOLWIEK blok po ruchu będzie miał bottom_row == 5 w tej kolumnie.
#
def is_column_safe(col, blocks_after_move):
    """
    Sprawdza, czy kolumna `col` będzie wolna od bloków na wierszu 5
    PO symulowanym ruchu bloków.

    Robot jest bezpieczny tylko wtedy, gdy żaden blok nie zajmuje
    wiersza 5 w jego kolumnie. Blok zajmuje wiersze od top_row do bottom_row.
    """
    for b in blocks_after_move:
        if b["col"] == col and b["bottom_row"] >= MAX_ROW:
            # Blok zajmuje wiersz 5 w tej kolumnie — niebezpiecznie!
            return False
    return True


# ─── Wybór komendy ───────────────────────────────────────────────────────────
#
# Logika decyzyjna z treści zadania:
# 1. Preferuj ruch w PRAWO (cel jest na prawo) — jeśli kolumna col+1 bezpieczna
# 2. Jeśli nie, CZEKAJ — jeśli bieżąca kolumna col bezpieczna
# 3. Jeśli też nie, COFAJ SIĘ w lewo — ucieczka przed nadjeżdżającym blokiem
#
# Ta hierarchia priorytetów to deterministyczna heurystyka. Nie potrzebujemy
# A*, BFS czy LLM — prosta chciwość (greedy) z opcją cofania wystarczy,
# bo bloki są cykliczne i ZAWSZE się odsuną po kilku turach.
#
def choose_command(player_col, blocks):
    """
    Podejmuje decyzję o kolejnym ruchu na podstawie SYMULACJI stanu planszy.

    Kluczowa idea: nie patrzymy na OBECNY stan, ale na stan PO ruchu.
    Gdy wysyłamy komendę, bloki poruszają się JEDNOCZEŚNIE z robotem.
    Więc musimy sprawdzić gdzie bloki BĘDĄ, nie gdzie SĄ.
    """
    # Symuluj pozycje bloków po jednym kroku
    future_blocks = simulate_blocks(blocks)

    # Priorytet 1: idź w prawo (jeśli kolumna docelowa bezpieczna po ruchu)
    target_col = player_col + 1
    if target_col <= MAX_COL and is_column_safe(target_col, future_blocks):
        return "right"

    # Priorytet 2: czekaj (jeśli bieżąca kolumna bezpieczna po ruchu)
    if is_column_safe(player_col, future_blocks):
        return "wait"

    # Priorytet 3: cofnij się (ucieczka — blok nadjeżdża na naszą kolumnę)
    if player_col > MIN_COL:
        return "left"

    # Skrajny przypadek: jesteśmy w kolumnie 1 i blok nadjeżdża
    # — jedyne co możemy to czekać i liczyć na odbicie
    return "wait"


# ─── Wysyłanie komendy do API ────────────────────────────────────────────────

def send_command(command):
    """
    Wysyła komendę do API reaktora i zwraca odpowiedź.

    To nasz jedyny "kanał komunikacji" z otoczeniem — odpowiednik
    narzędzia (tool) w agencie. Każde wywołanie to jeden krok agentowej
    pętli, a odpowiedź to feedback z otoczenia.
    """
    payload = {
        "apikey": HUB_API_KEY,
        "task": "reactor",
        "answer": {"command": command},
    }
    resp = requests.post(VERIFY_URL, json=payload)
    resp.raise_for_status()
    return resp.json()


# ─── Wizualizacja planszy ─────────────────────────────────────────────────────
#
# Wizualizacja to forma feedbacku DLA NAS (programistów). W kontekście lekcji
# to odpowiednik "obserwacji" i "ewaluacji" — widzimy co agent robi i czy
# podejmuje dobre decyzje. Bez tego debugowanie byłoby ślepe.
#
def print_board(state, step, command):
    """Wyświetla czytelną reprezentację planszy z pozycją robota i bloków."""
    board = state.get("board", [])
    player = state.get("player", {})

    print(f"\n{'─'*30}")
    print(f"Krok {step}: wysłano '{command}'")
    print(f"Robot: kolumna {player.get('col', '?')}, wiersz {player.get('row', '?')}")
    print(f"{'─'*30}")

    for row_idx, row in enumerate(board):
        row_num = row_idx + 1
        display = []
        for cell in row:
            if cell == "P":
                display.append("🤖")
            elif cell == "B":
                display.append("🟥")
            elif cell == "G":
                display.append("🏁")
            else:
                display.append("⬜")
        print(f"  {row_num} │ {' '.join(display)}")

    print(f"    └{'──' * 7}─")
    print(f"      {'  '.join(str(i) for i in range(1, 8))}")


# ─── Główna pętla agenta ─────────────────────────────────────────────────────
#
# To jest serce rozwiązania — pętla percepcja-decyzja-akcja.
#
# W terminologii lekcji:
# - Wyzwalacz (trigger): każda odpowiedź API uruchamia kolejny cykl
# - Stan otoczenia: pozycje bloków + kierunki ruchu
# - Feedback: czy robot przeżył, czy dotarł do celu
# - Proaktywność: robot nie czeka na instrukcje użytkownika, sam decyduje
#
# Porównanie z agentami z lekcji:
# - Agent kalendarza (03_03_calendar): reaguje na zdarzenia otoczenia →
#   nasz robot reaguje na ruch bloków
# - Agent przeglądarkowy (03_03_browser): uczy się na błędach →
#   nasz robot symuluje przyszłość żeby UNIKAĆ błędów
# - Hooki (onStepFinish): po każdym kroku sprawdzamy stan →
#   nasz check po każdej komendzie API
#

def main():
    print("=" * 50)
    print("S03E03 — Robot w reaktorze")
    print("=" * 50)

    # Krok 1: Inicjalizacja — wysyłamy 'start' żeby zacząć grę
    # (odpowiednik onStart hook w agencie)
    state = send_command("start")
    print_board(state, 0, "start")

    # Sprawdzamy czy API poprawnie zainicjowało grę
    if "blocks" not in state or "player" not in state:
        print(f"[BŁĄD] Nieoczekiwana odpowiedź: {json.dumps(state, indent=2)}")
        return

    step = 0

    # Krok 2: Pętla agentowa — powtarzaj aż robot dotrze do celu
    # Każda iteracja to jeden "krok" agenta (onStepStart → onStepFinish)
    while step < MAX_STEPS:
        player_col = state["player"]["col"]
        blocks = state["blocks"]

        # Sprawdź warunek zakończenia (odpowiednik onFinish)
        if state.get("reached_goal"):
            print(f"\n🎉 Robot dotarł do celu w {step} krokach!")
            print(f"Odpowiedź serwera: {state.get('message', '')}")
            return

        # Podejmij decyzję (rozumowanie na podstawie kontekstu)
        command = choose_command(player_col, blocks)

        # Wykonaj akcję (wywołanie narzędzia / tool call)
        state = send_command(command)
        step += 1

        # Feedback z otoczenia (onStepFinish)
        print_board(state, step, command)

        # Obsługa odpowiedzi (odpowiednik error handling w hookach)
        code = state.get("code", 0)
        msg = state.get("message", "")

        # API zwraca kod 0 z flagą gdy robot dotarł do celu
        if "FLG:" in msg or state.get("reached_goal"):
            print(f"\n🎉 Robot dotarł do celu w {step} krokach!")
            print(f"Flaga: {msg}")
            return

        if code != 100:
            print(f"\n[BŁĄD] Kod {code}: {msg}")

            # Jeśli robot zginął — trzeba zacząć od nowa
            # (lekcja mówi: feedback z błędów powinien prowadzić do poprawy)
            if "destroyed" in msg.lower() or "crushed" in msg.lower():
                print("Robot zniszczony! Restartuję...")
                state = send_command("start")
                step = 0
                continue
            break

    # Sprawdź cel po wyjściu z pętli
    if state.get("reached_goal"):
        print(f"\n🎉 Robot dotarł do celu w {step} krokach!")
        print(f"Odpowiedź serwera: {state.get('message', '')}")
    else:
        print(f"\n[TIMEOUT] Nie dotarłem do celu w {MAX_STEPS} krokach.")
        print(f"Ostatni stan: kolumna {state.get('player', {}).get('col', '?')}")


if __name__ == "__main__":
    main()
