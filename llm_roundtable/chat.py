import os
from dotenv import load_dotenv
load_dotenv()

"""
LLM Roundtable — 4 modele rozmawiają ze sobą.

Skrypt symuluje okrągły stół dyskusyjny, przy którym zasiadają cztery różne modele
językowe (LLM). Każdy z nich ma przypisaną "osobowość" (persona) i odpowiada kolejno
na wypowiedzi pozostałych — jak prawdziwa rozmowa grupowa.

Jak to działa (przepływ):
    1. Użytkownik podaje temat i liczbę rund.
    2. Skrypt iteruje rundy; w każdej rundzie każdy z 4 agentów zabiera głos raz.
    3. Przed każdą odpowiedzią agent dostaje pełną historię dotychczasowej rozmowy
       — dzięki temu "wie", co powiedzieli poprzednicy, i może na to reagować.
    4. Odpowiedź agenta trafia do wspólnej historii i jest widoczna dla kolejnych agentów
       w tej samej rundzie oraz we wszystkich następnych.

Użycie:
    python chat.py                          # domyślny temat, 3 rundy
    python chat.py "Czy AI jest świadome?"  # własny temat
    python chat.py --rounds 5               # własna liczba rund
    python chat.py "Temat" --rounds 2       # oba parametry naraz
"""

import sys
import time
import argparse
from openai import OpenAI

# ── konfiguracja klienta API ──────────────────────────────────────────────────
#
# Używamy biblioteki `openai` (oficjalny Python SDK od OpenAI), ale kierujemy ją
# na OpenRouter — bramkę, która obsługuje dziesiątki różnych modeli pod jednym
# wspólnym interfejsem (zgodnym ze standardem OpenAI Chat Completions API).
#
# Dzięki temu ten sam kod działa zarówno dla Claude, GPT, Llamy, jak i Gemini —
# różni się tylko wartość pola `model` w każdym zapytaniu.

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",   # OpenRouter zamiast api.openai.com
    api_key=OPENROUTER_API_KEY,
)

# ── kolory ANSI dla terminala ─────────────────────────────────────────────────
#
# Sekwencje sterujące ANSI pozwalają kolorować tekst w terminalu bez zewnętrznych
# bibliotek (jak `rich` czy `colorama`). Format: \033[ + kod + m
#
#   \033[0m  — reset (wyłącza wszystkie atrybuty)
#   \033[1m  — pogrubienie
#   \033[9Xm — kolor jasny: 94=niebieski, 92=zielony, 93=żółty, 95=magenta
#
# Każdy agent dostaje swój kolor z listy COLORS według indeksu.

RESET  = "\033[0m"
BOLD   = "\033[1m"
COLORS = [
    "\033[94m",  # niebieski  → Claude
    "\033[92m",  # zielony    → GPT
    "\033[93m",  # żółty      → Llama
    "\033[95m",  # magenta    → Gemini
]

# ── definicje agentów ─────────────────────────────────────────────────────────
#
# AGENTS to lista słowników. Każdy słownik opisuje jednego uczestnika dyskusji:
#
#   name    — wyświetlana nazwa, pojawia się też w historii rozmowy w formacie
#             "[Nazwa]: treść", żeby inne modele wiedziały, kto co powiedział.
#
#   model   — identyfikator modelu w formacie OpenRouter: "dostawca/nazwa-modelu".
#             OpenRouter tłumaczy to na konkretne API (Anthropic, OpenAI, Meta, Google).
#
#   persona — treść wiadomości systemowej (role=system). To "instrukcja osobowości":
#             mówi modelowi, jak ma się zachowywać, jaki ma styl wypowiedzi i jak długo
#             ma odpowiadać. Każdy model dostaje inną personę → różne punkty widzenia
#             w dyskusji, co czyni rozmowę ciekawszą niż gdyby wszystkie mówiły jednakowo.
#
# Kolejność w liście = kolejność zabierania głosu w każdej rundzie.

AGENTS = [
    {
        "name": "Claude",
        "model": "anthropic/claude-haiku-4.5",
        "persona": (
            "Jesteś Claude — rozważnym i empatycznym rozmówcą. "
            "Lubisz widzieć wiele perspektyw i zachęcasz do głębszego myślenia. "
            "Odpowiadasz krótko (2-4 zdania), na temat dyskusji."
        ),
    },
    {
        "name": "GPT",
        "model": "openai/gpt-4o-mini",
        "persona": (
            "Jesteś GPT — precyzyjnym i analitycznym rozmówcą. "
            "Lubisz fakty, strukturę i konkretne argumenty. "
            "Odpowiadasz krótko (2-4 zdania), na temat dyskusji."
        ),
    },
    {
        "name": "Llama",
        "model": "meta-llama/llama-3.1-8b-instruct",
        "persona": (
            "Jesteś Llama — bezpośrednim i praktycznym rozmówcą open-source. "
            "Lubisz proste wyjaśnienia i realne zastosowania. "
            "Odpowiadasz krótko (2-4 zdania), na temat dyskusji."
        ),
    },
    {
        "name": "Gemini",
        "model": "google/gemini-2.0-flash-001",
        "persona": (
            "Jesteś Gemini — kreatywnym i wielomodalnym rozmówcą. "
            "Lubisz nieoczekiwane połączenia idei i analogie. "
            "Odpowiadasz krótko (2-4 zdania), na temat dyskusji."
        ),
    },
]

# ── budowanie kontekstu wiadomości ────────────────────────────────────────────

def build_messages(agent: dict, topic: str, history: list[dict]) -> list[dict]:
    """
    Buduje listę wiadomości wysyłaną do API dla konkretnego agenta.

    Każde wywołanie API modelu językowego wymaga listy wiadomości w formacie:
        [
            {"role": "system",    "content": "..."},  # instrukcja systemowa
            {"role": "user",      "content": "..."},  # wiadomość użytkownika
            {"role": "assistant", "content": "..."},  # odpowiedź asystenta
            ...                                        # dalszy dialog
        ]

    W naszym przypadku:
    - Wiadomość systemowa łączy personę agenta z tematem dyskusji i instrukcją
      reagowania na poprzedników.
    - Reszta to `history` — lista dotychczasowych wypowiedzi wszystkich agentów,
      zakodowana jako wiadomości role=user z prefiksem "[Nazwa]: ".

    Dlaczego role=user dla historii (a nie naprzemiennie user/assistant)?
    Ponieważ historia zawiera wypowiedzi RÓŻNYCH agentów, nie naprzemienną rozmowę
    między jednym użytkownikiem a jednym asystentem. Użycie role=user dla wszystkich
    wpisów historii jest uproszczeniem, które działa dobrze w praktyce — model rozumie
    z prefiksów "[Claude]:", "[GPT]:" itp., że to różni rozmówcy.

    Parametry:
        agent   — słownik agenta (name, model, persona)
        topic   — temat dyskusji (wstrzykiwany do systemu)
        history — lista dotychczasowych wiadomości (wspólna dla wszystkich agentów)

    Zwraca:
        Gotową listę wiadomości do przekazania do client.chat.completions.create().
    """
    messages = [
        {
            "role": "system",
            "content": (
                f"{agent['persona']}\n\n"
                f"Temat dyskusji: {topic}\n"
                "Rozmawiasz z innymi modelami AI. Reaguj na to, co powiedzieli poprzednicy, "
                "wnosząc coś nowego. Nie powtarzaj cudzych słów dosłownie."
            ),
        }
    ]
    # Dołączamy całą historię rozmowy — agent "widzi" wszystko, co powiedziano wcześniej.
    # Im dłuższa dyskusja, tym więcej tokenów kontekstu jest zużywane przy każdym wywołaniu.
    messages.extend(history)
    return messages


# ── wywołanie modelu ──────────────────────────────────────────────────────────

def ask_agent(agent: dict, topic: str, history: list[dict]) -> str:
    """
    Wysyła zapytanie do konkretnego modelu przez OpenRouter i zwraca jego odpowiedź.

    Parametry wywołania API:
        model       — identyfikator modelu (np. "anthropic/claude-haiku-4.5")
        messages    — lista wiadomości zbudowana przez build_messages()
        max_tokens  — limit długości odpowiedzi (300 tokenów ≈ ok. 200-250 słów)
        temperature — losowość odpowiedzi: 0.0 = deterministyczny, 1.0 = bardzo kreatywny.
                      Wartość 0.8 daje odpowiedzi zróżnicowane, ale wciąż spójne.

    Obsługa błędów:
        Jeśli API zwróci błąd (np. nieznany model, przekroczony limit, brak środków),
        zamiast przerywać całą dyskusję, zwracamy czytelny komunikat o błędzie.
        Ten komunikat trafia do historii i pozostałe modele "widzą", że dany agent
        nie odpowiedział — rozmowa toczy się dalej.

    Zwraca:
        Tekst odpowiedzi modelu (stripped, bez zbędnych białych znaków).
    """
    messages = build_messages(agent, topic, history)
    try:
        response = client.chat.completions.create(
            model=agent["model"],
            messages=messages,
            max_tokens=300,
            temperature=0.8,
        )
        # response.choices[0].message.content — standardowe pole w OpenAI API
        # .strip() usuwa wiodące/kończące spacje i newliny
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[błąd: {e}]"


# ── wyświetlanie wypowiedzi ───────────────────────────────────────────────────

def print_agent(agent_idx: int, agent: dict, text: str) -> None:
    """
    Wypisuje wypowiedź agenta w kolorowej ramce do terminala.

    Wizualny format każdej wypowiedzi:
        ┌─ Nazwa (model/identyfikator)
        ────────────────────────────────
        │ linia 1 tekstu odpowiedzi
        │ linia 2 tekstu odpowiedzi
        ────────────────────────────────

    Prefix "│ " przy każdej linii sprawia, że tekst wygląda jak blok cytatu
    i jest wizualnie oddzielony od wypowiedzi innych agentów.

    Kolorowanie działa tak, że cały "obramowanie" (┌─, ─, │) jest w kolorze agenta,
    ale sam tekst odpowiedzi jest w domyślnym kolorze terminala (RESET po "│ ").
    Dzięki temu kolorowe jest "opakowanie", a treść jest łatwa do czytania.

    Parametry:
        agent_idx — indeks agenta w liście AGENTS (decyduje o kolorze)
        agent     — słownik agenta (name, model)
        text      — tekst do wyświetlenia (może zawierać newliny)
    """
    color = COLORS[agent_idx % len(COLORS)]   # % len zabezpiecza przed wyjściem poza zakres
    separator = "─" * 60
    print(f"\n{color}{BOLD}┌─ {agent['name']} ({agent['model']}){RESET}")
    print(f"{color}{separator}{RESET}")
    # Iterujemy po liniach, żeby każda zaczynała się od kolorowego "│ "
    for line in text.split("\n"):
        print(f"{color}│ {RESET}{line}")
    print(f"{color}{separator}{RESET}")


# ── główna pętla dyskusji ─────────────────────────────────────────────────────

def run_roundtable(topic: str, rounds: int = 3) -> None:
    """
    Uruchamia pełną dyskusję roundtable.

    Struktura pętli:
        for runda in 1..rounds:
            for agent in AGENTS:
                1. Wyświetl animację oczekiwania ("myśli...")
                2. Wyślij do API: persona agenta + temat + cała historia dotąd
                3. Odbierz odpowiedź
                4. Dopisz odpowiedź do wspólnej historii
                5. Wyświetl odpowiedź w kolorowej ramce
                6. Krótka pauza (rate limiting)

    Kluczowy mechanizm — wspólna historia (`history`):
        Lista `history` jest WSPÓLNA dla wszystkich agentów i rośnie przez całą
        dyskusję. Każda nowa wypowiedź jest dodawana jako:
            {"role": "user", "content": "[Nazwa]: treść"}

        Przed każdym wywołaniem agent dostaje tę samą listę + swoją personę systemową.
        Oznacza to, że:
        - Pierwszy agent w rundzie 1 nie widzi żadnych poprzednich wypowiedzi.
        - Drugi agent widzi 1 wypowiedź (pierwszego agenta).
        - Pierwszy agent w rundzie 2 widzi wszystkie N*1 wypowiedzi z rundy 1, itd.

        To jest klasyczny wzorzec "shared context window" — wszyscy czytają ten sam
        dziennik rozmowy, każdy dopisuje swój wpis na koniec.

    Parametry:
        topic  — temat dyskusji (przekazywany do każdego build_messages())
        rounds — liczba rund (ile razy każdy agent zabiera głos)
    """
    # Nagłówek dyskusji
    print(f"\n{'═' * 60}")
    print(f"{BOLD}  LLM ROUNDTABLE{RESET}")
    print(f"{'═' * 60}")
    print(f"  Temat: {BOLD}{topic}{RESET}")
    print(f"  Modele: {', '.join(a['name'] for a in AGENTS)}")
    print(f"  Rundy: {rounds}")
    print(f"{'═' * 60}\n")

    # Wspólna historia rozmowy — rośnie przez całą dyskusję.
    # Każdy element to dict {"role": "user", "content": "[Nazwa]: tekst"}.
    history: list[dict] = []

    for rnd in range(1, rounds + 1):
        print(f"\n{BOLD}{'━' * 60}{RESET}")
        print(f"{BOLD}  RUNDA {rnd}/{rounds}{RESET}")
        print(f"{BOLD}{'━' * 60}{RESET}")

        for idx, agent in enumerate(AGENTS):
            # Pokaż wskaźnik oczekiwania — flush=True wymusza natychmiastowy wydruk
            # (bez buforowania), end="" zapobiega newline, bo zaraz nadpiszemy tę linię
            print(f"\n  {COLORS[idx]}⏳ {agent['name']} myśli...{RESET}", end="", flush=True)

            # Główne wywołanie API — może trwać 1-10 sekund
            reply = ask_agent(agent, topic, history)

            # Wyczyść linię "myśli..." przez nadpisanie spacjami i powrót na początek linii.
            # \r (carriage return) przesuwa kursor na początek bieżącej linii bez nowej linii.
            print("\r" + " " * 50 + "\r", end="", flush=True)

            # Dopisz odpowiedź do historii — od tej chwili kolejne agenty ją "widzą".
            # Prefiks "[Nazwa]:" w treści pozwala modelom rozróżnić, kto co powiedział,
            # mimo że wszystkie wpisy mają role="user".
            history.append({
                "role": "user",
                "content": f"[{agent['name']}]: {reply}",
            })

            # Wyświetl odpowiedź w sformatowanej ramce
            print_agent(idx, agent, reply)

            # Krótka pauza między zapytaniami — zapobiega przekroczeniu rate limitów
            # OpenRouter przy szybkich sekwencyjnych wywołaniach.
            time.sleep(0.3)

    print(f"\n{'═' * 60}")
    print(f"{BOLD}  Koniec dyskusji.{RESET}")
    print(f"{'═' * 60}\n")


# ── punkt wejścia skryptu ─────────────────────────────────────────────────────
#
# Blok `if __name__ == "__main__"` sprawia, że ten kod uruchamia się tylko przy
# bezpośrednim wywołaniu skryptu (`python chat.py`), a NIE przy imporcie modułu
# w innym pliku. To standardowy Python idiom.
#
# argparse obsługuje argumenty linii poleceń:
#   topic       — pozycyjny, opcjonalny (nargs="?"), domyślnie DEFAULT_TOPIC
#   --rounds/-r — opcja nazwana, typ int, domyślnie 3

DEFAULT_TOPIC = "Czy modele językowe mogą być naprawdę kreatywne, czy tylko naśladują kreatywność?"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLM Roundtable — 4 modele dyskutują ze sobą na wybrany temat"
    )
    parser.add_argument(
        "topic",
        nargs="?",           # argument opcjonalny (0 lub 1 wartość)
        default=DEFAULT_TOPIC,
        help="Temat dyskusji (w cudzysłowie jeśli zawiera spacje)",
    )
    parser.add_argument(
        "--rounds", "-r",
        type=int,
        default=3,
        help="Liczba rund — ile razy każdy model zabiera głos (domyślnie: 3)",
    )
    args = parser.parse_args()

    run_roundtable(topic=args.topic, rounds=args.rounds)
