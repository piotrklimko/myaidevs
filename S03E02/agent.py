import os
from dotenv import load_dotenv
load_dotenv()

# ============================================================
# S03E02 — Firmware: agent do debugowania VM sterownika ECCS
# ============================================================
#
# Lekcja S03E02 omawia OGRANICZENIA MODELI na etapie założeń projektu.
# Główna myśl: zamiast budować "all-in-one" systemy, lepiej tworzyć
# wyspecjalizowane narzędzia z programistycznie kontrolowanym zakresem.
#
# Zadanie "firmware" doskonale ilustruje kilka konceptów z lekcji:
#
# 1. DEFINIOWANIE ROLI I ZAANGAŻOWANIA AI (sekcja 1 lekcji)
#    Lekcja: "jak stworzyć rozwiązanie, które wniesie wysoką wartość,
#    przy zminimalizowaniu ryzyk oraz negatywnego wpływu ograniczeń?"
#    Tu agent ma JEDNĄ rolę: debugger firmware. Nie jest czatbotem,
#    nie odpowiada na pytania — eksploruje, diagnozuje, naprawia.
#    To realizacja zasady: "dedykowane i wyspecjalizowane narzędzia".
#
# 2. KONTROLOWANIE POZIOMU TRUDNOŚCI (sekcja 2 lekcji)
#    Lekcja opisuje heartbeat pattern i plany zadań. Nasz agent
#    to uproszczony wariant: prompt systemowy definiuje KROKI
#    (plan), a pętla agentowa realizuje heartbeat — po każdym
#    cyklu sprawdzamy stan i kontynuujemy. Agent sam sekwencjonuje
#    kroki: help → eksploracja → diagnoza → naprawa → uruchomienie.
#
# 3. ZMNIEJSZANIE RYZYKA PROMPT INJECTION (sekcja 3 lekcji)
#    Lekcja: "nie możemy ufać agentom w zakresie udostępniania
#    informacji oraz pracy na zewnętrznych źródłach danych".
#    Tu ryzyko to naruszenie czarnej listy (ban + reset VM).
#    Adresujemy je PROGRAMISTYCZNIE: opis narzędzia shell_cmd
#    explicite wymienia zakazy, a prompt powtarza je w KROKACH.
#    Mimo to agent nie jest nieomylny — dlatego warstwa transportowa
#    obsługuje bany transparentnie (czeka i ponawia).
#
# 4. GENEROWANIE I WYKONYWANIE KODU (sekcja 4 lekcji)
#    Lekcja opisuje agenta z sandboxem Deno do przetwarzania danych.
#    Tu mamy analogię: shell VM to nasz sandbox — ograniczone
#    środowisko z kontrolowanymi uprawnieniami. Agent "generuje"
#    komendy (editline, rm, uruchomienie binarki), a VM je wykonuje.
#    Lekcja: "konfiguracja narzędzi w postaci systemu plików oraz
#    wykonania kodu sprawia, że agent posiada znacznie większe
#    możliwości" — tu wystarczą ls, cat, editline, rm.
#
# 5. WSKAZÓWKI Z ZADANIA — PODEJŚCIE AGENTOWE
#    Zadanie wprost mówi: "idealnie nadaje się do pętli agentowej
#    z Function Calling". Agent potrzebuje jednego narzędzia (shell)
#    i jednego do wysyłania odpowiedzi. Każde wywołanie to jedno
#    zapytanie HTTP — sekwencyjnie, krok po kroku.
#
# ARCHITEKTURA:
#    [Claude Sonnet 4.6] ←→ [Python agent loop] ←→ [Shell API VM]
#                                    ↓
#                              [Hub /verify]
#
# Agent realizuje wzorzec z lekcji S02E04 (Orchestrator):
# jeden agent, pętla z function calling, iteracja na feedbacku.
# Różnica: tu feedback to kody błędów z firmware (lock file,
# brak SAFETY_CHECK, test_mode), nie z huba weryfikacyjnego.

import json
import time
import requests
from openai import OpenAI

HUB_API_KEY = os.environ["HUB_API_KEY"]
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
SHELL_URL = "https://hub.ag3nts.org/api/shell"
VERIFY_URL = "https://hub.ag3nts.org/verify"

# Wskazówka z zadania: "Spróbuj użyć anthropic/claude-sonnet-4-6 —
# jego zdolność do śledzenia kontekstu i adaptacji do nieznanego API
# robi tutaj dużą różnicę." Słabsze modele mogą utknąć w pętli
# lub pomylić komendy niestandardowego shella.
MODEL = "anthropic/claude-sonnet-4-6"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# Opóźnienie między zapytaniami do Shell API — szanowanie rate limitów.
# Wzorzec z S01E05 (railway): prewencyjny backoff jest tańszy niż retry.
API_DELAY = 1.0


# ============================================================
# WARSTWA TRANSPORTOWA — obsługa banów i rate limitów
# ============================================================
#
# Lekcja S03E02 mówi o "programistycznych ograniczeniach, które
# w dużym stopniu adresują problem Prompt Injection". Tu ograniczenia
# narzuca VM: dotknięcie zabronionych ścieżek = tymczasowy ban.
#
# Shell API zwraca specyficzne kody błędów:
# - code=-733: Security policy violation — agent dotknął zakazanego
#   pliku/katalogu. VM zostaje zresetowana, ban na N sekund.
# - code=-735: Wciąż zbanowany — próba komendy w trakcie bana.
#
# Wskazówka z zadania: "Zadbaj o to, żeby agent widział te kody
# i mógł na nie zareagować — np. poczekać i spróbować ponownie.
# Możesz też obsługę tych błędów zaimplementować bezpośrednio
# w narzędziu, a agentowi odsyłać bardziej opisowe komunikaty."
#
# Wybieramy drugie podejście: warstwa transportowa TRANSPARENTNIE
# obsługuje bany (czeka + ponawia). Agent nie wie o banie — dostaje
# wynik jakby nic się nie stało. To analogia do obsługi rate limitów
# w S02E04 (zmail): warstwa Pythona chroni agenta przed detalami
# infrastrukturalnymi, żeby mógł skupić się na zadaniu.
#
# UWAGA: po banie VM jest resetowana do stanu początkowego,
# więc wszelkie zmiany agenta (editline, rm) zostają cofnięte.
# Agent musi być gotowy na powtórzenie kroków — dlatego prompt
# zawiera pełną strategię, nie tylko "kontynuuj od miejsca X".

def shell_request(cmd: str) -> dict:
    payload = {"apikey": HUB_API_KEY, "cmd": cmd}
    r = requests.post(SHELL_URL, json=payload, timeout=30)
    data = r.json()

    # Obsługa bana — czekamy i ponawiamy
    if data.get("code") in (-733, -735):
        ban_info = data.get("ban", {})
        wait = ban_info.get("ttl_seconds") or ban_info.get("seconds_left") or 25
        print(f"     [BAN: {ban_info.get('reason', '?')} — czekam {wait}s]")
        time.sleep(wait + 2)
        # Po banie VM jest zresetowana — ponów komendę
        r = requests.post(SHELL_URL, json=payload, timeout=30)
        data = r.json()

    time.sleep(API_DELAY)
    return data


# ============================================================
# NARZĘDZIA — minimalistyczny zestaw
# ============================================================
#
# Lekcja S03E02 opisuje agenta z dwoma typami narzędzi:
# - SYSTEM PLIKÓW (ls, cat, editline, rm — przez shell_cmd)
# - WYKONANIE KODU (uruchomienie binarki — też przez shell_cmd)
#
# Lekcja: "konfiguracja narzędzi w postaci systemu plików oraz
# wykonania kodu domyślnie sprawia, że agent posiada znacznie
# większe możliwości, które wykraczają poza [konkretne zadanie]."
#
# Tu mamy JEDNO narzędzie shell_cmd, które daje dostęp do obu
# tych typów. To świadomy wybór: shell VM oferuje ograniczony
# zestaw komend (help, ls, cat, editline, rm, find, reboot...),
# więc nie ma potrzeby rozbijać tego na osobne narzędzia.
# Agent sam odkrywa dostępne komendy przez `help`.
#
# Drugie narzędzie (submit_answer) to standardowy wzorzec
# z poprzednich zadań — akcja finalna, oddzielona od eksploracji.

def shell_cmd(cmd: str) -> str:
    """Wykonaj komendę w shellu VM."""
    result = shell_request(cmd)
    return json.dumps(result, ensure_ascii=False, indent=2)


def submit_answer(confirmation: str) -> str:
    """Wyślij kod ECCS do weryfikacji."""
    payload = {
        "apikey": HUB_API_KEY,
        "task": "firmware",
        "answer": {"confirmation": confirmation},
    }
    r = requests.post(VERIFY_URL, json=payload, timeout=30)
    return json.dumps(r.json(), ensure_ascii=False, indent=2)


# ============================================================
# SCHEMATY NARZĘDZI DLA FUNCTION CALLING
# ============================================================
#
# Opis shell_cmd to KRYTYCZNY element kontekstu — lekcja S01E03:
# "opis explicite pokazuje czego narzędzie oczekuje". Zawiera:
#
# 1. JAK ODKRYĆ KOMENDY: "poznasz przez `help`" — agent nie
#    zakłada standardowego Linuxa. Wskazówka z zadania: "Shell API
#    na tej maszynie wirtualnej ma niestandardowy zestaw komend.
#    Nie zakładaj, że wszystkie standardowe polecenia zadziałają."
#
# 2. JAK EDYTOWAĆ: "editline <file> <line> <content>" — explicite,
#    bo agent mógłby próbować `vim` lub `sed` (których nie ma).
#    Wskazówka z zadania: "edycja pliku odbywa się inaczej niż
#    w standardowym systemie."
#
# 3. CZEGO NIE ROBIĆ: "ZABRONIONE: /etc, /root, /proc..." —
#    to PROGRAMISTYCZNA BARIERA w promptcie. Lekcja S03E02:
#    "agenci powinni mieć zablokowaną bądź wysoce nadzorowaną
#    możliwość kontaktu ze światem zewnętrznym". Tu "świat
#    zewnętrzny" to zabronione katalogi systemowe.
#
# UWAGA: mimo jasnego opisu, agent MOŻE naruszyć zakazy
# (prompt injection, halucynacja, logiczny błąd). Dlatego
# warstwa transportowa obsługuje bany jako fallback.
# To realizacja zasady z lekcji: "kilka rodzajów barier,
# które zmniejszą ryzyko problemów" — prompt + kod.

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "shell_cmd",
            "description": (
                "Wykonaj komendę w shellu maszyny wirtualnej ze sterownikiem ECCS. "
                "Dostępne komendy poznasz przez `help`. Komendy niestandardowe — "
                "edycja pliku przez `editline <file> <line> <content>`, nie vim/nano. "
                "ZABRONIONE: /etc, /root, /proc oraz pliki/katalogi z .gitignore. "
                "Naruszenie = ban + reset VM."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {
                        "type": "string",
                        "description": "Komenda do wykonania w shellu VM",
                    },
                },
                "required": ["cmd"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": "Wyślij kod ECCS do centrali. Format: ECCS-<40 znaków hex>.",
            "parameters": {
                "type": "object",
                "properties": {
                    "confirmation": {
                        "type": "string",
                        "description": "Kod ECCS uzyskany po uruchomieniu cooler.bin",
                    },
                },
                "required": ["confirmation"],
            },
        },
    },
]


# ============================================================
# PROMPT SYSTEMOWY — rola wyspecjalizowanego debuggera
# ============================================================
#
# Lekcja S03E02 wielokrotnie podkreśla wartość WYSPECJALIZOWANYCH
# narzędzi vs "all-in-one" systemów. Prompt realizuje tę zasadę:
# agent ma JEDNĄ rolę (debugger firmware), JEDEN cel (uruchomić
# cooler.bin) i KONKRETNY plan kroków.
#
# Struktura promptu odzwierciedla wzorzec z lekcji:
#
# CEL — jednoznaczny, mierzalny ("uzyskać kod ECCS-xxx").
#   Lekcja: agent powinien wiedzieć kiedy jego zadanie jest DONE.
#
# KROKI — sekwencyjny plan, ale z elastycznością.
#   Lekcja opisuje "plan składający się z zadań posiadających
#   nazwy, opisy, statusy, zależności". Tu uproszczone do listy
#   kroków, bo zadanie jest liniowe (nie ma zależności równoległych).
#   Agent sam decyduje o szczegółach (jakie pliki czytać, co edytować).
#
# OGRANICZENIA — explicite wymienione zakazy.
#   Lekcja: "Prompt Systemowy powinien być traktowany jako publicznie
#   dostępny. Nie może więc w nim być jakichkolwiek danych, które po
#   zdobyciu mogą być wykorzystane w nieodpowiedni sposób."
#   Tu zakazy (.gitignore, /etc, /root, /proc) nie są tajemnicą —
#   to reguły bezpieczeństwa, które agent MUSI znać.
#
# WERYFIKACJA — "po każdym editline weryfikuj przez cat".
#   To realizacja wzorca z sekcji 4 lekcji: "agenci odpowiedzialni
#   za tak istotne dokumenty powinni pozostawać pod ścisłym nadzorem,
#   a ich logika powinna obejmować jasne procesy sterowane za pomocą
#   kodu wszędzie tam, gdzie to możliwe." Tu agent sam weryfikuje
#   efekt swojej pracy — programistyczny self-check.
#
# REBOOT — wentyl bezpieczeństwa.
#   Lekcja o heartbeat: "możliwe są też sytuacje gdy zadanie w trakcie
#   realizacji zmienia swój status na oczekujący". Tu `reboot` pozwala
#   agentowi wrócić do stanu wyjściowego, jeśli namieszał w systemie.

SYSTEM_PROMPT = """Jesteś agentem debugującym oprogramowanie sterownika ECCS na maszynie wirtualnej.

CEL: Uruchomić /opt/firmware/cooler/cooler.bin i uzyskać kod ECCS-xxx, a następnie wysłać go przez submit_answer.

KROKI:
1. Zacznij od `help` — poznaj dostępne komendy (to niestandardowy shell!)
2. Eksploruj system plików: ls /, ls /home, ls /opt/firmware/cooler/ itd.
3. Przeczytaj .gitignore w katalogu cooler — NIGDY nie dotykaj wymienionych tam plików/katalogów
4. NIE zaglądaj do /etc, /root, /proc — to spowoduje bana!
5. Spróbuj uruchomić cooler.bin — reaguj na komunikaty błędów
6. Szukaj hasła w systemie (np. /home/operator/, .bash_history, notatki)
7. Popraw settings.ini używając `editline` — odczytaj plik, policz linie, edytuj właściwą
8. Usuń pliki blokujące jeśli trzeba (lock files)
9. Po uzyskaniu kodu ECCS → submit_answer

WAŻNE:
- Edycja plików WYŁĄCZNIE przez `editline <plik> <numer_linii> <nowa_treść>`
- Po każdym editline weryfikuj wynik przez cat
- Linie numerowane od 1
- Puste linie też się liczą przy numerowaniu!
- Jeśli coś pójdzie nie tak: `reboot` resetuje VM"""


# Dispatch table — mapowanie nazw narzędzi na implementacje.
# Wzorzec identyczny jak w S02E04 (mailbox) — prosty dict
# z lambdami, bo mamy tylko dwa narzędzia.
TOOL_DISPATCH = {
    "shell_cmd": lambda args: shell_cmd(**args),
    "submit_answer": lambda args: submit_answer(**args),
}


# ============================================================
# PĘTLA AGENTA — heartbeat z function calling
# ============================================================
#
# Lekcja S03E02 opisuje wzorzec "heartbeat" — logikę pełniącą
# rolę managera, który przydziela zadania i aktualizuje stan.
# Tu uproszczona wersja: pętla while z wywołaniami LLM.
#
# Każda iteracja to jeden "heartbeat":
# 1. LLM analizuje dotychczasowy kontekst (historię komend + wyniki)
# 2. Decyduje o następnym kroku (tool_call lub zakończenie)
# 3. Python wykonuje narzędzie i dodaje wynik do kontekstu
#
# Lekcja: "wystarczy, że status systemu będzie sprawdzany po
# ukończeniu każdego z cykli aż do ukończenia wszystkich zadań."
# Tu "status systemu" to kody odpowiedzi z Shell API:
# - code=195 "Trying to execute" + komunikat błędu = trzeba naprawić
# - code=196 "Executed file" + ECCS-xxx = sukces
# - code=-691..-689 "Configuration check failed" = konkretny problem
#
# TRANSPORT WIEDZY — wyniki narzędzi trafiają do messages[],
# dzięki czemu agent pamięta co już zrobił. To ten sam wzorzec
# co w S02E04: "w trakcie sesji kluczowe informacje i postępy
# są dostępne w kontekście".
#
# 25 ITERACJI — wystarczający budżet na:
# - 3-4 iteracje na eksplorację (help, ls, cat .gitignore)
# - 3-4 iteracje na diagnozę (cat settings.ini, bash_history, pass.txt)
# - 3-4 iteracje na naprawę (rm lock, editline ×3, cat weryfikacja)
# - 1-2 iteracje na uruchomienie + submit
# - zapas na ewentualne błędy i powtórki
#
# WCZESNE WYJŚCIE NA FLADZE — jak w każdym zadaniu: po znalezieniu
# {FLG:...} w odpowiedzi huba natychmiast kończymy. Oszczędność
# tokenów i zapytań API.

def run_agent():
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Uruchom sterownik ECCS na maszynie wirtualnej. Zdiagnozuj i napraw problemy."},
    ]

    print("=== AGENT FIRMWARE START ===\n")

    for iteration in range(25):
        print(f"--- Iteracja {iteration + 1} ---")

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        msg = response.choices[0].message
        messages.append(msg)

        # Agent może wypowiedzieć się tekstowo (rozumowanie, plan)
        # zanim wywoła narzędzie — lub zamiast niego (zakończenie).
        if msg.content:
            print(f"Agent: {msg.content[:400]}")

        # Brak tool_calls = agent uznał zadanie za ukończone
        # (lub utknął — ale przy Sonnet 4.6 to rzadkie).
        if not msg.tool_calls:
            print("\n=== Agent zakończył ===")
            if msg.content:
                print(msg.content)
            break

        # Wykonanie narzędzi — sekwencyjne, bo komendy shellowe
        # zmieniają stan VM (np. rm lock → editline → run).
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            print(f"  >> {fn_name}({json.dumps(args, ensure_ascii=False)[:200]})")

            handler = TOOL_DISPATCH.get(fn_name)
            if handler:
                result = handler(args)
            else:
                result = json.dumps({"error": f"Nieznane narzędzie: {fn_name}"})

            # Sprawdź flagę przy submit — wczesne wyjście
            if fn_name == "submit_answer":
                print(f"\n  *** HUB: {result} ***\n")
                if "FLG" in result:
                    print("\n=== FLAGA ZNALEZIONA! ===")
                    print(result)
                    return

            preview = result[:600] + ("..." if len(result) > 600 else "")
            print(f"     => {preview}")

            # Wynik narzędzia wraca do kontekstu agenta —
            # agent widzi efekt każdej komendy i reaguje.
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    print("\n=== KONIEC AGENTA ===")


if __name__ == "__main__":
    run_agent()
