import os
from dotenv import load_dotenv
load_dotenv()

# ============================================================
# S02E04 — Organizowanie kontekstu dla wielu wątków
# ============================================================
#
# Lekcja S02E04 skupia się na systemach wieloagentowych i zarządzaniu
# kontekstem wykraczającym poza pojedynczą sesję. Choć zadanie "mailbox"
# rozwiązane jest jednym agentem, demonstruje kluczowe koncepty lekcji:
#
# 1. ARCHITEKTURA ORCHESTRATOR
#    Lekcja opisuje kilka architektur: Pipeline, Blackboard, Orchestrator,
#    Tree, Mesh, Swarm. Tu mamy wariant Orchestratora: jeden agent
#    koordynuje pracę, sam decyduje o kolejności kroków, reaguje na
#    feedback i iteruje. W rozbudowanej wersji mógłby delegować
#    wyszukiwanie do subagentów (delegate/message z lekcji).
#
# 2. DWUETAPOWE POBIERANIE DANYCH
#    API zmail działa jak Gmail: najpierw search/getInbox → metadane
#    (bez treści), potem getMessages → pełna treść. To realizacja
#    konceptu "nawigacji" z S02E03: agent nie pobiera WSZYSTKIEGO,
#    lecz nawiguje krok po kroku (perspektywa → nawigacja → szczegóły).
#
# 3. AKTYWNA SKRZYNKA = DYNAMICZNY KONTEKST
#    Skrzynka jest "żywa" — nowe maile napływają w trakcie pracy agenta.
#    Lekcja S02E04 ostrzega: "dynamiczny charakter środowiska, w którym
#    działają agenci, w połączeniu z wieloznacznością języka naturalnego
#    wcale nie pomaga". Agent musi zakładać, że dane się zmieniają
#    i ponownie szukać, jeśli nie znalazł czegoś za pierwszym razem.
#
# 4. SAMO-DOKUMENTUJĄCE API
#    Akcja "help" zwraca dokumentację API — ten sam wzorzec co w S01E05
#    (railway). Agent poznaje swoje możliwości dynamicznie, nie z góry.
#    Lekcja: "agent zaczyna od poznania dostępnych akcji, nie od zgadywania".
#
# 5. ITERACJA NA FEEDBACKU Z HUBA
#    Hub zwraca info "brakuje hasła" lub "kod nieprawidłowy" → agent
#    poprawia i wysyła ponownie. To ten sam wzorzec iteracyjny co
#    w S02E01 (prompt engineering) i S02E03 (kompresja logów).

import json
import time
import requests
from openai import OpenAI

HUB_API_KEY = os.environ["HUB_API_KEY"]
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
ZMAIL_URL = "https://hub.ag3nts.org/api/zmail"
VERIFY_URL = "https://hub.ag3nts.org/verify"
MODEL = "anthropic/claude-haiku-4.5"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# Opóźnienie między zapytaniami — szanowanie rate limitów API (S01E05).
# Bez tego agent może wysłać kilka zapytań w ułamku sekundy i dostać
# code=-9999 (rate limit). Prewencyjny backoff jest tańszy niż retry.
API_DELAY = 1.5


# ============================================================
# WARSTWA TRANSPORTOWA — obsługa rate limitów i retry
# ============================================================
#
# Wszystkie narzędzia korzystają z jednej funkcji zmail_request(),
# która obsługuje rate limiting (code=-9999) automatycznie.
# To ten sam wzorzec co call_api() w S01E05/railway.py:
# - wykryj błąd rate limit
# - zresetuj stan (akcja "reset")
# - odczekaj i ponów zapytanie
#
# Kluczowe: agent NIE WIE o rate limitach — to warstwa Pythona
# je obsługuje transparentnie. Lekcja S01E05: "aplikacja produkcyjna
# musi być zaprojektowana tak, żeby zewnętrzne ograniczenia nie
# powodowały awarii, lecz były obsługiwane gracefully".

def zmail_request(action: str, **kwargs) -> dict:
    payload = {"apikey": HUB_API_KEY, "action": action, **kwargs}
    r = requests.post(ZMAIL_URL, json=payload, timeout=30)
    data = r.json()
    if data.get("code") == -9999:
        print("     [rate limit — reset + retry]")
        requests.post(ZMAIL_URL, json={"apikey": HUB_API_KEY, "action": "reset"}, timeout=10)
        time.sleep(3)
        r = requests.post(ZMAIL_URL, json=payload, timeout=30)
        data = r.json()
    time.sleep(API_DELAY)
    return data


# ============================================================
# NARZĘDZIA — implementacje akcji API zmail
# ============================================================
#
# Struktura narzędzi odzwierciedla dwuetapowy model API:
#
# ETAP 1 — DISCOVERY (metadane, bez treści):
#   get_help()      → dokumentacja API (samo-dokumentujące API, S01E05)
#   get_inbox()     → lista wątków z metadanymi
#   search_emails() → wyszukiwanie (operatory Gmail: from:, subject:...)
#   get_thread()    → lista wiadomości w wątku (bez treści!)
#
# ETAP 2 — DETAIL (pełna treść):
#   get_messages()  → pobranie treści po messageID
#
# ETAP 3 — ACTION:
#   submit_answer() → wysłanie odpowiedzi do weryfikacji
#
# To realizacja konceptu "nawigacji przez powiązania" z lekcji S02E03:
# agent nie pobiera całej skrzynki — nawiguje: search → thread → message.
# Każdy krok zawęża kontekst, oszczędzając tokeny i pieniądze.

def get_help() -> str:
    """Pobierz dokumentację API — agent poznaje możliwości dynamicznie."""
    return json.dumps(zmail_request("help"), ensure_ascii=False, indent=2)


def get_inbox(page: int = 1, perPage: int = 20) -> str:
    """Perspektywa z lotu ptaka — metadane wątków, BEZ treści wiadomości."""
    return json.dumps(zmail_request("getInbox", page=page, perPage=perPage), ensure_ascii=False, indent=2)


def search_emails(query: str, page: int = 1, perPage: int = 20) -> str:
    """Nawigacja po skrzynce — operatory Gmail (from:, to:, subject:, OR, AND).

    To kluczowe narzędzie agenta: zamiast czytać 100 maili, szuka celowo
    np. "from:proton.me" (mail Wiktora) lub "subject:SEC" (ticket).
    Lekcja S02E04: "agent nie musi widzieć całej struktury — w każdej
    chwili może przeszukać to, czego potrzebuje".
    """
    return json.dumps(zmail_request("search", query=query, page=page, perPage=perPage), ensure_ascii=False, indent=2)


def get_thread(threadID: int) -> str:
    """Powiązania — lista wiadomości w wątku (rowID, messageID, BEZ treści).

    Ważne: "czytaj CAŁE wątki — ważna informacja może być w odpowiedzi".
    Dlatego agent pobiera listę wiadomości wątku, a potem ich treść.
    """
    return json.dumps(zmail_request("getThread", threadID=threadID), ensure_ascii=False, indent=2)


def get_messages(ids: list) -> str:
    """Szczegóły — pełna treść wiadomości po messageID.

    Wskazówka z lekcji: "nie próbuj odgadywać treści na podstawie samego
    tematu — ZAWSZE pobieraj pełną wiadomość przed wyciąganiem wniosków".
    To analogia do "instruction dropout" z S02E02: model może pominąć
    szczegóły jeśli zobaczy tylko subject, a nie body.
    """
    return json.dumps(zmail_request("getMessages", ids=ids), ensure_ascii=False, indent=2)


def submit_answer(password: str, date: str, confirmation_code: str) -> str:
    """Akcja finalna — wyślij zebrane dane do weryfikacji.

    Hub zwraca precyzyjny feedback ("brak hasła", "kod nieprawidłowy")
    który agent wykorzystuje do korekty — iteracja na feedbacku (S02E03).
    """
    payload = {
        "apikey": HUB_API_KEY,
        "task": "mailbox",
        "answer": {
            "password": password,
            "date": date,
            "confirmation_code": confirmation_code,
        },
    }
    r = requests.post(VERIFY_URL, json=payload, timeout=30)
    return json.dumps(r.json(), ensure_ascii=False, indent=2)


# ============================================================
# SCHEMATY NARZĘDZI DLA FUNCTION CALLING
# ============================================================
#
# Opisy narzędzi to krytyczna część kontekstu — model podejmuje decyzje
# na ich podstawie. Kilka wzorców z wcześniejszych lekcji:
#
# - get_help: "parameters: {}" = narzędzie bez argumentów. Model musi
#   wiedzieć, że MOŻE je wywołać bez żadnych danych.
#
# - search_emails: opis zawiera "operatory Gmail: from:, to:, subject:..."
#   to HINT w kontekście — model wie JAKIM JĘZYKIEM mówić do API.
#   Bez tego mógłby próbować search("kto wysłał mail o reaktorze"),
#   zamiast search("from:proton.me"). Lekcja S01E03: "opis explicite
#   pokazuje czego narzędzie oczekuje".
#
# - get_messages: "UWAGA: rowID mogą się zmieniać — preferuj messageID"
#   to ostrzeżenie zapobiegające halucynacjom. Skrzynka jest aktywna,
#   więc rowID (pozycja w inboxie) może zmienić się między zapytaniami.
#   messageID (hash) jest stabilny.

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_help",
            "description": "Pobierz dokumentację API zmail — dostępne akcje i parametry",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_inbox",
            "description": "Pobierz listę wątków z inboxa (metadane, bez treści wiadomości)",
            "parameters": {
                "type": "object",
                "properties": {
                    "page": {"type": "integer", "description": "Numer strony (domyślnie 1)", "default": 1},
                    "perPage": {"type": "integer", "description": "Wyników na stronę 5-20 (domyślnie 20)", "default": 20},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_emails",
            "description": "Wyszukaj maile. Obsługuje operatory Gmail: from:, to:, subject:, \"phrase\", -exclude, OR, AND. Brak operatora = AND. Min. 3 znaki frazy.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Zapytanie wyszukiwania"},
                    "page": {"type": "integer", "default": 1},
                    "perPage": {"type": "integer", "default": 20},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_thread",
            "description": "Pobierz listę wiadomości (rowID, messageID) w danym wątku. Bez treści — użyj get_messages aby pobrać treść.",
            "parameters": {
                "type": "object",
                "properties": {
                    "threadID": {"type": "integer", "description": "ID wątku"},
                },
                "required": ["threadID"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_messages",
            "description": "Pobierz pełną treść wiadomości po ich identyfikatorach (rowID lub 32-znakowy messageID). UWAGA: rowID mogą się zmieniać — preferuj messageID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Lista messageID (32-znakowe hashe) lub rowID do pobrania",
                    },
                },
                "required": ["ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": "Wyślij odpowiedź do weryfikacji. Użyj gdy masz WSZYSTKIE trzy wartości.",
            "parameters": {
                "type": "object",
                "properties": {
                    "password": {"type": "string", "description": "Hasło do systemu pracowniczego"},
                    "date": {"type": "string", "description": "Data ataku w formacie YYYY-MM-DD"},
                    "confirmation_code": {"type": "string", "description": "Kod potwierdzenia SEC-... (32 znaki łącznie)"},
                },
                "required": ["password", "date", "confirmation_code"],
            },
        },
    },
]

# ============================================================
# PROMPT SYSTEMOWY — rola koordynatora (Orchestrator)
# ============================================================
#
# Lekcja S02E04 opisuje agenta zarządzającego: "posiada wiedzę o systemie,
# szerokie uprawnienia dostępu do informacji, zna role agentów i zakres
# ich odpowiedzialności". Tu jeden agent pełni obie role (koordynator
# i wykonawca), ale prompt ma cechy koordynatora:
#
# - WIEDZA O SYSTEMIE: zna kontekst fabularny (Wiktor, proton.me, ruch oporu)
# - STRATEGIA: wie W JAKIEJ KOLEJNOŚCI szukać (Wiktor → hasło → SEC)
# - DECYZYJNOŚĆ: sam decyduje kiedy ma dość danych do submit_answer
# - ITEROWANIE: "jeśli hub odrzuci — szukaj dalej" (lekcja S02E04:
#   "agent może otrzymać jedynie częściowe informacje, co może wymagać
#   dodatkowej weryfikacji")
#
# WAŻNE: prompt zawiera GOTOWĄ strategię wyszukiwania. To wybór świadomy:
# w bardziej otwartym zadaniu agent odkrywałby strategię sam (agent),
# ale tu cel jest precyzyjny (3 konkretne wartości), więc dajemy plan
# (workflow-like). Lekcja: "umiejętność odnalezienia balansu pomiędzy
# logiką agentów, prostszymi workflow i logiką w kodzie jest kluczowa".

SYSTEM_PROMPT = """Jesteś agentem przeszukującym skrzynkę mailową operatora Systemu.

CEL: Znajdź trzy informacje i wyślij je przez submit_answer:
1. date — kiedy dział bezpieczeństwa planuje atak na elektrownię (YYYY-MM-DD)
2. password — hasło do systemu pracowniczego
3. confirmation_code — kod z ticketa SEC-... (32 znaki łącznie)

KONTEKST:
- Wiktor z ruchu oporu wysłał donos z domeny proton.me
- Skrzynka jest aktywna — nowe maile mogą napływać w trakcie pracy

STRATEGIA — oszczędzaj zapytania:
1. Szukaj maila Wiktora: search_emails("from:proton.me")
2. Z wyników weź threadID → get_thread → get_messages (pełna treść)
3. Szukaj hasła: search_emails("hasło") lub search_emails("password")
4. Szukaj ticketa SEC: search_emails("subject:SEC")
5. Czytaj CAŁE wątki — ważne info może być w odpowiedzi
6. Gdy masz 3 wartości → submit_answer
7. Jeśli hub odrzuci — szukaj dalej, mogły przyjść nowe maile

WAŻNE: Zawsze pobieraj pełną treść przez get_messages zanim wyciągniesz wnioski."""


TOOL_DISPATCH = {
    "get_help": lambda args: get_help(),
    "get_inbox": lambda args: get_inbox(**args),
    "search_emails": lambda args: search_emails(**args),
    "get_thread": lambda args: get_thread(**args),
    "get_messages": lambda args: get_messages(**args),
    "submit_answer": lambda args: submit_answer(**args),
}


# ============================================================
# PĘTLA AGENTA — Orchestrator z iteracją na feedbacku
# ============================================================
#
# Ta pętla to serce architektury Orchestrator z lekcji S02E04.
# W pełnym systemie wieloagentowym agent zarządzający miałby narzędzia
# delegate/message do zlecania pracy subagentom. Tu uproszczone:
# jeden agent sam przeszukuje, ale wzorzec jest identyczny.
#
# Kluczowe cechy z lekcji:
#
# 1. DELEGOWANIE → tu zastąpione bezpośrednim wywołaniem narzędzi.
#    W rozbudowanej wersji: search_emails mógłby być osobnym agentem
#    "Mail Searcher" z własnym kontekstem i strategią.
#
# 2. TRANSPORT WIEDZY → wyniki narzędzi trafiają do historii (messages),
#    dzięki czemu agent pamięta co już znalazł. Lekcja: "w trakcie
#    sesji kluczowe informacje i postępy są dostępne w kontekście".
#
# 3. DYNAMICZNY KONTEKST → max 30 iteracji, bo skrzynka jest aktywna
#    i agent może musieć czekać na nowe maile. Więcej iteracji niż
#    w S02E03 (20), bo tu agent potrzebuje wielu kroków:
#    search → thread → messages × 3 tematy + retry.
#
# 4. WCZESNE WYJŚCIE NA FLADZE → jak w każdym zadaniu: po znalezieniu
#    {FLG:...} natychmiast kończymy — oszczędność tokenów.

def run_agent():
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Przeszukaj skrzynkę i znajdź: datę ataku, hasło, kod SEC. Wyślij submit_answer."},
    ]

    print("=== AGENT MAILBOX START ===\n")

    for iteration in range(30):
        print(f"--- Iteracja {iteration + 1} ---")

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        msg = response.choices[0].message
        messages.append(msg)

        if msg.content:
            print(f"Agent: {msg.content[:300]}")

        if not msg.tool_calls:
            print("\n=== Agent zakończył ===")
            if msg.content:
                print(msg.content)
            break

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
                try:
                    if "FLG" in result:
                        print("\n=== FLAGA ZNALEZIONA! ===")
                        print(result)
                        return
                except Exception:
                    pass

            preview = result[:500] + ("..." if len(result) > 500 else "")
            print(f"     => {preview}")

            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    print("\n=== KONIEC AGENTA ===")


if __name__ == "__main__":
    run_agent()
