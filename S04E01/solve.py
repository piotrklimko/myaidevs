import os
from dotenv import load_dotenv
load_dotenv()

"""
S04E01 – okoeditor – Agent rozwiązujący zadanie OKO
=====================================================

KONTEKST LEKCJI (S04E01 — Wdrożenia rozwiązań AI):
Lekcja omawia praktyczne wdrożenia AI na przykładzie "Cyfrowego Ogrodu" (Digital Garden /
Second Brain). Kluczowe koncepcje to:

1. OCZEKIWANIA vs RZECZYWISTOŚĆ — AI nie rozwiązuje wszystkiego automatycznie.
   Agent ma ograniczenia (kontekst, koszty, czas), więc trzeba precyzyjnie wybrać
   gdzie AI wnosi wartość, a gdzie nie.

2. SYNCHRONICZNA vs ASYNCHRONICZNA współpraca z AI:
   - Synchroniczna: użytkownik steruje agentem w czasie rzeczywistym (jak ten skrypt)
   - Asynchroniczna: agent działa w tle wg zdefiniowanych procesów
   Tutaj stosujemy podejście synchroniczne — agent reaguje na bieżące dane.

3. MAPOWANIE PROCESÓW — przed wdrożeniem AI trzeba zrozumieć:
   - co chcemy zrobić (cele)
   - czego NIE chcemy robić (ograniczenia)
   - jak to zrobić (architektura)
   Lekcja podkreśla, że elementy klasycznego kodu (scraping, API, logika) stanowią
   większość architektury — AI to często 20-30% rozwiązania.

4. WERYFIKACJA ZAŁOŻEŃ PRZEZ PROSTE TESTY — zamiast budować wielki system,
   lepiej szybko przetestować kluczowe założenia. Ten skrypt jest właśnie takim
   szybkim testem: agent + narzędzia + pętla decyzyjna.

ARCHITEKTURA TEGO ROZWIĄZANIA:
┌──────────────────────────────────────────────────────┐
│                    AGENT LOOP                         │
│                                                       │
│  1. Zbierz kontekst (scraping panelu OKO)            │
│  2. Przekaż kontekst + zadania → LLM                 │
│  3. LLM decyduje jakie narzędzie wywołać             │
│  4. Wykonaj narzędzie (API call)                     │
│  5. Wynik → z powrotem do LLM                        │
│  6. Powtarzaj aż LLM uzna, że skończył              │
└──────────────────────────────────────────────────────┘

Podział odpowiedzialności (lekcja: "balans kod vs AI"):
- KOD (programistyczne): logowanie do panelu, scraping HTML, parsowanie danych,
  wywoływanie API, obsługa sesji, pętla agenta
- AI (model językowy): decyzje co zmodyfikować, generowanie treści opisów,
  wybór strategii (np. który incydent podmienić na Komarowo)

WYCIĄGNIĘTE WNIOSKI (z 1. nieudanej próby):
- LLM widząc pełny HTML strony wklejał go jako "content" do API — śmieci.
  LEKCJA: dane wejściowe dla LLM muszą być oczyszczone i ustrukturyzowane.
  To nawiązanie do lekcji S02E03 o kompresji logów — im czystszy kontekst,
  tym lepsze decyzje modelu.
- Opis narzędzia (tool description) musi jasno mówić czego NIE wysyłać.
  LEKCJA: prompt engineering dotyczy też opisów narzędzi, nie tylko system prompta.
"""

import json
import re
import requests
from openai import OpenAI
from bs4 import BeautifulSoup

# ── Konfiguracja ────────────────────────────────────────────────────────
# Klucze i URL-e — w projekcie edukacyjnym hardkodowane.
# W produkcji: zmienne środowiskowe lub secret manager.
HUB_API_KEY = os.environ["HUB_API_KEY"]
HUB_URL = "https://hub.ag3nts.org/verify"
OKO_URL = "https://oko.ag3nts.org"
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

# Klient OpenAI wskazujący na OpenRouter — pozwala używać różnych modeli
# przez jednolite API (kompatybilne z OpenAI SDK).
llm = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
MODEL = "anthropic/claude-haiku-4.5"


# ══════════════════════════════════════════════════════════════════════════
# WARSTWA 1: SCRAPING — zbieranie kontekstu dla agenta
# ══════════════════════════════════════════════════════════════════════════
# Lekcja podkreśla, że agent potrzebuje AKTUALNYCH DANYCH ze świata zewnętrznego.
# W koncepcji Digital Garden agent ma dostęp do plików markdown.
# Tutaj odpowiednikiem jest panel webowy OKO — agent musi "zobaczyć" co tam jest,
# żeby wiedzieć co zmienić. To klasyczny wzorzec: OBSERVE → DECIDE → ACT.
#
# Dlaczego scraping a nie bezpośrednie API?
# Bo panel OKO nie ma endpointu do odczytu danych — ma tylko /verify do modyfikacji.
# To realistyczny scenariusz: często trzeba łączyć różne źródła danych.

class OKOPanel:
    """Scraper panelu OKO — odpowiada za zbieranie kontekstu dla agenta.

    Używa requests.Session() do utrzymania ciasteczka sesji po logowaniu.
    To ważne — bez sesji każdy request zwraca stronę logowania.
    """

    def __init__(self):
        self.session = requests.Session()
        self._login()

    def _login(self):
        """Logowanie do panelu OKO formularzem HTML.

        Panel wymaga 4 pól: action (hidden), login, password, access_key.
        Po zalogowaniu serwer ustawia ciasteczko oko_session,
        które requests.Session() automatycznie wysyła w kolejnych requestach.
        """
        self.session.post(f"{OKO_URL}/", data={
            "action": "login",
            "login": "Zofia",
            "password": "Zofia2026!",
            "access_key": HUB_API_KEY,
        })
        print("[OKO] Zalogowano do panelu")

    def _extract_detail(self, html: str) -> dict:
        """Wyciąga USTRUKTURYZOWANE dane ze strony szczegółowej elementu.

        KLUCZOWA LEKCJA: Oczyszczanie danych wejściowych dla LLM.
        ──────────────────────────────────────────────────────────
        Pierwsza wersja skryptu podawała LLM surowy tekst strony (z nawigacją,
        nagłówkami, stopką). LLM nie odróżniał "treści wpisu" od "elementów UI"
        i wklejał całą stronę jako content do API → dane były odrzucane.

        Rozwiązanie: parsujemy HTML i wyciągamy TYLKO semantyczne elementy:
        - <h2>  → tytuł wpisu
        - <p> (dłuższe niż 20 znaków) → akapity treści
        - <span> z "wykonane"/"niewykonane" → status zadania
        - element z klasą badge/meta → metadane

        To nawiązuje do lekcji S02E03 (kompresja logów): czysty, minimalny
        kontekst → lepsze decyzje modelu → mniej tokenów → niższe koszty.
        """
        soup = BeautifulSoup(html, "html.parser")

        title_el = soup.find("h2")
        title = title_el.get_text(strip=True) if title_el else ""

        badge_el = soup.find(class_=re.compile(r"badge|meta|tag|chip"))
        badge = badge_el.get_text(strip=True) if badge_el else ""

        # Filtrujemy krótkie <p> — to zazwyczaj etykiety UI, nie treść.
        paragraphs = []
        for p in soup.find_all("p"):
            text = p.get_text(strip=True)
            if text and len(text) > 20:
                paragraphs.append(text)

        status = ""
        for span in soup.find_all("span"):
            t = span.get_text(strip=True)
            if t in ("wykonane", "niewykonane"):
                status = t

        return {
            "title": title,
            "badge": badge,
            "content": "\n\n".join(paragraphs),
            "status": status,
        }

    def _get_links(self, html: str, section: str) -> list[dict]:
        """Zbiera linki do elementów z danej sekcji (incydenty, zadania, notatki).

        Każdy link ma postać /{sekcja}/{32-znakowy-hex-id}.
        ID jest kluczowy — to jedyny sposób identyfikacji elementów w API.
        """
        soup = BeautifulSoup(html, "html.parser")
        items = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith(f"/{section}/") and len(href) > len(f"/{section}/"):
                item_id = href.split("/")[-1]
                items.append({"id": item_id, "text": a.get_text(strip=True)[:120]})
        return items

    def get_section(self, section: str) -> list[dict]:
        """Pobiera pełne dane sekcji: lista elementów + szczegóły każdego.

        Wzorzec: LIST → DETAIL dla każdego elementu.
        Najpierw pobieramy stronę listową (np. /incydenty), wyciągamy linki,
        potem wchodzimy w każdy element i parsujemy szczegóły.

        Wynik to lista słowników — czysta, ustrukturyzowana forma danych,
        gotowa do serializacji jako JSON i wstrzyknięcia do kontekstu LLM.
        """
        resp = self.session.get(f"{OKO_URL}/{section}")
        links = self._get_links(resp.text, section)
        results = []
        for item in links:
            detail_resp = self.session.get(f"{OKO_URL}/{section}/{item['id']}")
            detail = self._extract_detail(detail_resp.text)
            results.append({
                "id": item["id"],
                "title": detail["title"],
                "badge": detail["badge"],
                "content": detail["content"],
                "status": detail["status"],
            })
        return results


# ══════════════════════════════════════════════════════════════════════════
# WARSTWA 2: API HUB — wykonywanie akcji w świecie zewnętrznym
# ══════════════════════════════════════════════════════════════════════════
# To "ręce" agenta — może modyfikować dane w systemie OKO.
# API ma 3 akcje: help (dokumentacja), update (modyfikacja), done (weryfikacja).
# Każda akcja wymaga: apikey, task="okoeditor", answer={...}.

def call_hub(answer: dict) -> dict:
    """Wywołuje API HUB z podaną akcją.

    Wzorzec komunikacji z API:
    - Wysyłamy JSON POST z apikey, task i answer
    - Odpowiedź zawiera code (status) i message (opis/flaga)
    - code=110 → sukces update, code=0 → sukces done (flaga!),
      code<0 → błąd (np. -710 "Condition 2 not met")
    """
    payload = {"apikey": HUB_API_KEY, "task": "okoeditor", "answer": answer}
    print(f"[API] → {json.dumps(answer, ensure_ascii=False)}")
    resp = requests.post(HUB_URL, json=payload)
    data = resp.json()
    print(f"[API] ← code={data.get('code')} msg={data.get('message')}")
    return data


# ══════════════════════════════════════════════════════════════════════════
# WARSTWA 3: DEFINICJE NARZĘDZI (TOOLS) — interfejs LLM ↔ świat
# ══════════════════════════════════════════════════════════════════════════
# Lekcja mówi o "Code Mode" i koncepcji Skills w kontekście agentów.
# Tutaj implementujemy prostszy wariant: Function Calling (tool use).
#
# Kluczowa zasada z lekcji: agent powinien mieć DEDYKOWANE narzędzia,
# a nie nieograniczony dostęp. Dajemy mu 3 narzędzia:
# - oko_update: modyfikacja danych (ograniczona do 3 sekcji)
# - oko_done: weryfikacja zakończenia
# - oko_refresh: odświeżenie kontekstu (ponowne scraping)
#
# WAŻNA LEKCJA O OPISACH NARZĘDZI:
# ─────────────────────────────────
# Opis narzędzia (description) to prompt engineering tak samo jak system prompt!
# W pierwszej wersji opis oko_update był zbyt ogólny — LLM wklejał śmieci.
# Po dodaniu "WAŻNE: 'content' to TYLKO treść opisu, BEZ nagłówków strony"
# problem zniknął. To samo dotyczy Skills w Digital Garden z lekcji.

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "oko_update",
            # Opis jest celowo szczegółowy — mówi LLM czego NIE robić.
            # To kluczowa technika: negatywne instrukcje w opisie narzędzia
            # zapobiegają typowym błędom modelu.
            "description": (
                "Aktualizuje element w panelu OKO. "
                "WAŻNE: 'content' to TYLKO treść opisu (kilka zdań/akapitów), BEZ nagłówków strony, nawigacji, dat. "
                "'title' to tytuł wpisu (np. 'MOVE04 Opis zdarzenia'). "
                "Dla zadań: 'done' ustawia status."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "page": {"type": "string", "enum": ["incydenty", "notatki", "zadania"]},
                    "id": {"type": "string", "description": "32-znakowy hex ID elementu"},
                    "title": {"type": "string", "description": "Nowy tytuł wpisu"},
                    "content": {"type": "string", "description": "Nowa treść opisu (sam tekst, bez nawigacji/metadanych)"},
                    "done": {"type": "string", "enum": ["YES", "NO"], "description": "Status wykonania (tylko zadania)"},
                },
                "required": ["page", "id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "oko_done",
            "description": "Weryfikuje czy wszystkie wymagane zmiany zostały wprowadzone. Wywołaj po wykonaniu WSZYSTKICH 3 zadań.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "oko_refresh",
            # Narzędzie do ponownego zbierania kontekstu — agent może sam
            # zdecydować, że potrzebuje odświeżonych danych (np. po update).
            # To element SAMODZIELNOŚCI agenta z lekcji.
            "description": "Odświeża dane z panelu OKO dla wybranej sekcji.",
            "parameters": {
                "type": "object",
                "properties": {
                    "section": {"type": "string", "enum": ["incydenty", "zadania", "notatki"]},
                },
                "required": ["section"],
            },
        },
    },
]


def execute_tool(name: str, args: dict, panel: OKOPanel) -> str:
    """Dispatcher narzędzi — tłumaczy decyzje LLM na akcje w kodzie.

    To WARSTWA POŚREDNIA między LLM a światem zewnętrznym.
    Lekcja mówi o tym, że agent powinien działać w SANDBOXIE —
    tutaj sandbox to ograniczony zestaw narzędzi i walidacja argumentów.

    Każde narzędzie zwraca JSON string — LLM dostaje wynik i na jego
    podstawie decyduje co robić dalej (kolejne narzędzie lub zakończenie).
    """
    if name == "oko_update":
        answer = {"action": "update", "page": args["page"], "id": args["id"]}
        for field in ("title", "content", "done"):
            if field in args:
                answer[field] = args[field]
        result = call_hub(answer)
        return json.dumps(result, ensure_ascii=False)

    elif name == "oko_done":
        result = call_hub({"action": "done"})
        return json.dumps(result, ensure_ascii=False)

    elif name == "oko_refresh":
        data = panel.get_section(args["section"])
        return json.dumps(data, ensure_ascii=False, indent=2)

    return '{"error": "unknown tool"}'


# ══════════════════════════════════════════════════════════════════════════
# WARSTWA 4: PĘTLA AGENTA — serce całego rozwiązania
# ══════════════════════════════════════════════════════════════════════════
# Klasyczny wzorzec AGENT LOOP:
#
#   while not done:
#       response = llm(messages + tools)
#       if response has tool_calls:
#           for each tool_call:
#               result = execute_tool(tool_call)
#               messages.append(result)
#       else:
#           break  # LLM uznał, że skończył
#
# Lekcja opisuje to jako "logikę agenta" — model sam decyduje:
# - JAKIE narzędzie wywołać (oko_update, oko_done, oko_refresh)
# - Z JAKIMI argumentami (które ID, jaka treść)
# - KIEDY zakończyć (po wywołaniu oko_done z sukcesem)
#
# SYSTEM PROMPT to odpowiednik "procesów opisanych w plikach" z lekcji —
# definiuje ZASADY i CELE, a agent sam wybiera strategię realizacji.

def run_agent():
    panel = OKOPanel()

    # ── Krok 1: Zbierz pełny kontekst ──────────────────────────────────
    # Lekcja: agent musi mieć dostęp do AKTUALNYCH danych.
    # Scrapujemy wszystkie 3 sekcje panelu OKO, żeby LLM widział:
    # - jakie incydenty istnieją (tytuły z kodami MOVE/PROB/RECO, treści, ID)
    # - jakie zadania czekają (statusy, powiązania z miastami)
    # - jakie notatki są dostępne (kody klasyfikacji — kluczowa wiedza domenowa)
    print("\n[AGENT] Pobieram stan panelu OKO...")
    incydenty = panel.get_section("incydenty")
    zadania = panel.get_section("zadania")
    notatki = panel.get_section("notatki")

    # Formatujemy dane jako czytelny JSON — LLM dobrze radzi sobie z JSON-em.
    # Alternatywa: markdown, tabele, YAML — ale JSON jest jednoznaczny
    # i łatwo z niego wyciągnąć ID i pola do modyfikacji.
    state_text = f"""## INCYDENTY (6 wpisów)
{json.dumps(incydenty, ensure_ascii=False, indent=2)}

## ZADANIA (6 wpisów)
{json.dumps(zadania, ensure_ascii=False, indent=2)}

## NOTATKI (5 wpisów)
{json.dumps(notatki, ensure_ascii=False, indent=2)}"""

    # ── Krok 2: System prompt — zasady i cele agenta ───────────────────
    # Lekcja: procesy mogą być "opisane w plikach" (jak Skills / Workflows).
    # Tutaj system prompt pełni tę rolę — definiuje:
    # - JAKIE zadania wykonać (4 konkretne kroki)
    # - JAKIE zasady stosować (co to jest "content", czego nie ruszać)
    # - JAKĄ strategię przyjąć (modyfikuj TYLKO 3 elementy)
    #
    # KRYTYCZNE ZASADY — wynikają z pierwszej nieudanej próby:
    # - "content to TYLKO sam opis" — bez tego LLM wklejał nawigację HTML
    # - "NIE modyfikuj notatek" — bez tego LLM zaczął "porządkować" system
    # - "Modyfikuj TYLKO 3 elementy" — ogranicza blast radius błędów
    system_prompt = """Jesteś agentem modyfikującym dane w systemie OKO przez API.

ZADANIA:
1. Zmień klasyfikację incydentu o Skolwinie: MOVE03 → MOVE04 (zwierzęta).
   Zmień title (podmień MOVE03 na MOVE04) i content (przepisz treść tak, aby dotyczyła zwierząt/bobrów zamiast ludzi/pojazdów).

2. Zadanie "Zbadanie nagrań z okolic Skolwina": oznacz done=YES, w content wpisz że widziano bobry.

3. Dodaj incydent o Komarowie: wybierz jeden z mniej istotnych incydentów i zmień jego title na "MOVE01 Wykrycie ruchu ludzi w okolicach miasta Komarowo", a content na opis wykrycia obecności ludzi.

4. Po wykonaniu 1-3 wywołaj oko_done.

KRYTYCZNE ZASADY:
- 'content' to TYLKO sam opis tekstowy (kilka akapitów). NIE wstawiaj tam nagłówków strony, nazw sekcji, dat, nawigacji.
- 'title' to TYLKO tytuł wpisu (np. "MOVE04 Opis zdarzenia").
- NIE modyfikuj notatek ani żadnych elementów poza wymienionymi.
- Modyfikuj TYLKO 3 elementy: incydent Skolwin, zadanie Skolwin, jeden inny incydent (na Komarowo).
- Wykonuj narzędzia po jednym na krok."""

    # ── Krok 3: Inicjalizacja historii konwersacji ─────────────────────
    # Historia (messages) to PAMIĘĆ agenta w ramach jednej sesji.
    # Lekcja mówi o Digital Garden jako "pamięci agenta" — tutaj uproszczenie:
    # pamięć = lista wiadomości w kontekście LLM.
    # System prompt + user message z danymi = pełny kontekst startowy.
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Stan systemu OKO:\n\n{state_text}"},
    ]

    # ── Krok 4: Pętla agenta ──────────────────────────────────────────
    # Limit 20 kroków to ZABEZPIECZENIE — bez niego agent mógłby
    # wpaść w nieskończoną pętlę (np. retry po błędzie).
    # Lekcja: "samodzielność" agenta musi mieć granice.
    for step in range(20):
        print(f"\n[AGENT] === Krok {step + 1} ===")

        # Wywołanie LLM z narzędziami (Function Calling / Tool Use)
        # temperature=0 → deterministyczne odpowiedzi, mniej "kreatywności"
        # w wyborze narzędzi. Dla agentów wykonawczych chcemy przewidywalności.
        response = llm.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            temperature=0,
        )
        msg = response.choices[0].message

        # Dodajemy odpowiedź asystenta do historii — LLM widzi swoje
        # poprzednie decyzje i wyniki narzędzi, co pozwala mu planować
        # kolejne kroki (np. "zrobiłem 1 i 2, teraz 3").
        messages.append(msg)

        if msg.content:
            print(f"[LLM] {msg.content[:200]}")

        # Jeśli LLM nie wywołał żadnego narzędzia — uznał, że skończył.
        if not msg.tool_calls:
            print("[AGENT] Brak tool calls — koniec.")
            break

        # ── Wykonanie narzędzi ─────────────────────────────────────────
        # LLM może wywołać jedno lub więcej narzędzi w jednym kroku.
        # Każdy wynik wracamy jako message z role="tool" i tool_call_id.
        # To standard OpenAI Function Calling — LLM wie który wynik
        # odpowiada któremu wywołaniu.
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments)
            print(f"[TOOL] {fn_name}({json.dumps(fn_args, ensure_ascii=False)[:200]})")

            result = execute_tool(fn_name, fn_args, panel)
            print(f"[RESULT] {result[:300]}")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

            # Wczesne zakończenie — jeśli oko_done zwróciło sukces,
            # nie ma sensu kontynuować pętli.
            if fn_name == "oko_done" and '"code": 0' in result:
                print("\n[AGENT] SUKCES! Flaga otrzymana.")
                return

        # Dodatkowy warunek stopu — jeśli LLM zakończył bez narzędzi.
        if response.choices[0].finish_reason == "stop" and not msg.tool_calls:
            break

    print("\n[AGENT] Zakończono.")


if __name__ == "__main__":
    run_agent()
