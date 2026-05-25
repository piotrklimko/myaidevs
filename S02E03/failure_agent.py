import os
from dotenv import load_dotenv
load_dotenv()

# ============================================================
# S02E03 — Dokumenty oraz pamięć długoterminowa jako narzędzia
# ============================================================
#
# Lekcja S02E03 skupia się na dwóch kluczowych konceptach:
#
# 1. KOMPRESJA KONTEKSTU (główna technika tego zadania)
#    Plik logów ma 2137 linii (~82 000 tokenów) — nie mieści się w żadnym
#    rozsądnym oknie kontekstowym, a na pewno nie w limicie 1500 tokenów.
#    To bezpośrednia realizacja konceptu "Observational Memory" z lekcji:
#    zamiast wrzucać całość do kontekstu, KOMPRESUJEMY dane do esencji.
#    Observer/Reflector z lekcji kompresują historię rozmowy — tu kompresujemy
#    logi systemowe, ale zasada jest identyczna: zachowaj sygnał, odrzuć szum.
#
# 2. BAZA WIEDZY TWORZONA DLA AGENTA (nie podłączana)
#    Lekcja podkreśla różnicę: zamiast rzucać agentowi "przeszukaj 10k
#    dokumentów", dajemy mu NARZĘDZIA do nawigacji (get_logs_by_level,
#    search_logs). Agent nie widzi 82k tokenów naraz — widzi tylko wyniki
#    swoich zapytań. To odpowiednik "nawigacji przez powiązania" z lekcji:
#    agent eksploruje dane krok po kroku, zamiast dostać wszystko na raz.
#
# 3. ITERACJA NA PODSTAWIE FEEDBACKU
#    Centrala zwraca precyzyjny feedback ("brakuje info o komponencie X").
#    Agent poprawia logi i wysyła ponownie. To ten sam wzorzec co
#    engineer_prompt w S02E01, ale z zewnętrznym walidatorem (technicy)
#    zamiast modelu. Lekcja: "feedback od techników jest bardzo precyzyjny
#    — warto go wykorzystać do uzupełnienia wynikowego pliku".
#
# 4. SUBAGENT/NARZĘDZIE ZAMIAST MONOLITU
#    Wskazówka z lekcji: "warto mieć narzędzie do przeszukiwania logów,
#    zamiast trzymać je w całości w pamięci głównego agenta". Dlatego logi
#    są przeszukiwane przez funkcje Pythona (get_logs_by_level, search_logs),
#    a nie wrzucane do kontekstu LLM. Agent widzi WYNIKI przeszukań, nie surowe dane.

import re
import json
import requests
from openai import OpenAI

HUB_API_KEY = os.environ["HUB_API_KEY"]
HUB_URL = "https://hub.ag3nts.org/verify"
LOG_URL = f"https://hub.ag3nts.org/data/{HUB_API_KEY}/failure.log"

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = "anthropic/claude-haiku-4.5"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# ============================================================
# ŁADOWANIE LOGÓW — dane trzymane w pamięci Pythona, NIE w kontekście LLM
# ============================================================
#
# Kluczowy wzorzec z lekcji S02E03: "warto mieć narzędzie do przeszukiwania
# logów, zamiast trzymać je w całości w pamięci głównego agenta".
#
# Plik ma 2137 linii (~82 000 tokenów). Gdybyśmy wrzucili go do kontekstu
# LLM, natychmiast przekroczylibyśmy budżet tokenów (i budżet finansowy).
# Zamiast tego: logi żyją w Pythonie (LOG_LINES), a LLM sięga po nie
# przez narzędzia — dokładnie jak agent nawigujący po kodzie źródłowym
# z lekcji (grep, ls zamiast czytania całego projektu naraz).
print("Pobieranie logów...")
LOG_LINES = requests.get(LOG_URL).text.splitlines()
print(f"Załadowano {len(LOG_LINES)} linii logów.\n")


# ============================================================
# NARZĘDZIA — interfejs agenta do nawigacji po danych
# ============================================================
#
# Lekcja S02E03 opisuje 4 sposoby nawigacji po treściach:
# - perspektywa (spojrzenie z lotu ptaka) → get_logs_by_level
# - nawigacja (przeszukiwanie nazw/treści) → search_logs
# - powiązania (odnośniki między dokumentami) → ID komponentów łączą logi
# - szczegóły (czytanie oryginalnej treści) → wyniki narzędzi
#
# To analogia do agenta kodującego: zamiast czytać cały projekt, agent
# robi "grep" na logach. Każde narzędzie zwraca WYCINEK danych, nie całość.

def get_logs_by_level(level: str, max_results: int = 200) -> str:
    """Zwróć logi o podanym poziomie (CRIT, ERRO, WARN, INFO).

    To odpowiednik "perspektywy" z lekcji — agent widzi wszystkie zdarzenia
    danego poziomu ważności, ale NIE widzi reszty logów. Pozwala szybko
    ocenić skalę problemu (ile CRIT? ile ERRO?) bez czytania 2137 linii.

    max_results chroni kontekst LLM: nawet INFO ma setki wpisów, a każdy
    zbędny wiersz to tokeny (= pieniądze). Lekcja S02E03: "drogie modele
    wygenerują wysokie koszty jeśli będziesz wielokrotnie pracował na
    dużych zbiorach danych".
    """
    level = level.upper()
    tag = f"[{level}]"
    result = [l for l in LOG_LINES if tag in l]
    if len(result) > max_results:
        result = result[:max_results]
        note = f"(pokazano {max_results} z {len([l for l in LOG_LINES if tag in l])} — użyj search_logs by zawęzić)"
        return "\n".join(result) + f"\n\n{note}"
    return "\n".join(result) if result else f"Brak logów poziom {level}."


def search_logs(keyword: str, level_filter: str = "", max_results: int = 50) -> str:
    """Wyszukaj logi zawierające słowo kluczowe, opcjonalnie filtrując po poziomie.

    To odpowiednik "nawigacji" z lekcji — agent przeszukuje TREŚĆ logów
    po nazwie komponentu (ECCS8, WTANK07, WTRPMP...). Pozwala śledzić
    historię jednego podzespołu: od WARN przez ERRO do CRIT.

    level_filter dodaje drugą oś filtrowania — agent może np. zobaczyć
    TYLKO CRIT-y z danego komponentu. To precyzyjne zawężanie kontekstu
    zamiast przeszukiwania "na oślep".
    """
    keyword_lower = keyword.lower()
    results = []
    for line in LOG_LINES:
        if keyword_lower in line.lower():
            if not level_filter or f"[{level_filter.upper()}]" in line:
                results.append(line)
    if not results:
        return f"Brak wyników dla '{keyword}'."
    if len(results) > max_results:
        note = f"\n(pokazano {max_results} z {len(results)} wyników)"
        return "\n".join(results[:max_results]) + note
    return "\n".join(results)


def count_tokens(text: str) -> dict:
    """Oszacuj liczbę tokenów w tekście.

    Lekcja S02E03 explicite mówi: "zliczaj tokeny przed wysłaniem —
    wysyłanie logów przekraczających limit skończy się odrzuceniem.
    Wbuduj zliczanie tokenów jako osobny krok przed weryfikacją."

    Przelicznik chars/3.5 jest KONSERWATYWNY — lepiej odrzucić za dużo
    niż wysłać tekst przekraczający limit i zmarnować próbę. W produkcji
    użylibyśmy tiktoken (tokenizer OpenAI) lub API tokenizera providera.
    """
    chars = len(text)
    lines = text.count("\n") + 1
    estimated = int(chars / 3.5)
    return {
        "chars": chars,
        "lines": lines,
        "estimated_tokens": estimated,
        "within_limit": estimated <= 1500,
        "note": "Szacunek konserwatywny (chars/3.5). Rzeczywista liczba może być niższa."
    }


def submit_logs(logs: str) -> dict:
    """Wyślij skompresowane logi do weryfikacji przez Centralę.

    WAŻNE: funkcja sprawdza limit tokenów PRZED wysłaniem. To "fail-fast"
    z S02E01 — nie marnujemy zapytania HTTP jeśli wiemy, że odpowiedź
    będzie odrzucona. Centrala zwraca precyzyjny feedback ("brakuje info
    o komponencie X"), który agent wykorzystuje w kolejnej iteracji.

    To realizacja konceptu "iteracja na podstawie feedbacku" z lekcji:
    agent nie zgaduje idealnych logów za pierwszym razem — wysyła wersję,
    dostaje informację zwrotną, poprawia i wysyła ponownie.
    """
    token_check = count_tokens(logs)
    if not token_check["within_limit"]:
        return {
            "error": f"Logi przekraczają limit! Szacowane tokeny: {token_check['estimated_tokens']}/1500. Skróć jeszcze.",
            "token_info": token_check
        }

    r = requests.post(HUB_URL, json={
        "apikey": HUB_API_KEY,
        "task": "failure",
        "answer": {"logs": logs}
    })
    result = r.json()
    result["token_info"] = token_check
    return result


# ============================================================
# SCHEMATY NARZĘDZI DLA OPENAI API (Function Calling)
# ============================================================
#
# Każde narzędzie ma precyzyjny opis i enumy — to realizacja zasady
# z lekcji S01E03: "nie mapuj API 1:1, lecz zaprojektuj narzędzia pod kątem
# tego co model musi ZROZUMIEĆ i ZROBIĆ". Opis search_logs zawiera
# przykładowe ID komponentów (ECCS8, WTANK07...) — to "hint" dla modelu,
# żeby wiedział CZEGO szukać bez czytania całego pliku.
#
# Kluczowe: opisy narzędzi to część KONTEKSTU modelu. Dobrze napisany
# opis = mniej błędnych wywołań = mniej zmarnowanych tokenów.
# Lekcja S02E03: "prezentowanie zewnętrznych zasobów dla modelu" —
# enum poziomu [CRIT, ERRO, WARN, INFO] mówi modelowi jakie kategorie
# istnieją, zamiast kazać mu zgadywać.

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_logs_by_level",
            "description": "Pobierz logi o podanym poziomie ważności: CRIT (krytyczne), ERRO (błędy), WARN (ostrzeżenia), INFO (informacyjne). Zwraca max max_results linii.",
            "parameters": {
                "type": "object",
                "properties": {
                    "level": {
                        "type": "string",
                        "enum": ["CRIT", "ERRO", "WARN", "INFO"],
                        "description": "Poziom ważności logów"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maksymalna liczba wyników (domyślnie 200)",
                        "default": 200
                    }
                },
                "required": ["level"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_logs",
            "description": "Wyszukaj logi zawierające podane słowo kluczowe (np. nazwę komponentu: ECCS8, WTANK07, PWR01, WTRPMP, FIRMWARE, STMTURB12, WSTPOOL2). Opcjonalnie filtruj po poziomie.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "Słowo kluczowe do wyszukania (component ID, fragment opisu)"
                    },
                    "level_filter": {
                        "type": "string",
                        "enum": ["", "CRIT", "ERRO", "WARN", "INFO"],
                        "description": "Opcjonalny filtr poziomu",
                        "default": ""
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maks liczba wyników",
                        "default": 50
                    }
                },
                "required": ["keyword"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "count_tokens",
            "description": "Oszacuj liczbę tokenów w przygotowanym tekście logów przed wysłaniem.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Tekst do zliczenia"
                    }
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "submit_logs",
            "description": "Wyślij skompresowane logi do weryfikacji przez Centralę. Zwraca feedback lub flagę jeśli rozwiązanie jest poprawne. ZAWSZE sprawdź count_tokens przed wysłaniem.",
            "parameters": {
                "type": "object",
                "properties": {
                    "logs": {
                        "type": "string",
                        "description": "Skompresowany string logów, wiersze oddzielone \\n"
                    }
                },
                "required": ["logs"]
            }
        }
    }
]

TOOL_FUNCTIONS = {
    "get_logs_by_level": get_logs_by_level,
    "search_logs": search_logs,
    "count_tokens": count_tokens,
    "submit_logs": submit_logs,
}

# ============================================================
# PROMPT SYSTEMOWY — instrukcja kompresji dla agenta
# ============================================================
#
# Prompt realizuje kilka konceptów z lekcji S02E03:
#
# 1. STRATEGIA KOMPRESJI = Observer z Observational Memory
#    Tak jak Observer z lekcji kompresuje rozmowę do "precyzyjnych
#    i zwięzłych wpisów stanowiących esencję interakcji", tak tu agent
#    kompresuje 2137 linii logów do ~36 unikalnych zdarzeń. Zasada
#    jest ta sama: zachowaj timestamp, severity, ID komponentu — odrzuć
#    powtórzenia i szum informacyjny (INFO).
#
# 2. DEDUPLIKACJA = kluczowa technika kompresji
#    Logi powtarzają te same zdarzenia wielokrotnie (np. "ECCS8 runaway
#    outlet temperature" pojawia się co kilka godzin). Strategia: zachowaj
#    PIERWSZE i OSTATNIE wystąpienie — to wystarczy do odtworzenia
#    chronologii awarii. To analogia do Reflectora z lekcji, który
#    kompresuje dziennik gdy przekracza 60k tokenów.
#
# 3. LISTA KOMPONENTÓW W PROMPCIE
#    Podajemy agentowi nazwy komponentów (ECCS8, WTANK07...) explicite.
#    To "mapa obszarów bazy wiedzy" z lekcji — agent wie CZEGO szukać,
#    nie musi odkrywać struktury danych samodzielnie.

SYSTEM_PROMPT = """Jesteś agentem analizującym logi systemowe awarii elektrowni atomowej.

ZADANIE:
Przygotuj skondensowane logi (maks. 1500 tokenów), które pozwolą technikom przeanalizować przyczynę awarii.

WYMAGANIA:
- Uwzględnij zdarzenia dotyczące: zasilania, chłodzenia, pomp wodnych, oprogramowania i innych podzespołów elektrowni
- Format: jeden wpis na linię, zachowaj: datę (YYYY-MM-DD), godzinę (HH:MM), poziom ([CRIT]/[ERRO]/[WARN]), ID komponentu i opis
- LIMIT: 1500 tokenów — ZAWSZE sprawdź count_tokens przed submit_logs
- Możesz skracać opisy zdarzeń

STRATEGIA:
1. Pobierz logi CRIT i ERRO (najważniejsze)
2. Zidentyfikuj kluczowe komponenty i zdarzenia prowadzące do awarii
3. Usuń duplikaty (te same zdarzenia powtarzające się wielokrotnie — zachowaj tylko pierwsze i/lub ostatnie)
4. Skróć opisy jeśli za długie
5. Sprawdź count_tokens → wyślij → jeśli feedback wskazuje brakujące info, uzupełnij

KOMPONENTY: ECCS8 (chłodzenie rdzenia), WTANK07 (zbiornik wody chłodzącej), WTRPMP (pompa wody),
WSTPOOL2 (basen odpadów/rozpraszanie ciepła), PWR01 (zasilanie), FIRMWARE (oprogramowanie), STMTURB12 (turbina)"""


# ============================================================
# PĘTLA AGENTA — iteracyjna kompresja z feedbackiem
# ============================================================
#
# Ta pętla realizuje wzorzec "deep action" z lekcji S02E03:
# agent nie generuje wyniku za jednym zamachem, lecz ITERUJE:
#   1. Eksploracja — pobiera logi CRIT/ERRO (get_logs_by_level)
#   2. Zawężanie — szuka po komponentach (search_logs)
#   3. Kompresja — buduje skondensowany tekst
#   4. Walidacja — sprawdza count_tokens
#   5. Wysłanie — submit_logs → feedback od techników
#   6. Korekta — na podstawie feedbacku uzupełnia brakujące dane
#
# To jest AGENT, nie workflow: kolejność kroków nie jest z góry określona.
# Model sam decyduje, które narzędzie wywołać na podstawie aktualnego
# stanu wiedzy i feedbacku. To kluczowa różnica opisana w lekcji:
# "workflow: z góry wiemy co robić" vs "agent: wiemy tylko CEL".
#
# max_iterations=20 to zabezpieczenie (jak w każdym agentowym loopie).
# W praktyce agent potrzebuje ~5-8 iteracji: 2-3 na eksplorację,
# 1 na kompresję, 1-3 na iteracje feedback → korekta.

def run_agent():
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({
        "role": "user",
        "content": "Przeanalizuj logi awarii i wyślij skondensowaną wersję do Centrali. Zacznij od pobrania logów CRIT i ERRO."
    })

    iteration = 0
    max_iterations = 20

    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- Iteracja {iteration} ---")

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=4096,
        )

        msg = response.choices[0].message
        messages.append(msg)

        if msg.content:
            print(f"Agent: {msg.content[:300]}{'...' if len(msg.content or '') > 300 else ''}")

        if not msg.tool_calls:
            print("\nAgent zakończył bez wywołania narzędzi.")
            break

        # Wykonaj tool calls — agent może wywołać kilka narzędzi na raz
        # (parallel tool calling). Np. get_logs_by_level("CRIT") i
        # get_logs_by_level("ERRO") jednocześnie = mniej iteracji pętli.
        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)

            print(f"\n[TOOL] {fn_name}({', '.join(f'{k}={repr(v)[:40]}' for k,v in fn_args.items())})")

            result = TOOL_FUNCTIONS[fn_name](**fn_args)

            result_str = json.dumps(result, ensure_ascii=False) if isinstance(result, dict) else str(result)
            print(f"       -> {result_str[:200]}{'...' if len(result_str) > 200 else ''}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result_str
            })

            # Sprawdź czy dostaliśmy flagę — sukces = koniec pętli
            if isinstance(result, dict) and "FLG" in str(result.get("message", "")):
                print(f"\n{'='*50}")
                print(f"FLAGA: {result['message']}")
                print(f"{'='*50}")
                return result

    print("\nAgent zakończył pętlę bez flagi.")
    return None


if __name__ == "__main__":
    run_agent()
