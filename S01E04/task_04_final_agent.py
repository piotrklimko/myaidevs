import os
from dotenv import load_dotenv
load_dotenv()

import requests
import base64
import json

API_KEY = os.environ["HUB_API_KEY"]
OPENROUTER_KEY = f"Bearer {os.environ['OPENROUTER_API_KEY']}"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "anthropic/claude-haiku-4.5"

# ============================================================
# NARZĘDZIA — trzy rodzaje "oczu i rąk" agenta
# ============================================================
#
# Lekcja S01E04 skupia się na multimodalności: agent musi przetworzyć
# dokumentację zawierającą zarówno pliki tekstowe (.md) jak i obrazy (.png).
# Problem: model nie może "zobaczyć" adresu URL — obraz musi trafić do modelu
# jako Base64 lub publiczny URL w polu image_url wewnątrz wiadomości.
#
# Stąd potrzeba DWÓCH osobnych narzędzi do pobierania treści:
# - fetch_url  → dla tekstu (zwraca string)
# - analyze_image → dla obrazów (pobiera, konwertuje na Base64, wywołuje vision)
#
# To przykład zasady z lekcji S01E03: nie mapuj API 1:1, lecz zaprojektuj
# narzędzia pod kątem tego co model musi ZROZUMIEĆ i ZROBIĆ.
# Jedno generyczne "pobierz_plik" byłoby niejasne — model nie wiedziałby
# kiedy użyć vision, a kiedy wystarczy tekst.

tools = [
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            # Opis explicite wyklucza obrazki — zapobiega halucynacji w której
            # model użyje fetch_url do pobrania PNG i dostanie binarny śmieć.
            "description": "Pobiera zawartość tekstową spod podanego URL (np. pliki .md). Nie używaj do obrazków.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Pełny URL do pobrania"}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_image",
            # Kluczowe narzędzie multimodalne — model deleguje "patrzenie" na obraz
            # do osobnego wywołania vision. Z perspektywy agenta to jedno narzędzie,
            # ale w środku robi dwie rzeczy: pobiera obraz i wywołuje osobne
            # zapytanie do LLM z tym obrazem w kontekście.
            # Parametr "question" pozwala modelowi precyzyjnie ukierunkować analizę
            # zamiast pytać ogólnie "co to jest?" — wyższa skuteczność, mniej tokenów.
            "description": "Pobiera obrazek spod podanego URL i analizuje jego zawartość za pomocą vision. Użyj do plików .png, .jpg itp.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Pełny URL do obrazka"},
                    "question": {"type": "string", "description": "Co chcesz wiedzieć o tym obrazku?"}
                },
                "required": ["url", "question"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "submit_declaration",
            # Narzędzie finalne — jak submit_answer w S01E02.
            # Opis "tylko gdy masz gotową, kompletną deklarację" to wskazówka
            # dla modelu żeby nie wysyłał na próbę — każda próba to realne żądanie HTTP.
            "description": "Wysyła wypełnioną deklarację SPK do weryfikacji przez Hub. Używaj tylko gdy masz gotową, kompletną deklarację.",
            "parameters": {
                "type": "object",
                "properties": {
                    "declaration": {"type": "string", "description": "Pełny tekst deklaracji, sformatowany dokładnie według wzoru z dokumentacji"}
                },
                "required": ["declaration"]
            }
        }
    }
]

# ============================================================
# IMPLEMENTACJE NARZĘDZI
# ============================================================

def fetch_url(url: str) -> str:
    print(f"  [fetch_url] {url}")
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            # Limit 8000 znaków — ochrona przed przepełnieniem kontekstu.
            # Lekcja S01E02: zarządzanie kontekstem to priorytet, bo każdy krok
            # agenta wysyła całą historię do modelu.
            return r.text[:8000]
        return f"BŁĄD HTTP {r.status_code}"
    except Exception as e:
        return f"BŁĄD: {e}"

def analyze_image(url: str, question: str) -> str:
    """Pobiera obraz i wysyła go do modelu vision jako Base64.

    Dlaczego Base64, a nie URL?
    Lekcja S01E04 wyjaśnia: model nie może "zobaczyć" adresu URL — nie ma
    możliwości samodzielnego pobrania pliku. Jedyne opcje to:
    a) Base64 — kodujemy binarny plik jako string i wklejamy w payload JSON
    b) Publiczny URL — provider pobiera obraz po stronie serwera

    Tu używamy Base64, bo dokumentacja HUB-a jest dostępna pod zwykłym HTTPS,
    ale chcemy mieć pewność że model widzi aktualną wersję (nie cache).

    To jest OSOBNE zapytanie do LLM (zagnieżdżone wewnątrz narzędzia) —
    wynik vision wraca do agenta jako string i trafia do jego historii.
    """
    print(f"  [analyze_image] {url}")
    print(f"  [analyze_image] pytanie: {question}")
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return f"BŁĄD HTTP {r.status_code} przy pobieraniu obrazka"

        # Konwersja binarnego obrazu na Base64 string — format wymagany przez API
        img_b64 = base64.b64encode(r.content).decode()

        # Zapytanie vision: wiadomość zawiera JEDNOCZEŚNIE obraz i pytanie tekstowe.
        # To właśnie multimodalność — jeden model przetwarza oba typy danych razem.
        response = requests.post(
            OPENROUTER_URL,
            headers={"Authorization": OPENROUTER_KEY, "Content-Type": "application/json"},
            json={
                "model": MODEL,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                        {"type": "text", "text": question}
                    ]
                }]
            }
        )
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"BŁĄD: {e}"

def submit_declaration(declaration: str) -> str:
    print(f"  [submit_declaration] Wysyłam deklarację...")
    try:
        r = requests.post(
            "https://hub.ag3nts.org/verify",
            json={"apikey": API_KEY, "task": "sendit", "answer": {"declaration": declaration}},
            timeout=10
        )
        # Zwracamy pełną odpowiedź HUB-a — jeśli deklaracja jest błędna,
        # agent dostanie komunikat błędu i może poprawić i wysłać ponownie.
        # To dobra praktyka z lekcji S01E03: odpowiedź narzędzia powinna
        # zawierać wskazówki co zrobić dalej.
        return json.dumps(r.json(), ensure_ascii=False)
    except Exception as e:
        return f"BŁĄD: {e}"

def run_tool(name: str, args: dict) -> str:
    if name == "fetch_url":
        return fetch_url(**args)
    elif name == "analyze_image":
        return analyze_image(**args)
    elif name == "submit_declaration":
        return submit_declaration(**args)
    return f"Nieznane narzędzie: {name}"


# ============================================================
# PĘTLA AGENTA
# ============================================================
#
# Ta sama pętla co w S01E02, ale zadanie jest bardziej otwarte:
# agent sam odkrywa strukturę dokumentacji (zaczyna od index.md),
# sam decyduje które pliki przeczytać, które obrazy przeanalizować,
# i sam wypełnia formularz deklaracji na podstawie zebranych informacji.
#
# To przykład agenta vs workflow z lekcji S01E04:
# - Workflow: z góry wiemy że "czytaj plik A, potem B, wypełnij pole X wartością Y"
# - Agent: wiemy tylko CEL ("wypełnij deklarację") i OGRANICZENIA ("budżet 0 PP")
#   Resztę — kolejność kroków, co i kiedy czytać — decyduje model.
#
# Wadą jest niedeterminizm: agent może np. czytać te same pliki kilka razy
# lub pominąć istotny obrazek. Dlatego w instrukcji systemowej dajemy wskazówki
# ("czytaj dokumentację dokładnie - jest wiele plików, w tym graficzne"),
# ale nie narzucamy sztywnego scenariusza.

def run_agent():
    system_prompt = """Jesteś agentem, który musi wypełnić i wysłać deklarację transportu w Systemie Przesyłek Konduktorskich (SPK).

Działaj samodzielnie: czytaj dokumentację, analizuj pliki (w tym graficzne), wypełnij deklarację i wyślij ją.

Dokumentacja zaczyna się tutaj: https://hub.ag3nts.org/dane/doc/index.md

Dane przesyłki do nadania:
- Nadawca (identyfikator): 450202122
- Punkt nadawczy: Gdańsk
- Punkt docelowy: Żarnowiec
- Waga: 2800 kg
- Zawartość: kasety z paliwem do reaktora
- Budżet: 0 PP (przesyłka ma być darmowa lub finansowana przez System)
- Uwagi specjalne: BRAK - nie dodawaj żadnych uwag

Wskazówki:
- Czytaj dokumentację dokładnie - jest wiele plików, w tym graficzne (.png)
- Znajdź wzór deklaracji i wypełnij go precyzyjnie
- Ustal właściwą kategorię przesyłki i kod trasy
- Oblicz opłatę zgodnie z regulaminem
- Wyślij deklarację narzędziem submit_declaration
- Jeśli Hub odrzuci deklarację, przeczytaj błąd i popraw"""

    # Uwaga: instrukcja systemowa trafia jako wiadomość "user", a nie "system".
    # To dlatego, że używamy tu surowego API REST (nie SDK OpenAI) i upraszczamy
    # strukturę. W praktyce system prompt powinien być role: "system".
    messages = [{"role": "user", "content": system_prompt}]

    print("=== START AGENTA ===\n")

    for krok in range(1, 20):  # limit 20 kroków — zabezpieczenie przed pętlą nieskończoną
        print(f"\n--- Krok {krok} ---")

        response = requests.post(
            OPENROUTER_URL,
            headers={"Authorization": OPENROUTER_KEY, "Content-Type": "application/json"},
            json={
                "model": MODEL,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto"   # model sam decyduje czy i które narzędzie użyć
            }
        )

        data = response.json()
        msg = data["choices"][0]["message"]
        finish_reason = data["choices"][0]["finish_reason"]

        messages.append(msg)

        if msg.get("content"):
            print(f"Agent: {msg['content']}")

        if finish_reason == "stop" or not msg.get("tool_calls"):
            print("\n=== AGENT ZAKOŃCZYŁ PRACĘ ===")
            break

        # Wykonaj wszystkie narzędzia z tej tury (możliwy parallel tool calling)
        tool_results = []
        for tool_call in msg["tool_calls"]:
            name = tool_call["function"]["name"]
            args = json.loads(tool_call["function"]["arguments"])
            print(f"Wywołanie narzędzia: {name}({args})")

            result = run_tool(name, args)
            print(f"Wynik: {result[:300]}{'...' if len(result) > 300 else ''}")

            tool_results.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": result
            })

        # Wszystkie wyniki narzędzi trafiają do historii naraz — model dostanie
        # je w następnej iteracji i zdecyduje o kolejnym kroku
        messages.extend(tool_results)

    print("\n=== KONIEC ===")

if __name__ == "__main__":
    run_agent()
