import os
from dotenv import load_dotenv
load_dotenv()

"""
S03E05 — Podejście NIEDETERMINISTYCZNE (agent LLM)
====================================================

Kontekst lekcji:
    Lekcja S03E05 opisuje agentów, którzy nie podążają za sztywnym skryptem,
    lecz SAMODZIELNIE eksplorują przestrzeń problemu. Cytat z lekcji:

        "Żadna z podjętych decyzji oraz zachowań nie zostały wprost zdefiniowane
         w logice aplikacji ani promptach. To model 'decyduje' kiedy kontekst jest
         wystarczający, a kiedy trzeba sięgnąć do pamięci."

    Ten skrypt implementuje właśnie takie podejście. LLM dostaje:
    - CEL: "dotrzeć do Skolwina"
    - NARZĘDZIA: toolsearch, api_query, submit_answer
    - OGÓLNE REGUŁY: format mapy, kierunki, zasoby

    ...ale NIE dostaje:
    - gotowej mapy (musi ją sam pobrać)
    - listy pojazdów (musi je odkryć)
    - reguł terenu (musi je znaleźć w "books")
    - algorytmu pathfindingu (musi sam rozumować nad trasą)

Kluczowe różnice vs solve.py (deterministyczne):
    1. ODKRYWANIE vs KODOWANIE:
       - solve.py: programista z góry wie, że istnieją endpointy maps/wehicles/books
       - agent.py: LLM sam odkrywa narzędzia przez toolsearch
       To odpowiada koncepcji "proaktywności" z lekcji — model sam określa
       "kiedy podjąć działanie oraz jakiej wiedzy potrzebuje".

    2. WNIOSKOWANIE vs ALGORYTM:
       - solve.py: Dijkstra gwarantuje optimum, ale wymaga zakodowania reguł
       - agent.py: LLM rozumuje matematycznie ("potrzebuję min. 8 ruchów rakietą"),
         może się mylić, ale potrafi SKORYGOWAĆ błąd na podstawie feedbacku
         (np. "Food reached zero" → zmiana strategii).
       To jest ta "niedeterministyczna przewaga" — każde uruchomienie może
       pójść inną ścieżką rozumowania i odkryć inne rozwiązanie.

    3. ODPORNOŚĆ NA ZMIANĘ:
       - solve.py: zmiana reguł = zmiana kodu
       - agent.py: zmiana reguł = LLM przeczyta nowe dane z API i dostosuje się
       To odpowiada fragmentowi lekcji o "stwarzaniu warunków" zamiast
       "spełniania założeń".

    4. RYZYKO BŁĘDU:
       - solve.py: jeśli reguły zakodowane poprawnie → gwarancja sukcesu
       - agent.py: LLM może popełnić błąd (i popełnił — wybrał konia,
         zabrakło jedzenia), ale mechanizm pętli agenckiej pozwala
         na korektę. To "trial and error" sterowane rozumowaniem.

Architektura kognitywna (z lekcji):
    Lekcja odwołuje się do "Cognitive Architectures for Language Agents" i opisuje
    warstwy agenta. W tym skrypcie realizujemy je tak:

    - Tożsamość: system prompt definiuje rolę ("agent planujący trasę")
    - Zdolności poznawcze: narzędzia (toolsearch, api_query) pozwalają
      agentowi na "zadawanie pytań" światu zewnętrznemu
    - Synteza: LLM łączy dane z różnych źródeł (mapa + pojazdy + reguły terenu)
      bez explicite zakodowanych reguł łączenia
    - Mechaniki wzmacniające: instrukcja "Think carefully about resource management"
      zachęca do głębszego rozumowania, ale NIE mówi JAK to robić
"""

import requests
import json
from openai import OpenAI

API_KEY = os.environ["HUB_API_KEY"]
HUB = "https://hub.ag3nts.org"
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# ============================================================================
# DEFINICJA NARZĘDZI (Function Calling)
# ============================================================================
# To jest "przestrzeń", po której agent się porusza (cytat z lekcji:
# "agent cały czas porusza się po wyznaczonej przez nas przestrzeni").
#
# Zauważ, że narzędzia są GENERYCZNE — nie mówią agentowi CO ma wyszukać,
# a jedynie DAJĄ MU MOŻLIWOŚĆ wyszukiwania. To kluczowa różnica między
# "podążaniem za instrukcjami" a "stwarzaniem warunków".
#
# W solve.py odpowiednikiem jest hardkodowane api_call() z konkretnymi
# parametrami — tam programista decyduje, co odpytać.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "toolsearch",
            "description": "Search for available API tools. Returns matching tools with their endpoints. Use natural language or keywords.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query to find tools"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "api_query",
            "description": "Query a discovered API endpoint. Use this after finding tools via toolsearch. Known endpoints: /api/maps, /api/wehicles, /api/books",
            "parameters": {
                "type": "object",
                "properties": {
                    "endpoint": {"type": "string", "description": "API endpoint path, e.g. /api/maps"},
                    "query": {"type": "string", "description": "Query to send to the endpoint"}
                },
                "required": ["endpoint", "query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": "Submit the final answer to the savethem task. The answer is an array where the first element is the vehicle name and the rest are directions (up/down/left/right) or 'dismount'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Array of steps: [vehicle_name, direction, direction, ..., dismount, direction, ...]"
                    }
                },
                "required": ["answer"]
            }
        }
    },
]

# ============================================================================
# SYSTEM PROMPT — "stwarzanie warunków" zamiast sztywnych instrukcji
# ============================================================================
# Porównaj to z lekcją:
#   "Nie ma jasnych wytycznych o tym, jak połączyć oraz wykorzystać posiadaną wiedzę.
#    Nie ma więc tutaj instrukcji 'jeśli X to Y', lecz bardziej dynamiczne
#    dopasowanie do bieżącej sytuacji."
#
# Prompt opisuje CEL i OGRANICZENIA, ale NIE mówi:
# - w jakiej kolejności odpytywać API
# - jak obliczyć optymalną trasę
# - kiedy wybrać dismount
# - który pojazd jest najlepszy
#
# To wszystko LLM musi WYDEDUKOWAĆ sam na podstawie danych, które zbierze.
# Każde uruchomienie może pójść inną ścieżką rozumowania.
SYSTEM_PROMPT = """You are an agent tasked with planning an optimal route for a messenger to reach the city of Skolwin.

Your mission:
1. Use toolsearch to discover available API tools
2. Gather the terrain map for Skolwin
3. Learn about available vehicles and their fuel/food consumption
4. Learn terrain rules (what's passable, what's blocked, special costs)
5. Plan the optimal route considering:
   - You have 10 units of fuel and 10 portions of food
   - Faster vehicles burn more fuel but less food, slower ones burn more food
   - You can dismount from a vehicle to continue on foot
   - Vehicle is chosen only at the start
   - The map is 10x10 grid
6. Submit the answer as an array: ["vehicle_name", "direction", "direction", ...]

Directions are: up, down, left, right. You can also use "dismount" to switch to walking.

The map uses: S=start, G=goal, .=open, W=water, T=trees, R=rocks.

Think carefully about resource management. Plan step by step.

IMPORTANT: All tools communicate only in English!"""


def execute_tool(name, args):
    """
    Wykonuje narzędzie i zwraca wynik w formacie JSON.

    To jest "warstwa wykonawcza" — most między decyzjami LLM a światem zewnętrznym.
    Agent nie ma bezpośredniego dostępu do API; musi poprosić o wywołanie narzędzia.
    Host (ten skrypt) decyduje, czy i jak je wykonać.

    W lekcji odpowiada to: "rola hosta, ponieważ wówczas odpowiada on nie tylko
    za interakcję między użytkownikiem a agentem, ale także obsługę interfejsu."
    """
    if name == "toolsearch":
        r = requests.post(f"{HUB}/api/toolsearch", json={"apikey": API_KEY, "query": args["query"]})
        return r.json()
    elif name == "api_query":
        r = requests.post(f"{HUB}{args['endpoint']}", json={"apikey": API_KEY, "query": args["query"]})
        return r.json()
    elif name == "submit_answer":
        r = requests.post(f"{HUB}/verify", json={"apikey": API_KEY, "task": "savethem", "answer": args["answer"]})
        return r.json()
    else:
        return {"error": f"Unknown tool: {name}"}


def run_agent():
    """
    Główna pętla agencka (agentic loop).

    To serce niedeterministycznego podejścia. Pętla działa tak:
    1. Wyślij dotychczasową historię do LLM
    2. LLM decyduje: wywołać narzędzie? które? z jakimi parametrami?
    3. Wykonaj narzędzie, dodaj wynik do historii
    4. Powtórz — LLM widzi wynik i decyduje o kolejnym kroku

    Kluczowe: to LLM steruje przepływem, nie kod. Kod tylko:
    - dostarcza narzędzia (TOOLS)
    - wykonuje wywołania (execute_tool)
    - pilnuje limitu kroków (max_steps)

    W naszym uruchomieniu agent:
    - Krok 1-3: odkrył narzędzia i zebrał dane (mapa, pojazdy, reguły)
    - Krok 4-6: rozumował nad trasą, przeliczał zasoby
    - Krok 7: popełnił BŁĄD (wybrał konia → "Food reached zero")
    - Krok 8: SKORYGOWAŁ się — przeliczył matematycznie, wybrał rakietę + dismount
    - Krok 9: sukces!

    Ten cykl błąd → korekta jest niemożliwy w solve.py, ale naturalny dla LLM.
    Lekcja mówi o "odkrywaniu ścieżek, o których trudno byłoby pomyśleć
    na poziomie założeń projektu" — i dokładnie to się tu wydarzyło.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Please find the optimal route to Skolwin and submit it. Start by discovering available tools, then gather all necessary information about the map, vehicles, and terrain rules before planning the route."}
    ]

    step = 0
    max_steps = 25  # zabezpieczenie przed nieskończoną pętlą

    while step < max_steps:
        step += 1
        print(f"\n{'='*60}")
        print(f"STEP {step}")
        print(f"{'='*60}")

        # Wywołanie LLM — to tu zachodzi "niedeterministyczna magia".
        # Ten sam prompt może dać RÓŻNE odpowiedzi przy każdym uruchomieniu.
        # temperature=0.2 ogranicza losowość, ale jej nie eliminuje.
        #
        # Lekcja mówi:
        #   "Zachowanie modelu jest uzależnione od poprzedzającej treści.
        #    Jeśli ta się nie zmienia, to wynik zazwyczaj będzie bardzo podobny."
        # Ale tu treść ZMIENIA SIĘ z każdym krokiem (nowe wyniki narzędzi),
        # więc zachowanie agenta jest dynamiczne i adaptacyjne.
        response = client.chat.completions.create(
            model="anthropic/claude-haiku-4.5",
            messages=messages,
            tools=TOOLS,
            temperature=0.2,
        )

        msg = response.choices[0].message
        print(f"\n[ASSISTANT] {msg.content or '(no text)'}")

        # Jeśli LLM nie wywołuje żadnego narzędzia — uznaje że skończył.
        # To też jest decyzja niedeterministyczna: model SAM ocenia,
        # kiedy zebrał wystarczająco informacji i kiedy zadanie jest ukończone.
        if not msg.tool_calls:
            print("\n=== Agent finished (no more tool calls) ===")
            break

        # Dodaj odpowiedź asystenta do historii — to buduje "pamięć"
        # konwersacji, która wpływa na kolejne decyzje modelu.
        messages.append(msg)

        # Wykonaj każde wywołane narzędzie i dodaj wynik do kontekstu.
        # LLM może wywołać WIELE narzędzi w jednym kroku (parallel tool calls),
        # co odpowiada "proaktywności" opisanej w lekcji.
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            print(f"\n  -> Tool: {tc.function.name}({json.dumps(args, ensure_ascii=False)})")

            result = execute_tool(tc.function.name, args)
            result_str = json.dumps(result, ensure_ascii=False, indent=2)
            print(f"  <- Result: {result_str[:500]}")

            # Wynik narzędzia wraca do LLM jako "tool message".
            # To jest "stan otoczenia" z lekcji — model reaguje na to,
            # co odkrywa, a nie na z góry zaplanowany scenariusz.
            # Np. komunikat "Food reached zero" ZMIENIŁ dalsze rozumowanie agenta.
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_str,
            })

            if tc.function.name == "submit_answer":
                print(f"\n{'='*60}")
                print(f"ANSWER SUBMITTED: {json.dumps(args['answer'])}")
                print(f"RESPONSE: {result_str}")
                print(f"{'='*60}")

    print(f"\nTotal steps: {step}")


if __name__ == "__main__":
    run_agent()
