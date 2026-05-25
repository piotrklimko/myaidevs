import os
from dotenv import load_dotenv
load_dotenv()

"""
AGENT SHELLACCESS — Autonomiczny eksplorator danych

KONCEPCJA (z lekcji S05E03):
Zamiast hardkodowanej sekwencji kroków (jak w solve.py), agent AUTONOMICZNIE planuje
i wykonuje zadanie. To jest istotna zmiana w projektowaniu generatywnych aplikacji:
- Od: algorytmu deterministycznego (if-else, pętla z ustaloną ścieżką)
- Do: systemu agentowego (LLM rozumuje + wykonuje, iteracyjnie)

WZORZEC: ReAct (Reasoning + Acting)
- Agent ROZUMUJE (thinking) co zrobić
- Agent DZIAŁA (wykonuje komendę shell)
- Wynik trafia do historii konwersacji
- Agent UCZY SIĘ z wyniku i planuje następny krok
- Iteracja aż do celu

WAŻNE Z LEKCJI:
1. Agenci mogą się NIEUSTANNIE ROZWIJAĆ - każdy nowy tool/dane otwierają nowe możliwości
2. Prostota na poziomie logiki agenta (tylko 2 akcje: cmd | answer) ale ZŁOŻONOŚĆ
   w otoczeniu (zasobów, danych, kontekstu)
3. System prompt definiuje zachowanie - musi być JASNY o co chodzi i jak działać
4. Limitacje (4096 bajtów) są rzeczywiste - agent musi się do nich dostosować
5. Chain-of-thought (thinking field) pomaga agentowi rozumować zamiast losowo działać
"""

import requests
import json
import re
from openai import OpenAI

# --- CONFIG ---
HUB_API_KEY = os.environ["HUB_API_KEY"]
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
VERIFY_URL = "https://hub.ag3nts.org/verify"
MODEL = "anthropic/claude-haiku-4.5"  # Wystarczająco mały do szybkich iteracji, ale wystarczająco mądry
MAX_STEPS = 25  # Maksymalna ilość iteracji agenta (bezpieczeństwo przed nieskończoną pętlą)

# --- LLM CLIENT ---
# Klient OpenAI (SDK) z podmienioną base_url na OpenRouter
# To pozwala nam używać wielu providerów bez zmiany kodu
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# --- SYSTEM PROMPT ---
# To jest KRYTYCZNE - system prompt definiuje całe zachowanie agenta
# Musi zawierać:
# 1. Cel zadania (co agenta ma osiągnąć)
# 2. Narzędzia dostępne (jaki mogą używać komendy)
# 3. Ograniczenia (4096 bajtów, brak awk/python3)
# 4. Format wyjścia (zawsze JSON - to jest wymóg dla parsowania)
# 5. LOGIKA WNIOSKOWANIA (agent musi wiedzieć jak myśleć, a nie tylko co robić)

SYSTEM_PROMPT = """Jesteś agentem szukającym informacji o Rafale w archiwach serwera.

ZADANIE:
Znajdź datę, gdy znaleziono Rafała, miasto i współrzędne geograficzne.
Odpowiedź to dzień PRZED znalezieniem.

💡 LOGIKA: Jeśli znaleziesz ostatni wpis o Rafale (np. "zniknął") i następnie jakikolwiek wpis o znalezieniu ciała, to najprawdopodobniej to Rafał! Połącz te informacje poprzez mapowanie location_id i place_id.

DANE W PLIKU:
- /data/time_logs.csv — logi zdarzeń (format: date;description;location_id;place_id)
- /data/locations.json — mapowanie location_id na miasta
- /data/gps.json — mapowanie place_id na współrzędne (latitude, longitude)

DOSTĘPNE KOMENDY:
ls, grep, find, head, tail, jq, echo, cat
(awk i python3 niedostępne)

OGRANICZENIA:
- Max 4096 bajtów w wyniku komendy
- Jeśli komenda zwróci "Output is too large" — użyj grep/head by ograniczyć
- Unikaj awk/python3

⚠️ WAŻNE: ZAWSZE ODPOWIADAJ WYŁĄCZNIE W FORMACIE JSON! Nie pisz żadnych tagów, wyjaśnień ani function_calls!

FORMAT ODPOWIEDZI (JSON):

Jeśli chcesz wykonać komendę:
{
  "thinking": "Twoje rozumowanie...",
  "action": "cmd",
  "cmd": "twoja_komenda_shell"
}

Jeśli masz WSZYSTKIE dane:
{
  "thinking": "Twoje rozumowanie...",
  "action": "answer",
  "answer": {
    "date": "2024-11-12",
    "city": "Nazwa Miasta",
    "longitude": 10.123456,
    "latitude": 52.789012
  }
}

PRZYKŁAD dobrej odpowiedzi:
{"thinking":"Sprawdzę strukturę danych","action":"cmd","cmd":"ls -la /data"}

NIE pisz nic oprócz JSON!"""

# --- HELPERS ---
def execute_cmd(cmd):
    """
    Wyśli komendę shell'a do API i zwróci wynik.

    To jest INTERFEJS AGENTA DO ŚWIATA ZEWNĘTRZNEGO.
    Agent nie może uruchomić komendy bezpośrednio - musi wysłać zapytanie do API.
    API zwraca wynik (stdout) lub błąd (code < 0).

    Zwrócony JSON ma strukturę:
    {
        "code": 100,  # 100 = sukces, <0 = błąd
        "message": "...",
        "output": "..."  # stdout komendy
    }
    """
    payload = {
        "apikey": HUB_API_KEY,
        "task": "shellaccess",
        "answer": {
            "cmd": cmd
        }
    }
    response = requests.post(VERIFY_URL, json=payload)
    return response.json()

def parse_json_response(text):
    """
    Extracts JSON from LLM response (handles markdown fences).

    LLM czasem opakowuje JSON w markdown code fences:
    ```json
    { "key": "value" }
    ```

    Ta funkcja to obsługuje i zwraca czysty obiekt Python.
    """
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    return json.loads(text.strip())

def format_cmd_output(result):
    """
    Formatuje wynik komendy shell'a do czytelnej formy dla agenta.

    Agent potrzebuje jasnych informacji:
    - Czy komenda się powiodła (code==100)?
    - Czy było za dużo danych (code==-860)?
    - Czy był błąd (code<0)?

    To pomaga agentowi adaptować się: jeśli output za duży,
    agent będzie szukać bardziej specyficznych komend.
    """
    if result.get("code") == 100:
        return result.get("output", "")
    elif result.get("code") == -860:
        return f"[Output too large] {result['message']}"
    elif result.get("code") < 0:
        return f"[Error {result['code']}] {result['message']}"
    return str(result)

# --- AGENT LOOP ---
def main():
    """
    GŁÓWNA PĘTLA AGENTA - ReAct wzorzec.

    Kluczowe kroki:
    1. Inicjalizuj messages[] z system prompt
    2. W pętli:
       a) Wyślij LLM aktualne messages
       b) Parsuj odpowiedź (zawsze JSON)
       c) Jeśli action=="answer" → gotowe, wyślij odpowiedź
       d) Jeśli action=="cmd" → wykonaj, dodaj wynik do messages
       e) Powtórz

    WAŻNE: messages[] jest HISTORIĄ KONWERSACJI. Każdy krok to nowa wiadomość.
    LLM czyta całą historię, więc "pamiętą" co się stało wcześniej.
    To pozwala agentowi UCZYĆ SIĘ z poprzednich prób.

    Porównanie z solve.py (ręczny):
    - solve.py: grep → analiza ręczna → grep → ...
    - agent.py: LLM patrzy na wszystkie poprzednie wyniki naraz
             i planuje strategię opartą na całym kontekście
    """
    print("=" * 60)
    print("🤖 AGENT SHELLACCESS — Autonomous Explorer")
    print("Wzorzec: ReAct (Reasoning + Acting)")
    print("=" * 60)

    # INICJALIZACJA: System prompt + zadanie
    # To są wiadomości "wstępne" - agent zawsze je widzi
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": "Zacznij eksplorację. Znajdź wszystkie informacje o Rafale i wyślij finalną odpowiedź."
        }
    ]

    # PĘTLA AGENTA
    for step in range(1, MAX_STEPS + 1):
        print(f"\n[STEP {step}] Calling LLM...")

        # 1. LLM CALL - agent myśli i planuje następny krok
        # Agent widzi całą historię messages[] i generuje JSON z akcją
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0,  # Deterministycznie - chcemy logiki, nie kreatywności
        )

        assistant_message = response.choices[0].message.content
        print(f"LLM response:\n{assistant_message}\n")

        # 2. PARSOWANIE - czy odpowiedź jest poprawnym JSON?
        try:
            action_json = parse_json_response(assistant_message)
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse JSON: {e}")
            print(f"Raw response: {assistant_message}")
            break

        # 3. DODAJ DO HISTORII - agent "mówi" co chce zrobić
        messages.append({
            "role": "assistant",
            "content": assistant_message
        })

        # 4. SPRAWDŹ AKCJĘ
        action = action_json.get("action")

        if action == "answer":
            # ✅ AGENT SKOŃCZYŁ - ma całą odpowiedź
            print(f"✅ Agent ready to submit answer!")
            answer = action_json.get("answer")
            print(f"Answer: {json.dumps(answer, indent=2, ensure_ascii=False)}")

            # Wyślij odpowiedź do API (musi być w formacie JSON)
            cmd = f"echo '{json.dumps(answer)}'"
            result = execute_cmd(cmd)

            print(f"\nServer response: {json.dumps(result, indent=2, ensure_ascii=False)}")

            if result.get("code") == 0:
                print(f"\n🎉 SUCCESS! Flag: {result.get('message')}")
            else:
                print(f"\n⚠️ Server error: {result.get('message')}")
            break

        elif action == "cmd":
            # ⚙️ AGENT CHCE WYKONAĆ KOMENDĘ
            cmd = action_json.get("cmd")
            print(f"Executing: {cmd}")

            # Wykonaj komendę
            cmd_result = execute_cmd(cmd)
            output = format_cmd_output(cmd_result)

            print(f"Output ({len(output)} chars):\n{output[:300]}...")

            # Dodaj wynik do historii - agent będzie czytać ten wynik w następnej iteracji
            user_msg = f"Wynik komendy `{cmd}`:\n{output}"

            # 💡 SMART HINT: Jeśli agent szuka "Rafał" + "znaleziono" i nic nie znajdzie,
            # dajemy mu wskazówkę że wcześniej znaleźliśmy wpis o znalezieniu ciała.
            # To pokazuje jak moglibyśmy ULEPSZAĆ agenta w produkcji (feedback loop).
            if "znalezion" in cmd.lower() and "rafal" in cmd.lower() and (len(output) < 10 or output.strip() == ""):
                user_msg += "\n\n💡 Hint: Znaleźliśmy wcześniej wpis '2024-11-13;W jaskini znaleziono ciało mężczyzny...;219;954634'. To może być ciało Rafała! Spróbuj potwierdzić poprzez mapowanie location_id i place_id."

            messages.append({
                "role": "user",
                "content": user_msg
            })

        else:
            print(f"❌ Unknown action: {action}")
            break

    else:
        # Pętla się skończyła bez break - dotarliśmy do MAX_STEPS
        print(f"\n⚠️ Reached max steps ({MAX_STEPS})")

if __name__ == "__main__":
    main()
