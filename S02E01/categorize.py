import os
from dotenv import load_dotenv
load_dotenv()

import csv
import io
import requests
from openai import OpenAI

# ============================================================
# S02E01 — Zarządzanie kontekstem w konwersacji
# ============================================================
#
# Lekcja S02E01 skupia się na Context Engineering — projektowaniu tego,
# co trafia do okna kontekstowego modelu i jak wpływa to na jego zachowanie.
#
# Zadanie "categorize" demonstruje kilka kluczowych konceptów naraz:
#
# 1. PROMPT CACHING (główna technika kosztowa)
#    System klasyfikuje 10 towarów osobnymi zapytaniami. Przy budżecie 1.5 PP
#    każdy token się liczy. Trick: statyczna część promptu (instrukcje) zawsze
#    jest taka sama → provider cache'uje ją → kolejne zapytania płacą ~0.5x
#    za tokeny wejściowe. Warunek: dynamiczne dane ({id}, {description})
#    MUSZĄ być na KOŃCU promptu, żeby prefiks był identyczny.
#
# 2. ARCHITEKTURA "MODEL INŻYNIERUJE PROMPT DLA MODELU"
#    Silny model (Claude Sonnet) piszą prompty dla słabego modelu docelowego
#    (archaiczny klasyfikator z oknem 100 tokenów). To meta-poziom:
#    AI jako "inżynier promptów" iteracyjnie poprawia instrukcje na podstawie
#    feedbacku z systemu docelowego. Lekcja S02E01: "podejście agentowe —
#    użyj modelu LLM jako inżyniera promptów".
#
# 3. SIGNAL vs NOISE w ograniczonym oknie
#    100 tokenów to za mało na elaborate instrukcje. Każde słowo musi
#    nieść maksymalny "sygnał" — pojęcie z lekcji S02E01. Angielski jest
#    krótszy od polskiego, więc to wybór kontekstowy, nie estetyczny.
#
# 4. WYJĄTEK ZAKODOWANY W PROMPCIE (celowe "zatrucie" klasyfikatora)
#    Kasety do reaktora to "DNG" semantycznie, ale mają być klasyfikowane NEU.
#    Prompt musi zawierać regułę override, która jest sprzeczna z intuicją
#    modelu. To przykład jak instrukcja systemowa może zmieniać zachowanie modelu.

HUB_API_KEY = os.environ["HUB_API_KEY"]
HUB_URL = "https://hub.ag3nts.org/verify"
CSV_URL = f"https://hub.ag3nts.org/data/{HUB_API_KEY}/categorize.csv"

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)


def fetch_csv():
    """Pobiera aktualną listę towarów z HUB-a.

    Uwaga: CSV zmienia się co kilka minut — dlatego pobieramy je ŚWIEŻO
    przed każdą próbą, nie raz na początku. To przykład zasady z lekcji:
    nie zakładaj, że zewnętrzne dane są stabilne. Pobieraj tuż przed użyciem.
    """
    r = requests.get(CSV_URL)
    r.raise_for_status()
    reader = csv.DictReader(io.StringIO(r.text))
    items = list(reader)
    print(f"Pobrano {len(items)} towarów:")
    for item in items:
        print(f"  {item}")
    return items


def reset_budget():
    """Resetuje licznik budżetu w HUB-ie do wartości startowej (1.5 PP).

    Bez resetu każda nieudana próba klasyfikacji wyczerpuje budżet dalej.
    To mechanizm "clean slate" — lekcja S02E01 sugeruje reset before każdą
    próbą, aby mieć pełny budżet na testowanie nowej wersji promptu.
    """
    r = requests.post(HUB_URL, json={
        "apikey": HUB_API_KEY,
        "task": "categorize",
        "answer": {"prompt": "reset"}
    })
    print(f"Reset: {r.json()}")


def classify_item(prompt_template: str, item: dict) -> dict:
    """Wysyła jeden towar do archaicznego klasyfikatora przez HUB.

    Kluczowy detal: {id} i {description} są PODSTAWIANE NA KOŃCU szablonu.
    Dzięki temu prefiks (statyczne instrukcje) jest identyczny dla wszystkich
    10 zapytań w danej próbie → provider może go cache'ować.

    Schemat kosztu z lekcji S02E01:
    - pierwsze zapytanie: pełna cena za statyczny prefiks
    - zapytania 2-10:    prefiks z cache (0.5x ceny), tylko dane towaru full-price
    Przy prompcie ~80 tokenów statycznych oszczędność jest znacząca.
    """
    # Interpolacja szablonu — statyczne instrukcje + dynamiczne dane na końcu
    prompt = prompt_template.replace("{id}", item["code"]).replace("{description}", item["description"])
    payload = {
        "apikey": HUB_API_KEY,
        "task": "categorize",
        "answer": {"prompt": prompt}
    }
    r = requests.post(HUB_URL, json=payload)
    return r.json()


def run_classification(prompt_template: str, items: list) -> tuple[bool, list]:
    """Uruchamia pełny cykl klasyfikacji wszystkich 10 towarów.

    Zatrzymuje się natychmiast przy błędzie klasyfikacji lub wyczerpaniu
    budżetu — nie ma sensu kontynuować skoro wiemy, że próba jest nieudana.
    To oszczędność tokenów: wyjście wczesne (fail-fast) zamiast klasyfikowania
    pozostałych towarów za darmo.

    Zwraca (True, wyniki) gdy wszystkie towary OK, (False, wyniki) przy błędzie.
    """
    results = []
    for item in items:
        resp = classify_item(prompt_template, item)
        results.append({"item": item, "response": resp})
        code = resp.get("code", 0)
        msg = resp.get("message", "")
        print(f"  [{item['code']}] {msg[:80]}")
        # Wczesne wyjście: błąd klasyfikacji lub przekroczony budżet
        if code == -1 or "budget" in msg.lower() or "reset" in msg.lower():
            print("  !! Błąd budżetu lub klasyfikacji — przerywam.")
            return False, results
    # Flaga w ostatniej odpowiedzi = wszystkie 10 towarów poprawne
    last = results[-1]["response"]
    if "{FLG:" in str(last):
        return True, results
    all_ok = all(r["response"].get("code", -1) != -1 for r in results)
    return all_ok, results


def engineer_prompt(items: list, previous_prompt: str, error_info: str) -> str:
    """Używa Claude Sonnet jako "meta-promptera" do ulepszenia promptu klasyfikatora.

    To jest serce architektury: silny model (Sonnet) generuje instrukcje
    dla słabego modelu docelowego. Lekcja S02E01 opisuje to jako "agentowe
    podejście do inżynierii promptów" — LLM iteracyjnie poprawia prompt
    na podstawie feedbacku z poprzedniej próby.

    Parametr error_info zawiera konkretne błędy z HUB-a (który towar, jaki błąd).
    Dzięki temu model wie, co poprawić — nie zgaduje w ciemno.

    Ograniczenia w meta-prompcie:
    - 100 tokenów max (twarde ograniczenie systemu docelowego)
    - dynamiczne dane na końcu (wymóg prompt caching)
    - angielski (krótszy = więcej miejsca na instrukcje logiczne)
    - zwróć TYLKO szablon (bez wyjaśnień — będzie użyty bezpośrednio)
    """
    items_str = "\n".join(f"id={i['code']}, description={i['description']}" for i in items)

    messages = [
        {
            "role": "user",
            "content": f"""You are a prompt engineer. You need to write a classification prompt for a legacy system.

REQUIREMENTS:
- The prompt must be max 100 tokens total (including the item data filled in)
- Output only DNG (dangerous) or NEU (neutral)
- Items related to reactor/nuclear fuel cartridges must ALWAYS be classified as NEU (neutral) even if they sound dangerous
- All other dangerous items should be DNG
- Use prompt caching: put static instructions first, dynamic data ({{id}}, {{description}}) at the end
- Write in English to save tokens

ITEMS TO CLASSIFY:
{items_str}

PREVIOUS PROMPT:
{previous_prompt}

ERROR FEEDBACK:
{error_info}

Write an improved prompt template using {{id}} and {{description}} as placeholders.
Return ONLY the prompt template, nothing else."""
        }
    ]

    # Sonnet 4.6 jako "inżynier" — mocniejszy model generuje prompty dla słabszego.
    # max_tokens=200 wystarczy: szablon promptu to ~50-100 tokenów.
    resp = client.chat.completions.create(
        model="anthropic/claude-sonnet-4-6",
        messages=messages,
        max_tokens=200,
    )
    return resp.choices[0].message.content.strip()


# ============================================================
# GŁÓWNA PĘTLA AGENTOWA — iteracyjne doskonalenie promptu
# ============================================================
#
# To jest "agentowe podejście" z lekcji S02E01:
# zamiast ręcznie pisać i testować prompty, tworzymy agenta, który:
#   1. Pobiera dane (CSV)
#   2. Testuje aktualny prompt (run_classification)
#   3. Jeśli błąd: analizuje feedback, generuje poprawiony prompt (engineer_prompt)
#   4. Resetuje budżet i powtarza
#
# Pętla kończy się sukcesem (flaga) lub wyczerpaniem max_attempts.
# Ograniczenie liczby prób = zabezpieczenie przed nieskończoną pętlą (jak w S01E02).

def main():
    print("=== S02E01 Categorize Agent ===\n")

    # Punkt startowy — ręcznie napisany wstępny prompt.
    # Struktura: statyczne instrukcje NAJPIERW → {id}, {description} NA KOŃCU.
    # To wymóg prompt caching: prefiks musi być identyczny dla wszystkich zapytań.
    current_prompt = "Classify item as DNG(dangerous) or NEU(neutral). Reactor/nuclear fuel items=NEU. Item {id}: {description}. Answer:"

    max_attempts = 8

    for attempt in range(1, max_attempts + 1):
        print(f"\n--- Próba {attempt} ---")
        print(f"Prompt: {current_prompt}\n")

        # Zawsze świeże CSV — dane mogą się zmienić między próbami
        items = fetch_csv()

        # Reset budżetu przed każdą próbą poza pierwszą — czyste konto
        if attempt > 1:
            reset_budget()

        success, results = run_classification(current_prompt, items)

        if success:
            print("\nSukces! Wszystkie towary sklasyfikowane poprawnie.")
            for r in results:
                resp = r["response"]
                if "{FLG:" in str(resp):
                    print(f"\nFLAGA: {resp}")
            break

        # Zbierz informacje o błędach — trafią do engineer_prompt jako feedback.
        # Im bardziej precyzyjny feedback, tym lepsze poprawki promptu.
        errors = []
        for r in results:
            resp = r["response"]
            msg = resp.get("message", "")
            if resp.get("code", 0) == -1 or "wrong" in msg.lower() or "incorrect" in msg.lower():
                errors.append(f"Item {r['item']['code']} ({r['item']['description']}): {msg}")

        error_info = "\n".join(errors) if errors else "Budget exceeded or unknown error"
        print(f"\nBłędy: {error_info}")

        if attempt < max_attempts:
            # Claude Sonnet analizuje błędy i pisze lepszą wersję promptu
            print("\nUlepszam prompt przy użyciu Claude...")
            current_prompt = engineer_prompt(items, current_prompt, error_info)
    else:
        print("\n❌ Wyczerpano limit prób.")


if __name__ == "__main__":
    main()
