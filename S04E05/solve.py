import os
from dotenv import load_dotenv
load_dotenv()

"""
S04E05 — foodwarehouse: Agent do zarządzania magazynem żywności i narzędzi
==========================================================================

KONTEKST LEKCJI:
Lekcja S04E05 dotyczy projektowania rozwiązań AI wewnątrz firm. Omawia:
- Jak różni użytkownicy (doświadczeni vs. nowicjusze) inaczej pracują z AI
- Proste wdrożenia AI: checklisty, onboarding, prompty stylistyczne, generatywne UI
- Bezpieczeństwo danych: ryzyko halucynacji, prompt injection, wycieku danych
- MCP Apps i generatywne interfejsy jako narzędzia wewnątrzfirmowe
- Koncept agenta "review" — przetwarzanie dokumentu akapit po akapicie z komentarzami

ZADANIE:
Przeprogramować system dystrybucji magazynu Zygfryda tak, aby jedzenie i narzędzia
trafiły do potrzebujących miast. Agent musi:
1. Pobrać zapotrzebowanie miast z food4cities.json
2. Odczytać bazę SQLite (destinations, users) przez API
3. Wygenerować podpisy SHA1 dla zamówień (signatureGenerator)
4. Utworzyć zamówienie dla każdego miasta z poprawnymi danymi
5. Uzupełnić zamówienia towarami
6. Wywołać "done" do weryfikacji

ARCHITEKTURA AGENTA:
Ten skrypt jest "agentem" w tym sensie, że:
- Ma zdefiniowane NARZĘDZIA (tools) — funkcje do komunikacji z API magazynu
- Działa KROK PO KROKU — każdy krok zależy od wyników poprzedniego
- Podejmuje DECYZJE — np. którego użytkownika wybrać, jak mapować miasta
- Ma mechanizm OBSŁUGI BŁĘDÓW — reset i retry
- LOGUJE swoje działania — żebyśmy widzieli co się dzieje

W prawdziwym wdrożeniu firmowym taki agent mógłby:
- Być podłączony do interfejsu czatu (jak opisano w lekcji)
- Mieć MCP Apps do wizualizacji zamówień
- Być częścią większego pipeline'u z human-in-the-loop
"""

import requests
import json
import sys

# ============================================================================
# KONFIGURACJA
# W projekcie edukacyjnym klucze są hardkodowane. W produkcji użylibyśmy
# zmiennych środowiskowych lub vault'a (np. AWS Secrets Manager).
# Lekcja S04E05 podkreśla, że nawet w bezpiecznym środowisku (Bedrock/Azure)
# agent może przypadkowo ujawnić dane — dlatego ograniczamy uprawnienia.
# ============================================================================
API_URL = "https://hub.ag3nts.org/verify"
API_KEY = os.environ["HUB_API_KEY"]
TASK = "foodwarehouse"
FOOD4CITIES_URL = "https://hub.ag3nts.org/dane/food4cities.json"


def call_api(answer: dict) -> dict:
    """
    Główne narzędzie agenta — wysyła żądanie do API magazynu.

    WZORZEC Z LEKCJI:
    Każde narzędzie agenta powinno mieć jasno zdefiniowany interfejs.
    Tutaj API wymaga trzech pól: apikey, task, answer (z polem tool).
    To odpowiada koncepcji MCP — serwer udostępnia narzędzia,
    a agent (klient) z nich korzysta.
    """
    payload = {
        "apikey": API_KEY,
        "task": TASK,
        "answer": answer,
    }
    resp = requests.post(API_URL, json=payload)
    resp.raise_for_status()
    data = resp.json()
    return data


def log(step: str, msg: str):
    """Prosty logger — w produkcji użylibyśmy structured logging."""
    print(f"[{step}] {msg}")


# ============================================================================
# KROK 1: Pobranie zapotrzebowania miast
# Dane zewnętrzne — to jest nasz "kontekst" dla agenta.
# W lekcji mowa o tym, że agent potrzebuje danych z wielu źródeł
# (narzędzi, baz danych, plików) — tu łączymy plik JSON + SQLite + API.
# ============================================================================
log("1/7", "Pobieram zapotrzebowanie miast z food4cities.json...")
food_resp = requests.get(FOOD4CITIES_URL)
food_resp.raise_for_status()
food4cities = food_resp.json()
log("1/7", f"Miasta do obsłużenia: {list(food4cities.keys())}")
for city, items in food4cities.items():
    log("1/7", f"  {city}: {items}")

# ============================================================================
# KROK 2: Odczyt bazy danych — tabela destinations
# Agent musi znaleźć kody docelowe (destination_id) dla każdego miasta.
# Używamy narzędzia "database" z akcją SQL SELECT.
#
# WAŻNE Z LEKCJI: Baza jest read-only — to przykład ograniczania uprawnień
# agenta. Gdyby agent miał dostęp do zapisu, mógłby przypadkowo uszkodzić
# dane (jeden z ryzyk wymienionych w sekcji o prywatności danych).
# ============================================================================
log("2/7", "Odczytuję tabelę destinations z bazy SQLite...")

# API zwraca max 30 wierszy, a tabela ma 40 — musimy pobrać konkretne miasta
city_destinations = {}
for city_name in food4cities.keys():
    # Szukamy case-insensitive — nazwy w bazie mają wielką literę
    result = call_api({
        "tool": "database",
        "query": f"SELECT * FROM destinations WHERE LOWER(name) = '{city_name.lower()}'"
    })
    if result.get("rows"):
        row = result["rows"][0]
        city_destinations[city_name] = row["destination_id"]
        log("2/7", f"  {city_name} -> destination_id: {row['destination_id']}")
    else:
        log("2/7", f"  UWAGA: Nie znaleziono destination dla {city_name}!")
        sys.exit(1)

# ============================================================================
# KROK 3: Odczyt bazy danych — tabela users
# Potrzebujemy użytkownika (creatorID, login, birthday) do podpisu zamówienia.
# Wybieramy aktywnego użytkownika z rolą "Obsługa transportów" (role=2)
# — to najbardziej logiczny wybór dla zamówień magazynowych.
#
# WZORZEC AGENTA: Agent podejmuje decyzję (wybór użytkownika) na podstawie
# kontekstu (rola użytkownika). To jest "reasoning" — agent nie zgaduje,
# lecz wybiera na podstawie danych.
# ============================================================================
log("3/7", "Wybieram użytkownika do tworzenia zamówień...")
users_result = call_api({
    "tool": "database",
    "query": "SELECT * FROM users WHERE role = 2 AND is_active = 1 LIMIT 1"
})
user = users_result["rows"][0]
log("3/7", f"  Wybrany użytkownik: {user['login']} (ID: {user['user_id']}, "
    f"urodziny: {user['birthday']})")

# ============================================================================
# KROK 4: Reset stanu zamówień
# Zaczynamy od czystego stanu — usuwamy ewentualne wcześniejsze próby.
# To odpowiada zasadzie z lekcji: "jeśli po drodze namieszasz, użyj reset".
# ============================================================================
log("4/7", "Resetuję stan zamówień do początkowego...")
reset_result = call_api({"tool": "reset"})
log("4/7", f"  Reset: {reset_result.get('message', 'OK')}")

# Usuwamy zamówienia seedowe (te które istnieją po resecie)
orders_result = call_api({"tool": "orders", "action": "get"})
for order in orders_result.get("orders", []):
    delete_result = call_api({
        "tool": "orders",
        "action": "delete",
        "id": order["id"]
    })
    log("4/7", f"  Usunięto seedowe zamówienie: {order['title']}")

# ============================================================================
# KROK 5: Generowanie podpisów i tworzenie zamówień
# Dla każdego miasta generujemy podpis SHA1 i tworzymy zamówienie.
#
# PODPIS (signature):
# API signatureGenerator wymaga: login, birthday, destination.
# To mechanizm autoryzacji — w lekcji mowa o tym, że agent musi
# mieć odpowiednie uprawnienia. Podpis zapewnia, że zamówienie
# zostało utworzone przez autoryzowanego użytkownika.
#
# WZORZEC AGENTA: Agent wykonuje sekwencję kroków zależnych od siebie:
# 1. Generuj podpis -> 2. Utwórz zamówienie -> 3. Dodaj towary
# To jest "chain of actions" — każdy krok wymaga wyniku poprzedniego.
# ============================================================================
log("5/7", "Tworzę zamówienia dla każdego miasta...")

order_ids = {}  # city -> order_id

for city_name, items in food4cities.items():
    dest_id = city_destinations[city_name]

    # 5a. Generowanie podpisu SHA1
    sig_result = call_api({
        "tool": "signatureGenerator",
        "action": "generate",
        "login": user["login"],
        "birthday": user["birthday"],
        "destination": dest_id,
    })
    signature = sig_result.get("hash")
    log("5/7", f"  [{city_name}] Podpis: {signature}")

    # 5b. Tworzenie zamówienia
    create_result = call_api({
        "tool": "orders",
        "action": "create",
        "title": f"Dostawa dla {city_name.capitalize()}",
        "creatorID": user["user_id"],
        "destination": dest_id,
        "signature": signature,
    })
    order_id = create_result.get("order", {}).get("id")
    order_ids[city_name] = order_id
    log("5/7", f"  [{city_name}] Zamówienie utworzone: {order_id}")

# ============================================================================
# KROK 6: Uzupełnianie zamówień towarami
# Używamy batch mode — jednym wywołaniem dodajemy wszystkie towary.
# To optymalizacja: zamiast N wywołań dla N towarów, robimy 1 na miasto.
#
# LEKCJA O EFEKTYWNOŚCI:
# W kontekście wdrożeń firmowych, minimalizacja wywołań API to
# nie tylko kwestia wydajności, ale też kosztów (tokeny LLM, rate limits).
# ============================================================================
log("6/7", "Uzupełniam zamówienia towarami (batch mode)...")

for city_name, items in food4cities.items():
    order_id = order_ids[city_name]
    append_result = call_api({
        "tool": "orders",
        "action": "append",
        "id": order_id,
        "items": items,  # batch: {"chleb": 45, "woda": 120, "mlotek": 6}
    })
    log("6/7", f"  [{city_name}] Towary dodane: {items}")

# ============================================================================
# KROK 7: Weryfikacja — wywołanie "done"
# Agent kończy pracę i prosi system o weryfikację.
#
# HUMAN-IN-THE-LOOP:
# W lekcji podkreślono, że agent powinien mieć mechanizm weryfikacji.
# Tu Centrala pełni rolę "człowieka" — sprawdza poprawność zamówień.
# W realnym wdrożeniu przed "done" moglibyśmy pokazać podsumowanie
# w generatywnym UI (MCP Apps) i poprosić użytkownika o akceptację.
# ============================================================================
log("7/7", "Weryfikuję kompletność zamówień...")

# Najpierw sprawdźmy co mamy
final_orders = call_api({"tool": "orders", "action": "get"})
log("7/7", f"  Liczba zamówień: {final_orders.get('count', 0)}")
for order in final_orders.get("orders", []):
    log("7/7", f"  - {order['title']}: {order['items']}")

# Finalna weryfikacja
done_result = call_api({"tool": "done"})
log("7/7", f"  Wynik: {json.dumps(done_result, indent=2, ensure_ascii=False)}")

# ============================================================================
# PODSUMOWANIE
# Ten agent pokazuje kluczowe koncepcje z lekcji S04E05:
#
# 1. NARZĘDZIA (tools): Agent używa API jak narzędzi MCP — orders, database,
#    signatureGenerator, done, reset. Każde ma jasny interfejs.
#
# 2. SEKWENCYJNE PRZETWARZANIE: Jak agent "review" z lekcji przetwarza
#    dokument akapit po akapicie, tak ten agent przetwarza miasta jedno po
#    drugim — pobiera dane, generuje podpis, tworzy zamówienie, dodaje towary.
#
# 3. OGRANICZANIE UPRAWNIEŃ: Baza jest read-only. Agent nie może uszkodzić
#    danych źródłowych. To realizacja zasady "least privilege" z lekcji.
#
# 4. KONTEKST Z WIELU ŹRÓDEŁ: Agent łączy dane z pliku JSON, bazy SQLite
#    i API — to typowy scenariusz firmowy opisany w lekcji (przenoszenie
#    danych między narzędziami).
#
# 5. WERYFIKACJA: Mechanizm "done" to odpowiednik human-in-the-loop —
#    system sprawdza poprawność przed akceptacją.
#
# 6. DETERMINIZM: Choć lekcja mówi o niedeterministycznej naturze LLM,
#    ten agent jest w pełni deterministyczny — nie używa LLM do decyzji.
#    To świadomy wybór: gdy proces jest dobrze zdefiniowany, nie potrzeba AI.
#    AI przydaje się gdy trzeba interpretować, klasyfikować lub generować.
# ============================================================================
