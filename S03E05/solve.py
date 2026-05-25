import os
from dotenv import load_dotenv
load_dotenv()

"""
S03E05 — Podejście DETERMINISTYCZNE (algorytmiczne)
====================================================

Kontekst lekcji:
    Lekcja S03E05 mówi o "niedeterministycznej naturze modeli jako przewadze".
    Ten skrypt celowo pokazuje podejście PRZECIWNE — w pełni deterministyczne.
    Człowiek (my) ręcznie odkrywa narzędzia API, zbiera reguły i koduje algorytm,
    który gwarantuje znalezienie optymalnego rozwiązania.

    To podejście działa tu doskonale, ale ma fundamentalne ograniczenie:
    WYMAGA, żebyśmy z góry wiedzieli, czego szukać i jak to połączyć.
    Każdy krok eksploracji (toolsearch → maps → books → wehicles) został
    wykonany ręcznie, a reguły (np. "drzewa zwiększają spalanie o 0.2")
    zostały zakodowane na sztywno po przeczytaniu dokumentacji.

    W lekcji to podejście odpowiada "agentowi działającemu według skryptu" —
    przewidywalnemu, ale pozbawionemu elastyczności i zdolności do adaptacji.
    Gdyby reguły gry się zmieniły (nowy typ terenu, inny format mapy),
    trzeba by RĘCZNIE modyfikować kod. Agent LLM (agent.py) poradziłby
    sobie sam, bo rozumuje nad danymi zamiast podążać za zakodowaną logiką.

Architektura:
    1. ZBIERANIE DANYCH — zapytania do API (maps, wehicles, books)
       wykonane z góry, wyniki wypisane na ekran dla przejrzystości.
    2. MODELOWANIE — ręcznie zakodowane reguły terenu i kosztów
       (które pola blokują, które zwiększają spalanie itd.).
    3. PATHFINDING — algorytm Dijkstry przeszukuje WSZYSTKIE możliwe
       ścieżki z uwzględnieniem zasobów (paliwo/jedzenie) i opcji dismount.
       Gwarantuje znalezienie optymalnej trasy — coś, czego LLM nie gwarantuje.
    4. WERYFIKACJA — symulacja trasy krok po kroku dla potwierdzenia.
    5. WYSYŁKA — automatyczne przesłanie odpowiedzi do centrali.
"""

import requests
import heapq
import json

API_KEY = os.environ["HUB_API_KEY"]
BASE = "https://hub.ag3nts.org"

def api_call(endpoint, query):
    r = requests.post(f"{BASE}{endpoint}", json={"apikey": API_KEY, "query": query})
    return r.json()


# ============================================================================
# FAZA 1: Zbieranie danych z API
# ============================================================================
# Te zapytania wykonaliśmy RĘCZNIE — to my zdecydowaliśmy, czego szukać.
# W podejściu agentowym (agent.py) to LLM decyduje jakie pytania zadać.
# Różnica jest kluczowa: tutaj programista musi ZNAĆ domenę problemu,
# a agent LLM może ją ODKRYWAĆ samodzielnie.

print("=== Tree fuel cost ===")
tree_info = api_call("/api/books", "trees fuel burn extra cost how much additional consumption")
for n in tree_info.get("notes", []):
    print(f"  [{n['id']}] {n['content']}")

print("\n=== Water crossing details ===")
water_info = api_call("/api/books", "water crossing horse walk fuel food cost consumption")
for n in water_info.get("notes", []):
    print(f"  [{n['id']}] {n['content']}")

print("\n=== Rocks info ===")
rocks_info = api_call("/api/books", "rocks blocked impassable all vehicles")
for n in rocks_info.get("notes", []):
    print(f"  [{n['id']}] {n['content']}")


# ============================================================================
# FAZA 2: Zakodowanie modelu świata (reguły gry)
# ============================================================================
# To jest serce podejścia deterministycznego — WSZYSTKIE reguły są zakodowane
# explicite w Pythonie. Nie ma tu miejsca na "interpretację" czy "wnioskowanie".
#
# Porównaj z lekcją:
#   "Agent dąży do tego, aby 'czytać między słowami', 'wychodzić z inicjatywą'
#    czy rozpoznawać różnicę pomiędzy swoją bazową wiedzą a zewnętrznym kontekstem."
#
# W tym skrypcie żadne "czytanie między słowami" nie zachodzi — mamy twarde if-y.

# Mapa 10x10 pobrana z /api/maps dla miasta Skolwin.
# S = start (pozycja startowa wysłannika)
# G = goal (miasto Skolwin — cel podróży)
# . = otwarty teren (przejezdny dla wszystkich)
# W = woda (przejezdna TYLKO pieszo lub konno)
# T = drzewa (przejezdne, ale zwiększają spalanie paliwa o 0.2)
# R = skały (CAŁKOWICIE blokują ruch dla wszystkich środków transportu)
MAP = [
    [".",".",".",".",".",".",".",".","W","W"],  # row 0
    [".",".",".",".",".",".",".","W","W","."],  # row 1
    [".","T",".",".",".",".","W","W",".","."],  # row 2
    [".",".",".",".",".",".","W",".",".",".",],  # row 3
    [".",".","T",".",".",".","W",".","G","."],  # row 4  ← cel G na (4,8)
    [".",".",".",".","R",".","W",".",".",".",],  # row 5
    [".",".",".","R","R",".","W","W",".","."],  # row 6
    ["S","R",".",".",".",".",".","W",".","."],  # row 7  ← start S na (7,0)
    [".",".",".",".",".",".","W","W",".","."],  # row 8
    [".",".",".",".",".","W","W",".",".",".",],  # row 9
]

start = (7, 0)
goal = (4, 8)

# Parametry pojazdów pobrane z /api/wehicles.
# Kluczowy trade-off z lekcji: "im szybciej się poruszasz, tym więcej spalasz
# paliwa, ale im wolniej idziesz, tym więcej konsumujesz prowiantu."
#
# rocket — najszybszy, ale pali 1.0 paliwa/ruch (za to tylko 0.1 jedzenia)
# car    — kompromis: 0.7 paliwa, 1.0 jedzenia
# horse  — zero paliwa, ale 1.6 jedzenia/ruch
# walk   — zero paliwa, ale aż 2.5 jedzenia/ruch (najdroższy w jedzeniu)
VEHICLES = {
    "rocket": {"fuel": 1.0, "food": 0.1},
    "car":    {"fuel": 0.7, "food": 1.0},
    "horse":  {"fuel": 0.0, "food": 1.6},
    "walk":   {"fuel": 0.0, "food": 2.5},
}

# Kierunki ruchu na mapie (north at the top, per /api/books).
DIRS = {
    "up": (-1, 0),     # w górę = zmniejszamy numer wiersza
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

# Zasoby startowe — stałe, nie można ich uzupełnić w trakcie podróży
# (brak stacji benzynowych — info z /api/books "no-gas-stations").
MAX_FUEL = 10.0
MAX_FOOD = 10.0


def can_enter(tile, vehicle):
    """
    Sprawdza czy dany pojazd może wejść na pole.
    Reguły zakodowane na sztywno na podstawie /api/books:
    - R (skały): blokują WSZYSTKO
    - W (woda): przejezdna TYLKO dla horse i walk
    - T, ., S, G: przejezdne dla wszystkich
    """
    if tile == "R":
        return False
    if tile == "W":
        return vehicle in ("horse", "walk")
    return True


def get_cost(tile, vehicle):
    """
    Zwraca (koszt_paliwa, koszt_jedzenia) za wejście na pole.

    Drzewa (T) zwiększają spalanie paliwa o 0.2 dla pojazdów silnikowych.
    Informacja z /api/books (note "trees-and-burn"):
        "entering a tile marked with T increases fuel consumption
         for powered travel by an additional 0.2 units"

    To dobry przykład "wiedzy ukrytej" — nie wynika ona z samej mapy,
    trzeba ją ODKRYĆ pytając odpowiednie API. Agent LLM robi to
    samodzielnie, tu zakodowaliśmy to ręcznie.
    """
    base = VEHICLES[vehicle]
    fuel = base["fuel"]
    food = base["food"]
    if tile == "T":
        fuel += 0.2  # dodatkowe spalanie za drzewa
    return fuel, food


# ============================================================================
# FAZA 3: Algorytm Dijkstry — gwarantowane optimum
# ============================================================================
# To jest KLUCZOWA przewaga podejścia deterministycznego:
# algorytm przeszukuje WSZYSTKIE możliwe ścieżki i GWARANTUJE znalezienie
# najkrótszej, która mieści się w limitach zasobów.
#
# Agent LLM (agent.py) tego NIE gwarantuje — rozumuje heurystycznie,
# może popełnić błąd (i popełnił — najpierw wybrał konia, zabrakło jedzenia).
# Ale za to LLM potrafi ADAPTOWAĆ się do nowych sytuacji bez zmiany kodu.
#
# Stan w algorytmie: (wiersz, kolumna, aktualny_pojazd)
# Zasoby (paliwo, jedzenie) są śledzone jako część ścieżki.
# Opcja "dismount" pozwala przesiąść się z pojazdu na piechotę w dowolnym momencie.

def solve():
    results = []

    for start_vehicle in ["rocket", "car", "horse", "walk"]:
        # Próbujemy każdy pojazd startowy osobno.
        # Pojazd można wybrać TYLKO na początku podróży (/api/books "vehicle-selection").

        # Kolejka priorytetowa: (liczba_ruchów, paliwo*100, jedzenie*100, wiersz, kolumna, pojazd, ścieżka)
        # Mnożymy zasoby ×100 i zamieniamy na int, żeby uniknąć problemów z precyzją float.
        pq = [(0, int(MAX_FUEL*100), int(MAX_FOOD*100), start[0], start[1], start_vehicle, [start_vehicle])]

        # Słownik odwiedzonych stanów → najlepsze zasoby w tym stanie.
        # Dzięki temu nie eksplorujemy ponownie stanów, do których dotarliśmy
        # z lepszymi lub równymi zasobami — drastycznie przycinamy przestrzeń.
        visited = {}
        found = False

        while pq:
            moves, fuel, food, r, c, veh, path = heapq.heappop(pq)

            # Sprawdź czy dotarliśmy do celu
            if (r, c) == goal:
                real_fuel = fuel / 100
                real_food = food / 100
                print(f"\n=== SOLUTION with {start_vehicle} ===")
                print(f"  Moves: {moves}, Fuel left: {real_fuel}, Food left: {real_food}")
                print(f"  Path: {path}")
                results.append((moves, real_fuel, real_food, path))
                found = True
                break

            # Pruning: jeśli byliśmy tu z lepszymi zasobami, pomijamy
            state = (r, c, veh)
            if state in visited:
                old_fuel, old_food = visited[state]
                if fuel <= old_fuel and food <= old_food:
                    continue
            visited[state] = (fuel, food)

            # Próbuj ruch w każdym z 4 kierunków
            for dir_name, (dr, dc) in DIRS.items():
                nr, nc = r + dr, c + dc
                if 0 <= nr < 10 and 0 <= nc < 10:
                    tile = MAP[nr][nc]
                    if not can_enter(tile, veh):
                        continue

                    fuel_cost, food_cost = get_cost(tile, veh)
                    new_fuel = fuel - int(fuel_cost * 100)
                    new_food = food - int(food_cost * 100)

                    # Jeśli zasoby wystarczą — dodaj do kolejki
                    if new_fuel >= 0 and new_food >= 0:
                        new_path = path + [dir_name]
                        heapq.heappush(pq, (moves + 1, new_fuel, new_food, nr, nc, veh, new_path))

            # Opcja dismount: przesiadka na piechotę (nie kosztuje ruchu).
            # To kluczowa mechanika — pozwala np. lecieć rakietą do granicy wody,
            # a potem przejść pieszo przez wodę do celu.
            if veh != "walk":
                state_walk = (r, c, "walk")
                if state_walk not in visited or fuel > visited[state_walk][0] or food > visited[state_walk][1]:
                    new_path = path + ["dismount"]
                    heapq.heappush(pq, (moves, fuel, food, r, c, "walk", new_path))

        if not found:
            print(f"\n=== NO SOLUTION with {start_vehicle} ===")

    return results


results = solve()

if results:
    # Wybierz najlepszy wynik (najmniej ruchów)
    best = min(results, key=lambda x: x[0])
    print(f"\n=== BEST SOLUTION ===")
    print(f"Moves: {best[0]}, Fuel: {best[1]}, Food: {best[2]}")
    print(f"Path: {best[3]}")
    answer = best[3]
    print(f"Answer JSON: {json.dumps(answer)}")

    # ========================================================================
    # FAZA 4: Weryfikacja — symulacja trasy krok po kroku
    # ========================================================================
    # Dodatkowe zabezpieczenie typowe dla podejścia deterministycznego:
    # po znalezieniu trasy symulujemy ją, żeby upewnić się, że zasoby wystarczą.
    # Agent LLM tego nie robi — "ufa" swoim obliczeniom (i czasem się myli).
    print("\n=== VERIFYING PATH ===")
    r, c = start
    fuel, food = MAX_FUEL, MAX_FOOD
    veh = None
    for i, step in enumerate(answer):
        if step in VEHICLES:
            veh = step
            print(f"  Step {i}: Select vehicle '{veh}' at ({r},{c})")
        elif step == "dismount":
            veh = "walk"
            print(f"  Step {i}: Dismount to walk at ({r},{c})")
        else:
            dr, dc = DIRS[step]
            nr, nc = r + dr, c + dc
            tile = MAP[nr][nc]
            fc, fdc = get_cost(tile, veh)
            fuel -= fc
            food -= fdc
            r, c = nr, nc
            print(f"  Step {i}: {step} -> ({r},{c}) tile={tile} fuel={fuel:.1f} food={food:.1f}")

    print(f"\n  Final position: ({r},{c}), Goal: {goal}")
    print(f"  Reached goal: {(r,c) == goal}")

    # ========================================================================
    # FAZA 5: Wysyłka odpowiedzi
    # ========================================================================
    print("\n=== SUBMITTING ===")
    resp = requests.post(f"{BASE}/verify", json={
        "apikey": API_KEY,
        "task": "savethem",
        "answer": answer
    })
    print(f"Response: {resp.text}")
