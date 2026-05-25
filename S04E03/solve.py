import os
from dotenv import load_dotenv
load_dotenv()

"""
S04E03 — Domatowo: Misja ratunkowa (agent taktyczny)
=====================================================

KONTEKST LEKCJI (S04E03 — Kontekstowa współpraca z AI):
=========================================================
Lekcja dotyczy integracji agentów AI z codziennymi narzędziami i procesami.
To rozszerzenie lekcji S03E03 (agenci) i S04E01 (cyfrowy ogród) — teraz skupiamy
się na agentach działających W TLE, autonomicznie, bez bezpośredniej interakcji.

Kluczowe koncepcje z lekcji:

  1. IZOLACJA AGENTÓW
     - Każdy agent odpowiada za swój obszar, nie ma zbędnych zależności
     - Agent klasyfikujący zgłoszenia nie musi wiedzieć o agencie robiącym zestawienia
     - W tym skrypcie: każdy klaster B3 jest obsługiwany niezależnie
       (oddzielny transporter, oddzielni zwiadowcy, oddzielna trasa)

  2. KONTEKSTOWE DOPASOWANIE
     - Agent najpierw rozpoznaje środowisko (mapę), potem planuje
     - Decyzje są oparte na danych (pozycje B3, sieć dróg, koszty AP)
     - To jak agent podłączony do kalendarza — najpierw czyta zdarzenia,
       potem sugeruje optymalizacje

  3. REAKTYWNE PRZETWARZANIE
     - Po inspekcjach agent analizuje logi i reaguje na wyniki
     - Użycie LLM do klasyfikacji logów — bo heurystyki słów kluczowych
       nie radzą sobie z naturalnym językiem (negacje, fałszywe pozytywy)
     - To odpowiednik agenta monitorującego sygnały z różnych źródeł

  4. OPTYMALIZACJA ZASOBÓW
     - Transportery (1 AP/pole) dowożą zwiadowców (7 AP/pole) blisko celów
     - To jak "batch processing" w automatyzacjach — zamiast wielu drogich
       operacji, jedna tania zbiorcza (np. hurtowe przetwarzanie e-maili)

  5. OBSERWACJA SKUTECZNOŚCI
     - Śledzenie wydatków AP (expenses) pozwala ocenić efektywność
     - Analogia do biznesu: monitoring wskaźników (MRR, Churn, NPS)
       aby wykryć problemy zanim staną się krytyczne

ZADANIE:
========
Na mapie 11×11 zniszczonego miasta Domatowo ukrywa się partyzant
w "jednym z najwyższych bloków" (B3 = Blok 3-piętrowy).
Mamy 300 punktów akcji (AP). Musimy go znaleźć i wezwać helikopter.

PODEJŚCIE AGENTOWE (plan → execute → observe → react):
======================================================
1. Pobranie mapy i analiza terenu (identyfikacja B3 + sieć dróg)
2. Planowanie tras — transportery (1 AP/pole) dowożą zwiadowców blisko celu
3. Zwiadowcy (7 AP/pole) chodzą pieszo i inspekcjonują budynki (1 AP)
4. Po inspekcjach LLM klasyfikuje logi (które pole ma pozytywny wynik)
5. Wezwanie helikoptera na wskazane pole

KOSZTY AKCJI (budżet 300 AP):
  - Transporter: 5 AP bazowo + 5 AP za każdego pasażera
  - Ruch transportera: 1 AP / pole (tylko po drogach!)
  - Ruch zwiadowcy: 7 AP / pole (dowolny teren, ortogonalnie)
  - Inspekcja: 1 AP
  - Wysadzenie (dismount): 0 AP

MAPA DRÓG (symbole UL = "Ulica"):
  Row 1:  B1, C1, D1
  Row 2:  D2, E2, I2
  Row 3:  D3, I3
  Row 4:  D4, I4
  Row 5:  D5, I5
  Row 6:  A6-J6 (główna arteria — tu spawn jednostek)
  Row 7:  D7
  Row 8:  D8
  Row 9:  B9-J9
  Row 10-11: brak dróg (tu są klastry B3!)

KLASTRY B3 (cele inspekcji):
  Klaster 1 (góra):       F1, G1, F2, G2      (4 pola)
  Klaster 2 (dół-lewo):   A10, B10, C10,       (6 pól)
                           A11, B11, C11
  Klaster 3 (dół-prawo):  H10, I10, H11, I11   (4 pola)
  RAZEM: 14 pól do przeszukania

KOSZTORYS (typowy przebieg ~185 AP / 300 AP):
  Tworzenie: 3 transportery × (5+5×pasażerowie) = 10+15+10 = 35 AP
  Transport: ~8+8+9 = ~25 AP (transportery po drogach)
  Zwiadowcy: ~14 inspekcji × 1 + ~20 ruchów × 7 = ~154 AP
  Odczyt logów i LLM: 0 AP (to zewnętrzne narzędzia)
"""

import requests
import json
from openai import OpenAI

# ============================================================
# KONFIGURACJA
# W projekcie edukacyjnym klucze są hardkodowane.
# W produkcji byłyby w zmiennych środowiskowych lub secret manager.
# ============================================================

API_URL = "https://hub.ag3nts.org/verify"
API_KEY = os.environ["HUB_API_KEY"]
TASK = "domatowo"

# Klient LLM przez OpenRouter — używamy go do klasyfikacji logów
# (patrz FAZA 6). To kluczowy element "agentowego" podejścia:
# zamiast heurystyk słów kluczowych, LLM rozumie kontekst.
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
LLM_MODEL = "anthropic/claude-haiku-4.5"

# ============================================================
# KOMUNIKACJA Z API
# ============================================================
# Każde wywołanie to jeden request POST z JSON-em.
# Agent wysyła akcję i analizuje odpowiedź — to serce
# pętli agentowej: act → observe → decide → act.
#
# Analogia z lekcji: to tak jak agent podłączony do API
# Google Calendar, Linear czy CRM — wysyła zapytania,
# odczytuje stan, podejmuje decyzje na podstawie danych.
# ============================================================

def send(action_data: dict) -> dict:
    """Wysyła akcję do API Domatowa i zwraca odpowiedź JSON.

    API przyjmuje format:
        { apikey, task, answer: { action: "...", ...params } }

    Logujemy każdą akcję — transparentność jest kluczowa w systemach
    agentowych. Bez logów nie wiemy, co agent robi i dlaczego.
    To odpowiednik "obserwacji skuteczności" z lekcji — jak agenci
    mogą monitorować swoje własne działania i wykrywać problemy.
    """
    payload = {
        "apikey": API_KEY,
        "task": TASK,
        "answer": action_data
    }
    resp = requests.post(API_URL, json=payload)
    data = resp.json()
    action = action_data.get("action", "?")
    msg = data.get("message", "")
    code = data.get("code", "?")
    print(f"  [{action:>15}] code={code} | {msg[:120]}")
    return data


def get_objects() -> dict:
    """Pobiera aktualny stan jednostek na mapie.

    Zwraca słownik {id: {typ, position, ...}} — pozwala agentowi
    śledzić, gdzie są jego jednostki. To odpowiednik getObjects
    w systemach do zarządzania zadaniami (Linear, ClickUp) —
    agent musi wiedzieć, co jest w grze, zanim podejmie decyzję.
    """
    data = send({"action": "getObjects"})
    units = {}
    if "objects" in data:
        for obj in data["objects"]:
            units[obj["id"]] = obj
    return units


# ============================================================
# ANALIZA MAPY — "Context Gathering"
# ============================================================
# Agent najpierw rozpoznaje teren. To odpowiednik fazy
# "context gathering" w architekturze agentowej:
#   - Zanim agent zarządzi kalendarzem, musi go przeczytać
#   - Zanim agent sklasyfikuje zgłoszenie, musi je zrozumieć
#   - Zanim agent zaplanuje trasę, musi poznać mapę
#
# Bez kontekstu agent podejmuje złe decyzje — wysyła
# zwiadowców w losowe miejsca, przepala budżet AP.
# Z kontekstem wie, że B3 = najwyższe bloki, i skupia
# się TYLKO na nich (zawężenie search space).
# ============================================================

def find_b3_tiles(grid: list) -> list:
    """Identyfikuje wszystkie pola B3 (Blok 3-piętrowy) na mapie.

    Partyzant powiedział: "Ukryłem się w jednym z najwyższych bloków"
    → B3 (3-piętrowy) to najwyższe budynki. Są też B1 i B2, ale
    partyzant wyraźnie mówi o "najwyższych" — to nasz search space.

    Konwersja współrzędnych (grid → API):
      - grid[row][col] → litera = chr(65+col), numer = row+1
      - Przykład: grid[0][5] → F1 (kolumna F = 5+65='F', wiersz 0+1=1)
      - API używa formatu szachowego: A1..K11
    """
    tiles = []
    for row in range(len(grid)):
        for col in range(len(grid[row])):
            if grid[row][col] == "block3":
                coord = chr(65 + col) + str(row + 1)
                tiles.append(coord)
    return tiles


# ============================================================
# PLAN OPERACJI — "Static Plan + Dynamic Execution"
# ============================================================
# Definiujemy 3 klastry B3 i planujemy optymalną alokację
# zasobów. To "statyczny plan" — ale wykonanie jest dynamiczne
# (agent reaguje na wyniki inspekcji w czasie rzeczywistym).
#
# Kluczowe decyzje planistyczne:
#   - Który transporter jedzie gdzie (po drogach)
#   - Ilu zwiadowców zabiera (trade-off: koszt tworzenia vs pokrycie)
#   - Kolejność inspekcji w każdym klastrze (zygzak minimalizuje kroki)
#
# Analogia z lekcji: to jak "szablony projektowe" dla powtarzalnych
# zadań — z góry wiemy, jakie kroki wykonać, ale agent dopasowuje
# się do aktualnych warunków (np. przerywa, gdy cel znaleziony).
#
# IZOLACJA KLASTRÓW (kluczowa koncepcja z lekcji):
# Każdy klaster jest niezależny — ma swój transporter, swoich
# zwiadowców, swoją trasę. Nie ma zależności między klastrami.
# To odpowiednik izolacji agentów: agent od kalendarza nie musi
# wiedzieć o agencie od e-maili. Brak zależności = brak konfliktów.
# ============================================================

CLUSTERS = [
    {
        # KLASTER 1: Góra mapy (F1, G1, F2, G2)
        # Transporter jedzie główną arterią D na północ: A6→D6→D1
        # Zwiadowca idzie na wschód do bloków B3
        "name": "Cluster 1 (góra)",
        "transport_target": "D1",   # koniec drogi na północ
        "passengers": 1,            # 1 zwiadowca wystarczy na 4 pola
        "scout_routes": [
            # Scout 0: po dismount ląduje na E1
            # Trasa "zygzak" minimalizuje liczbę kroków:
            # E1 → F1(inspect) → G1(inspect) → G2(inspect) → F2(inspect)
            # 4 ruchy × 7 AP + 4 inspekcje × 1 AP = 32 AP
            [
                ("move", "F1"), ("inspect",),
                ("move", "G1"), ("inspect",),
                ("move", "G2"), ("inspect",),
                ("move", "F2"), ("inspect",),
            ],
        ],
    },
    {
        # KLASTER 2: Dół-lewo (A10, B10, C10, A11, B11, C11)
        # Transporter jedzie D7→D8→D9→C9→B9 (po drogach)
        # 2 zwiadowców rozdziela 6 pól (4+2) dla efektywności
        "name": "Cluster 2 (dół-lewo)",
        "transport_target": "B9",   # najbliższe pole drogowe
        "passengers": 2,            # 2 zwiadowców na 6 pól
        "scout_routes": [
            # Scout 0: obsługuje zachodnią kolumnę (A-B)
            # B10(i) → A10(i) → A11(i) → B11(i)
            # 4 ruchy × 7 + 4 inspekcje × 1 = 32 AP
            [
                ("move", "B10"), ("inspect",),
                ("move", "A10"), ("inspect",),
                ("move", "A11"), ("inspect",),
                ("move", "B11"), ("inspect",),
            ],
            # Scout 1: obsługuje kolumnę C (bliżej transportera)
            # C10(i) → C11(i)
            # 2 ruchy × 7 + 2 inspekcje × 1 = 16 AP (scout startuje blisko)
            [
                ("move", "C10"), ("inspect",),
                ("move", "C11"), ("inspect",),
            ],
        ],
    },
    {
        # KLASTER 3: Dół-prawo (H10, I10, H11, I11)
        # Transporter jedzie wzdłuż row 9 do I9
        "name": "Cluster 3 (dół-prawo)",
        "transport_target": "I9",   # najbliższe pole drogowe
        "passengers": 1,            # 1 zwiadowca na 4 pola
        "scout_routes": [
            # Scout 0: po dismount ląduje na I8 (obok transportera)
            # I10(i) → H10(i) → H11(i) → I11(i)
            # 4 ruchy × 7 + 4 inspekcje × 1 = 32 AP
            [
                ("move", "I10"), ("inspect",),
                ("move", "H10"), ("inspect",),
                ("move", "H11"), ("inspect",),
                ("move", "I11"), ("inspect",),
            ],
        ],
    },
]


# ============================================================
# KLASYFIKACJA LOGÓW PRZEZ LLM
# ============================================================
# Dlaczego LLM zamiast prostych słów kluczowych?
#
# Problem: logi inspekcji są generowane losowo w naturalnym języku.
# Negatywne wyniki mogą brzmieć jak pozytywne (pułapki językowe):
#   - "Znaleziono jedynie metalową skrzynkę" → "znaleziono" ale NEGATYW
#   - "Nie odnaleziono człowieka" → "odnaleziono" ale NEGATYW (negacja!)
#   - "Brak kontaktu z osobą" → "osobą" ale NEGATYW
#
# Pozytywne wyniki mają różne formy:
#   - "Mamy go w zasięgu. To mężczyzna około 30 lat..."
#   - "Osoba jest na miejscu. Mężczyzna około 30 lat..."
#   - "Jest kontakt z celem. Mężczyzna około 30 lat..."
#
# Heurystyki słów kluczowych (keyword matching) ZAWODZĄ tu,
# bo nie rozumieją kontekstu, negacji ani semantyki.
# LLM rozumie znaczenie zdań i poprawnie klasyfikuje.
#
# To ilustruje kluczową lekcję: AI jest niezastąpiona w zadaniach
# wymagających rozumienia naturalnego języka — klasyfikacja
# zgłoszeń, analiza feedbacku, routing wiadomości.
# Prosty if/else z regexem nie wystarczy.
# ============================================================

def classify_logs_with_llm(logs: list) -> str:
    """Używa LLM do znalezienia pola, gdzie zwiadowca znalazł człowieka.

    Przygotowuje logi jako tekst i prosi LLM o wskazanie JEDNEGO pola
    z pozytywnym wynikiem. Temperatura 0 dla deterministyczności.

    Returns:
        Koordynat pola (np. "H10") lub pusty string.
    """
    logs_text = "\n".join(
        f"[{log.get('field', '?')}]: {log.get('msg', '')}"
        for log in logs
    )

    llm = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

    # System prompt jest precyzyjny — ostrzega przed pułapkami.
    # To odpowiednik "prompt engineeringu" w agentach produkcyjnych:
    # im lepszy prompt, tym mniej błędów klasyfikacji.
    response = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": (
                "Jesteś klasyfikatorem logów zwiadowczych. "
                "Przeanalizuj logi i wskaż DOKŁADNIE JEDNO pole (np. H10), "
                "w którym zwiadowca ZNALAZŁ ŻYWEGO CZŁOWIEKA. "
                "Uwaga na pułapki językowe: "
                "'Nie odnaleziono' = NEGATYW (negacja!), "
                "'Znaleziono jedynie skrzynkę' = NEGATYW (znaleziono przedmiot, nie osobę), "
                "'Brak kontaktu z osobą' = NEGATYW. "
                "Szukasz logu gdzie jest JEDNOZNACZNE POTWIERDZENIE obecności żywej osoby. "
                "Odpowiedz TYLKO koordynatem pola, np.: H10"
            )},
            {"role": "user", "content": logs_text}
        ],
        max_tokens=10,       # wystarczy na "H10"
        temperature=0,       # deterministyczna odpowiedź
    )
    return response.choices[0].message.content.strip()


# ============================================================
# GŁÓWNA PĘTLA AGENTA
# ============================================================
# Architektura: Plan → Execute → Observe → React
#
# To jest pełny cykl agentowy z lekcji:
#   PLAN:    Analiza mapy, identyfikacja celów, planowanie tras
#   EXECUTE: Tworzenie jednostek, ruch, inspekcje
#   OBSERVE: Odczyt logów, analiza wyników przez LLM
#   REACT:   Wezwanie helikoptera lub diagnostyka
#
# Analogia z lekcji o kontekstowej współpracy:
#   - Agent podłączony do CRM: czyta dane klienta (plan),
#     tworzy ofertę (execute), sprawdza feedback (observe),
#     modyfikuje ofertę (react)
#   - Agent monitorujący newslettery: pobiera artykuły (plan),
#     klasyfikuje ważność (execute+LLM), wysyła alert (react)
# ============================================================

def main():
    print("=" * 60)
    print("MISJA RATUNKOWA — DOMATOWO")
    print("=" * 60)

    # -------------------------------------------------------
    # FAZA 0: Reset stanu gry
    # -------------------------------------------------------
    # Każde uruchomienie skryptu zaczyna od czystego stanu.
    # Pozycja partyzanta jest losowana na nowo przy resecie.
    # To gwarantuje powtarzalność — jak reset bazy testowej
    # przed uruchomieniem testów integracyjnych.
    print("\n[FAZA 0] Reset stanu gry...")
    send({"action": "reset"})

    # -------------------------------------------------------
    # FAZA 1: Rozpoznanie terenu (Context Gathering)
    # -------------------------------------------------------
    # Pobieramy mapę i identyfikujemy cele (pola B3).
    # BEZ TEJ FAZY agent działałby na ślepo — to jak agent
    # próbujący zarządzać kalendarzem bez odczytania zdarzeń.
    # "Context Engineering" z wcześniejszych lekcji w praktyce.
    print("\n[FAZA 1] Rozpoznanie terenu...")
    map_data = send({"action": "getMap"})
    grid = map_data["map"]["grid"]
    b3_tiles = find_b3_tiles(grid)
    print(f"  Zidentyfikowano {len(b3_tiles)} pól B3: {b3_tiles}")

    # -------------------------------------------------------
    # FAZA 2: Tworzenie jednostek (Resource Allocation)
    # -------------------------------------------------------
    # Tworzymy 3 transportery z odpowiednią liczbą zwiadowców.
    # Spawn automatycznie przydziela pozycje A6→D6 (w kolejności).
    #
    # TRADE-OFF koszt vs pokrycie:
    #   - 1 zwiadowca/transporter = tańsze, ale wolniejsze
    #   - Klaster 2 ma 6 pól = potrzebuje 2 zwiadowców
    #   - Klastry 1 i 3 po 4 pola = 1 zwiadowca wystarczy
    #
    # Łączny koszt tworzenia: 10 + 15 + 10 = 35 AP
    print("\n[FAZA 2] Tworzenie jednostek...")
    transporters = []
    all_scouts = []  # zbieramy ID zwiadowców po dismount

    for cluster in CLUSTERS:
        result = send({
            "action": "create",
            "type": "transporter",
            "passengers": cluster["passengers"]
        })
        if "object" in result:
            transporters.append(result["object"])
            print(f"  Transporter {result['object'][:8]}... "
                  f"z {cluster['passengers']} zwiadowcami")
            print(f"    AP left: {result.get('action_points_left', '?')}")

    # Pobierz aktualny stan — potrzebujemy ID wszystkich jednostek
    units = get_objects()
    print(f"  Jednostki na mapie: {len(units)}")
    for h, u in units.items():
        print(f"    {u.get('typ', '?'):>12} {h[:8]}... @ {u.get('position', '?')}")

    # Sortujemy transportery po pozycji spawn (A6, B6, C6)
    # aby przypisać je do klastrów w odpowiedniej kolejności
    transporter_list = sorted(
        [u for u in units.values() if u.get("typ") == "transporter"],
        key=lambda u: u.get("position", "")
    )

    # -------------------------------------------------------
    # FAZA 3-5: Ruch, wysadzanie, inspekcja (Execute)
    # -------------------------------------------------------
    # Dla każdego klastra:
    #   3. Transporter jedzie do punktu docelowego (po drogach)
    #   4. Zwiadowcy wysiadają (dismount = 0 AP!)
    #   5. Zwiadowcy chodzą po B3 i inspekcjonują
    #
    # KLUCZOWA OPTYMALIZACJA:
    #   Transportery = "autobusy" — tanio (1 AP/pole) dowożą
    #   drogich zwiadowców (7 AP/pole) jak najbliżej celów.
    #   Bez transporterów cały budżet zjadłby ruch pieszych.
    #
    #   Lekcja mówi o "batch processing" — to ten sam pomysł:
    #   zamiast osobno przetwarzać każdy e-mail, zbiorczo
    #   pobieramy, klasyfikujemy, rozsyłamy.
    print("\n[FAZA 3-5] Ruch, wysadzanie, inspekcja...")

    for i, cluster in enumerate(CLUSTERS):
        if i >= len(transporter_list):
            print(f"  UWAGA: Brak transportera dla {cluster['name']}")
            continue

        t_hash = transporter_list[i]["id"]
        target = cluster["transport_target"]
        print(f"\n  >>> {cluster['name']}: transporter {t_hash[:8]}... → {target}")

        # Ruch transportera — API samo wyznacza najkrótszą trasę
        # po drogach (pathfinding wbudowany w serwer).
        # Agent nie musi znać algorytmu Dijkstry — wystarczy,
        # że zna cel. To jak wywołanie API Google Maps.
        send({"action": "move", "object": t_hash, "where": target})

        # Wysadzenie zwiadowców — dismount jest za darmo (0 AP).
        # Zwiadowcy pojawiają się na sąsiednich wolnych polach.
        # Agent wie, gdzie wylądowali, z odpowiedzi API.
        print(f"  Wysadzanie {cluster['passengers']} zwiadowców...")
        dismount_result = send({
            "action": "dismount",
            "object": t_hash,
            "passengers": cluster["passengers"]
        })

        # Pobieramy ID i pozycje zwiadowców z odpowiedzi dismount.
        # API zwraca: { "spawned": [{"scout": "id", "where": "E1"}, ...] }
        # To jest REAKCJA na dane — agent nie zakłada, gdzie wylądują,
        # lecz odczytuje faktyczne pozycje z odpowiedzi.
        scouts_here = []
        if "spawned" in dismount_result:
            for sp in dismount_result["spawned"]:
                scouts_here.append({
                    "id": sp["scout"],
                    "position": sp["where"]
                })
        else:
            # Fallback: pobierz ze stanu gry (gdyby API zmieniło format)
            units = get_objects()
            scouts_here = [
                u for u in units.values()
                if u.get("typ") == "scout"
                and u["id"] not in all_scouts
            ]
            scouts_here.sort(key=lambda u: u.get("position", ""))

        print(f"  Dostępni zwiadowcy: {len(scouts_here)}")
        for s in scouts_here:
            print(f"    scout {s['id'][:8]}... @ {s.get('position', '?')}")

        # Każdy zwiadowca realizuje zaplanowaną trasę:
        # sekwencja ruchów (move) i inspekcji (inspect).
        # Trasy są zaprojektowane jako "zygzaki" minimalizujące
        # liczbę kroków — odpowiednik optymalizacji workflow.
        for route_idx, route in enumerate(cluster["scout_routes"]):
            if route_idx >= len(scouts_here):
                print(f"  UWAGA: Brak zwiadowcy dla trasy {route_idx}")
                continue

            scout_hash = scouts_here[route_idx]["id"]
            all_scouts.append(scout_hash)
            print(f"\n  Zwiadowca {scout_hash[:8]}... realizuje trasę {route_idx}:")

            for step in route:
                if step[0] == "move":
                    # Ruch zwiadowcy — 7 AP za pole, drogo!
                    # Dlatego trasy są krótkie (max 4-5 kroków)
                    send({"action": "move", "object": scout_hash,
                          "where": step[1]})
                elif step[0] == "inspect":
                    # Inspekcja budynku — 1 AP, generuje log.
                    # Log zawiera opis tego, co zwiadowca znalazł.
                    # Wynik jest zapisywany do systemowego dziennika (getLogs).
                    send({"action": "inspect", "object": scout_hash})

    # -------------------------------------------------------
    # FAZA 6: Analiza logów i ewakuacja (Observe + React)
    # -------------------------------------------------------
    # Po przeszukaniu wszystkich 14 pól B3 analizujemy logi.
    # Tu wchodzi LLM — klasyfikuje, który log opisuje
    # ZNALEZIENIE ŻYWEGO CZŁOWIEKA.
    #
    # DLACZEGO LLM, NIE REGEX?
    # Próbowaliśmy dwóch podejść heurystycznych:
    #   1. Wykluczanie negatywnych fraz ("Brak celu", "Nic tu nie ma")
    #      → FAIL: zbyt wiele wariantów negatywnych, nie da się
    #        przewidzieć wszystkich ("Pomieszczenie martwe", "Pusto" itd.)
    #   2. Szukanie pozytywnych fraz ("znaleziono", "osoba")
    #      → FAIL: fałszywe pozytywy ("Znaleziono jedynie skrzynkę")
    #        i negacje ("Nie odnaleziono człowieka")
    #
    # LLM rozumie SEMANTYKĘ zdań — wie, że "Nie odnaleziono"
    # to negacja, a "Mamy go w zasięgu" to potwierdzenie.
    # To dokładnie ten scenariusz z lekcji, gdzie AI pomaga
    # w zadaniach wymagających rozumienia otwartych tekstów:
    # analiza ankiet, routing zgłoszeń, klasyfikacja feedbacku.
    print("\n[FAZA 6] Analiza logów inspekcji...")
    logs_data = send({"action": "getLogs"})
    logs = logs_data.get("logs", [])

    print(f"\nWszystkie logi ({len(logs)}):")
    for log in logs:
        msg = log.get("msg", str(log))
        field = log.get("field", "?")
        print(f"   [{field}]: {msg}")

    # Klasyfikacja przez LLM
    print("\n  Klasyfikacja logów przez LLM...")
    found_human_at = classify_logs_with_llm(logs)
    print(f"  LLM wskazał: {found_human_at}")

    # -------------------------------------------------------
    # FAZA 7: Wezwanie helikoptera (Final Action)
    # -------------------------------------------------------
    # Gdy LLM wskaże pole z partyzantem, wzywamy helikopter.
    # API zwraca flagę, jeśli pole jest poprawne.
    #
    # To "last mile" agenta — finalna akcja na podstawie
    # wszystkich zebranych i przeanalizowanych danych.
    # W biznesie to może być: wysłanie e-maila, utworzenie
    # ticketa, zmiana statusu w CRM.
    if found_human_at:
        print(f"\n  *** POTENCJALNY CEL na {found_human_at}! "
              f"Wzywam helikopter... ***")
        result = send({
            "action": "callHelicopter",
            "destination": found_human_at
        })
        print(f"\n{'='*60}")
        print(f"WYNIK: {json.dumps(result, indent=2, ensure_ascii=False)}")
        print(f"{'='*60}")
    else:
        print("\n  Nie znaleziono partyzanta. Sprawdzam wydatki...")
        expenses = send({"action": "expenses"})
        print(json.dumps(expenses, indent=2, ensure_ascii=False))

    # -------------------------------------------------------
    # PODSUMOWANIE — Monitoring skuteczności
    # -------------------------------------------------------
    # Sprawdzamy, ile AP zużyliśmy. To odpowiednik monitoringu
    # kosztów w produkcji — ile tokenów LLM zużyliśmy?
    # Ile requestów API? Czy mieścimy się w budżecie?
    #
    # Lekcja mówi o agentach obserwujących własną skuteczność:
    # "jeśli newsletter przestajemy czytać, jaki sens go wysyłać?"
    # Tu analogicznie: jeśli zużywamy 250/300 AP, może warto
    # zoptymalizować trasy.
    print("\n[PODSUMOWANIE] Wydatki AP:")
    expenses = send({"action": "expenses"})
    if "expenses" in expenses:
        total = sum(e.get("cost", 0) for e in expenses["expenses"])
        print(f"  Łączny koszt: {total} / 300 AP")
        print(f"  Wykorzystanie budżetu: {total/300*100:.0f}%")


if __name__ == "__main__":
    main()
