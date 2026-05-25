import os
from dotenv import load_dotenv
load_dotenv()

"""
S04E02 – windpower – Konfiguracja turbiny wiatrowej w 40 sekund
================================================================

KONTEKST LEKCJI (S04E02 — Aktywna współpraca z AI):
Lekcja omawia praktyczne aspekty pracy z agentami AI, od wyboru interfejsu
(CLI, MCP, komunikatory, dedykowane rozwiązania) po personalizację interakcji
i projektowanie meta-promptów.

KLUCZOWE KONCEPCJE Z LEKCJI ZASTOSOWANE W TYM ZADANIU:

1. ASYNCHRONICZNOŚĆ I RÓWNOLEGŁOŚĆ (sekcja: "Synchroniczna i asynchroniczna współpraca")
   ─────────────────────────────────────────────────────────────────────────
   Lekcja mówi o dwóch trybach pracy z AI: synchronicznym (użytkownik czeka
   na odpowiedź) i asynchronicznym (agenci pracują w tle). To zadanie wymusza
   podejście ASYNCHRONICZNE — API kolejkuje żądania, a my pollujemy wyniki.

   "Liniowe wykonywanie wszystkich akcji nie umożliwi Ci ukończenia zadania."
   → Musimy kolejkować wiele żądań RÓWNOLEGLE i zbierać wyniki w miarę
   jak się pojawiają. To dokładnie tak jak agenci w tle z lekcji.

2. PIPELINE Z OGRANICZENIEM CZASOWYM (sekcja: "Weryfikowanie założeń")
   ─────────────────────────────────────────────────────────────────────────
   40-sekundowe okno serwisowe wymusza precyzyjne zaplanowanie kolejności
   operacji. Nie ma miejsca na retry ani eksplorację — plan musi być gotowy
   PRZED startem timera. To nawiązanie do lekcji o prototypowaniu:
   "szybkie iteracje" i "weryfikowanie tez" przed właściwym wdrożeniem.

3. BALANS KOD vs AI (sekcja: "Mapowanie procesów")
   ─────────────────────────────────────────────────────────────────────────
   Lekcja pokazuje diagram z proporcjami kod/AI w architekturze. Tu 100%
   to kod — LLM nie zmieściłby się w 40s limicie. To ilustruje ważną zasadę:
   "nie stosować AI tam, gdzie korzyści będą mniejsze". Analiza pogody to
   proste reguły (if wind > 14 → storm), nie potrzebują modelu językowego.

4. MIKRO-AKCJE I JEDNORAZOWE ZADANIA (sekcja: "Jednorazowe zadania")
   ─────────────────────────────────────────────────────────────────────────
   Lekcja mówi o prostych, szybkich akcjach przypisanych do skrótów klawiszowych.
   Ten skrypt jest właśnie taką "mikro-akcją" — jednorazowe uruchomienie,
   konkretny cel, bez rozbudowanego interfejsu. Nie każde wdrożenie AI
   wymaga czatbota czy systemu wieloagentowego.

ARCHITEKTURA ROZWIĄZANIA (pipeline z ograniczeniem 40s):
┌──────────────────────────────────────────────────────────────────┐
│  Faza 0 (poza timerem): Dokumentacja turbiny                    │
│  ↓                                                               │
│  Faza 1: START sesji → kolejkuj weather + powerplant + turbine   │
│  ↓                                                               │
│  Faza 2: POLL wyników (weather jest najwolniejszy ~20s)          │
│  ↓                                                               │
│  Faza 3: ANALIZA — wichury, okna produkcji (czysta logika kodu) │
│  ↓                                                               │
│  Faza 4: Kolejkuj unlockCodeGenerator dla każdego config         │
│  ↓                                                               │
│  Faza 5: BATCH CONFIG — wyślij wszystkie konfiguracje naraz      │
│  ↓                                                               │
│  Faza 6: turbinecheck (wymagany przed done)                      │
│  ↓                                                               │
│  Faza 7: DONE → flaga                                           │
│                                                                   │
│  Budżet czasu: ~24s (dane) + ~4s (kody) + ~12s (config+check)   │
└──────────────────────────────────────────────────────────────────┘

WNIOSKI Z DEBUGOWANIA:
- unlockCodeGenerator zwraca kody w LOSOWEJ kolejności → trzeba mapować
  po signedParams (zawiera startDate + startHour), nie po kolejności wysyłki.
- powerDeficitKw jest ZMIENNE w czasie (raz 3, raz 4-5 kW) → trzeba
  reagować na aktualne dane, nie zakodowane na sztywno wartości.
- Prognoza pogody też się zmienia między sesjami → wiatr 4.9 m/s w jednej
  sesji, 6.6 m/s w innej. Algorytm musi być odporny na zmienność.
"""

import json
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# ══════════════════════════════════════════════════════════════════════════
# KONFIGURACJA
# ══════════════════════════════════════════════════════════════════════════
# Parametry turbiny z dokumentacji API (action: "get", param: "documentation").
# Zapisane tu, żeby nie tracić czasu na ponowne pobieranie w 40s oknie.
#
# Lekcja: "Mapowanie procesów" — zanim zaczniemy budować, musimy zrozumieć
# dziedzinę. Dokumentacja turbiny definiuje kluczowe parametry:
# - ratedPowerKw: 14 kW (moc nominalna)
# - cutoffWindMs: 14 m/s (powyżej → uszkodzenie łopat)
# - minOperationalWindMs: 4 m/s (poniżej → brak generacji)
#
# Yield (procent mocy nominalnej) zależy od DWÓCH zmiennych:
# 1. Prędkość wiatru → windPowerYieldPercent (krzywa mocy)
# 2. Kąt nachylenia łopat (pitch) → pitchAngleYieldPercent
# Rzeczywista moc = ratedPower × windYield × pitchYield

HUB_API_KEY = os.environ["HUB_API_KEY"]
HUB_URL = "https://hub.ag3nts.org/verify"

CUTOFF_WIND = 14  # m/s — powyżej tego wiatru łopaty się łamią
MIN_WIND = 4      # m/s — poniżej turbina nie generuje prądu
RATED_POWER = 14  # kW — moc nominalna turbiny


# ══════════════════════════════════════════════════════════════════════════
# WARSTWA KOMUNIKACJI Z API
# ══════════════════════════════════════════════════════════════════════════
# API windpower jest ASYNCHRONICZNE (queue + poll). To kluczowy wzorzec
# z lekcji o asynchronicznej współpracy z AI:
#
#   1. Wysyłasz żądanie → API odpowiada "Task has been queued"
#   2. Wielokrotnie pytasz "getResult" → API zwraca wynik gdy gotowy
#   3. Wyniki przychodzą w LOSOWEJ kolejności
#
# To dokładnie jak agenci działający w tle z lekcji — zlecasz zadania,
# a potem zbierasz wyniki. Różnica: tu mamy twardy limit 40 sekund.

def api(answer: dict) -> dict:
    """Wysyła request do API hub.

    Każde wywołanie to POST z apikey + task + answer.
    timeout=10 zapobiega zawieszeniu na pojedynczym requeście.
    """
    payload = {"apikey": HUB_API_KEY, "task": "windpower", "answer": answer}
    resp = requests.post(HUB_URL, json=payload, timeout=10)
    return resp.json()


def poll_results(expected: int, timeout: float = 30, interval: float = 0.4) -> list:
    """Polluje getResult aż zbierze expected wyników lub minie timeout.

    WZORZEC: Polling loop z adaptacyjnym oczekiwaniem.
    ──────────────────────────────────────────────────
    API nie oferuje webhooków ani SSE (Server-Sent Events), więc jedyny
    sposób na odebranie wyników to aktywne odpytywanie (polling).

    interval=0.3-0.4s to kompromis:
    - Za krótki (0.1s) → niepotrzebne obciążenie API, ryzyko rate-limitu
    - Za długi (1s) → marnujemy cenny czas z 40s budżetu

    Każdy odebrany wynik jest USUWANY z kolejki (pobrać można tylko raz).
    Dlatego zbieramy je w listę i identyfikujemy po sourceFunction.

    code=12 → wynik gotowy
    code=11 → jeszcze nie ma wyników
    code=-805 → timeout sesji (40s minęło)
    """
    results = []
    start = time.time()
    while len(results) < expected and (time.time() - start) < timeout:
        r = api({"action": "getResult"})
        if r.get("code") == 12:
            results.append(r)
            print(f"  [✓] Otrzymano: {r.get('sourceFunction')} ({len(results)}/{expected})")
        elif r.get("code") == -805:
            print("  [!] Timeout sesji!")
            break
        time.sleep(interval)
    return results


# ══════════════════════════════════════════════════════════════════════════
# LOGIKA DZIEDZINOWA — analiza pogody i mocy turbiny
# ══════════════════════════════════════════════════════════════════════════
# To część, gdzie AI NIE jest potrzebne.
# Lekcja (sekcja "Mapowanie procesów"): "decyzja, aby NIE stosować AI,
# na przykład tam, gdzie korzyści będą mniejsze".
#
# Analiza pogody to proste reguły:
# - wind > 14 m/s → wichura → tryb ochronny (pitch=90°, idle)
# - wind >= 4 m/s i moc >= deficyt → okno produkcji (pitch=0°, production)
# Nie potrzeba LLM, żeby to obliczyć — wystarczy pętla i porównania.

def estimate_power(wind_ms: float, pitch: int) -> float:
    """Szacuje moc [kW] na podstawie wiatru i kąta łopat.

    Wzór: moc = RATED_POWER × wind_yield × pitch_yield

    wind_yield pochodzi z dokumentacji turbiny (windPowerYieldPercent):
      4 m/s → 10-15%   (bierzemy średnią 12.5%)
      6 m/s → 30-40%   (średnia 35%)
      8 m/s → 60-70%   (średnia 65%)
      10 m/s → 90-100% (średnia 95%)
      12-14 m/s → 100%
      14+ m/s → damage (0% — turbina musi być wyłączona)

    pitch_yield (pitchAngleYieldPercent):
      0° → 100%  (łopaty prostopadle do wiatru — max chwyt)
      45° → 65%  (częściowy chwyt)
      90° → 0%   (łopaty równolegle do wiatru — zero oporu)

    Kąt 90° służy do OCHRONY przed wichurą — łopaty nie chwytają wiatru.
    """
    if wind_ms < 4:
        wind_yield = 0
    elif wind_ms < 5:
        wind_yield = 0.125
    elif wind_ms < 7:
        wind_yield = 0.35
    elif wind_ms < 9:
        wind_yield = 0.65
    elif wind_ms < 11:
        wind_yield = 0.95
    elif wind_ms <= 14:
        wind_yield = 1.0
    else:
        wind_yield = 0

    pitch_yield = {0: 1.0, 45: 0.65, 90: 0.0}
    p_yield = pitch_yield.get(pitch, 0)

    return RATED_POWER * wind_yield * p_yield


def analyze_weather(forecast: list, deficit_kw: float) -> dict:
    """Analizuje 7-dniową prognozę pogody i generuje konfiguracje turbiny.

    Zwraca dict: timestamp → {pitchAngle, turbineMode, windMs}

    STRATEGIA (z instrukcji fabularnej):
    ────────────────────────────────────
    1. WICHURY: każdy punkt z wiatrem > 14 m/s → pitch=90°, turbineMode=idle
       "Wystarczy ustawić łopaty wirnika tak, aby nie stawiały oporu wiatrowi"
       Po wichurze (1h) łopaty wracają do domyślnego ustawienia — ale prognoza
       jest co 2h, więc każda wichura to osobny punkt konfiguracji.

    2. PRODUKCJA: pierwszy moment, gdy moc >= deficyt elektrowni
       "Musisz znaleźć PIERWSZE możliwe okno pogodowe" (zależy nam na czasie)
       pitch=0° (max chwyt), turbineMode=production

    Priorytet: bezpieczeństwo (wichury) PRZED produkcją.
    """
    configs = {}

    # ── Krok 1: Identyfikacja wichur ──────────────────────────────────
    for entry in forecast:
        if entry["windMs"] > CUTOFF_WIND:
            ts = entry["timestamp"]
            configs[ts] = {
                "pitchAngle": 90,       # łopaty równolegle do wiatru
                "turbineMode": "idle",  # brak produkcji
                "windMs": entry["windMs"],
            }
            print(f"  [STORM] {ts}: {entry['windMs']} m/s → pitch=90, idle")

    # ── Krok 2: Szukanie pierwszego okna produkcji ────────────────────
    # Iterujemy chronologicznie — bierzemy PIERWSZY wystarczający moment.
    production_ts = None
    for entry in forecast:
        ts = entry["timestamp"]
        wind = entry["windMs"]

        if wind > CUTOFF_WIND or wind < MIN_WIND:
            continue

        power = estimate_power(wind, 0)
        if power >= deficit_kw:
            production_ts = ts
            configs[ts] = {
                "pitchAngle": 0,
                "turbineMode": "production",
                "windMs": wind,
            }
            print(f"  [PROD] {ts}: {wind} m/s → {power:.1f} kW (potrzeba {deficit_kw} kW)")
            break

    # ── Fallback: najlepszy dostępny wiatr ────────────────────────────
    # Jeśli żaden punkt nie daje wystarczająco mocy, bierzemy najlepszy.
    # powerDeficitKw jest zmienne — czasem 3 kW, czasem 5 kW.
    if not production_ts:
        safe = [e for e in forecast if MIN_WIND <= e["windMs"] <= CUTOFF_WIND]
        safe.sort(key=lambda e: e["windMs"], reverse=True)
        print(f"  [!] Nie znaleziono okna z mocą >= {deficit_kw} kW. Top wiatry:")
        for e in safe[:10]:
            p = estimate_power(e["windMs"], 0)
            print(f"      {e['timestamp']}: {e['windMs']} m/s → {p:.1f} kW")
        if safe:
            best = safe[0]
            ts = best["timestamp"]
            configs[ts] = {
                "pitchAngle": 0,
                "turbineMode": "production",
                "windMs": best["windMs"],
            }
            print(f"  [PROD-BEST] {ts}: {best['windMs']} m/s → {estimate_power(best['windMs'], 0):.1f} kW")

    return configs


# ══════════════════════════════════════════════════════════════════════════
# GŁÓWNY PIPELINE — orkiestracja w 40 sekund
# ══════════════════════════════════════════════════════════════════════════
# To serce rozwiązania. Implementuje wzorzec PIPELINE z lekcji:
# "Jedno zdarzenie może zapoczątkować aktywność całego zespołu agentów,
#  którzy wspólnie dążą do osiągnięcia postawionego celu."
#
# Tu "agentami" są równoległe żądania API, a "orkiestratorem" jest
# główna funkcja run() — koordynuje ich pracę w ramach limitu czasu.
#
# KLUCZOWA LEKCJA: ThreadPoolExecutor
# ────────────────────────────────────
# Zadanie explicite mówi: "Liniowe wykonywanie nie umożliwi ukończenia."
# ThreadPoolExecutor pozwala KOLEJKOWAĆ wiele żądań jednocześnie.
# To Python-owy odpowiednik tego, co w lekcji opisane jest jako
# "współpraca wielu agentów" — każdy thread to osobny "agent" HTTP.
#
# Alternatywy: asyncio + aiohttp (niższy narzut), ale ThreadPoolExecutor
# jest prostszy i wystarczający przy 3-4 równoległych requestach.

def run():
    elapsed_start = time.time()

    # ── Faza 0: Dokumentacja — POMINIĘTA ────────────────────────────
    # Wartości znamy z wcześniejszego testu (ratedPowerKw=14, cutoffWindMs=14).
    # Każda sekunda się liczy w 40s oknie, więc pomijamy ten request.
    # Lekcja: "weryfikowanie początkowych założeń przez proste testy"
    # — najpierw testujemy (faza eksploracji), potem optymalizujemy.
    print("=== Faza 0: Dokumentacja (hardcoded) ===")
    print(f"  Rated power: {RATED_POWER} kW, Cutoff wind: {CUTOFF_WIND} m/s")

    # ── Faza 1: START sesji → kolejkowanie danych ─────────────────────
    # Od tego momentu biegnie 40-sekundowy timer!
    # Natychmiast kolejkujemy TRZY żądania danych równolegle.
    #
    # Dlaczego równolegle? Bo każde żądanie jest ASYNCHRONICZNE —
    # API przetwarza je w tle i wynik pojawia się po kilku sekundach.
    # Sekwencyjne kolejkowanie zmarnowałoby ~2-3s na same requesty.
    print("\n=== Faza 1: Start sesji ===")
    start_resp = api({"action": "start"})
    session_start = time.time()
    print(f"  Sesja: {start_resp.get('sessionStart')} (timeout: {start_resp.get('sessionTimeout')}s)")

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [
            pool.submit(api, {"action": "get", "param": "weather"}),
            pool.submit(api, {"action": "get", "param": "powerplantcheck"}),
            pool.submit(api, {"action": "get", "param": "turbinecheck"}),
        ]
        for f in as_completed(futures):
            r = f.result()
            print(f"  Queued: code={r.get('code')}")

    # ── Faza 2: Polling wyników — REAKTYWNY wzorzec ─────────────────
    # Zamiast czekać na WSZYSTKIE 3 wyniki, zaczynamy przetwarzanie
    # natychmiast po otrzymaniu weather (najwolniejszy, ~20-24s).
    # powerplantcheck i turbinecheck są szybsze (~8s).
    #
    # KLUCZOWA OPTYMALIZACJA: Nie czekamy na powerplantcheck.
    # Deficyt mocy to 3-5 kW — zakładamy 5 kW (worst case).
    # Jeśli powerplantcheck zdąży — aktualizujemy. Jeśli nie — fallback.
    #
    # Lekcja: "Integracje — agent może wykonywać kod"
    # Tu agent (nasz skrypt) aktywnie polluje wyniki, reagując na to,
    # co jest dostępne. To wzorzec REACTIVE — nie czekamy pasywnie,
    # ale przetwarzamy dane w miarę ich napływania.
    print("\n=== Faza 2: Pobieranie danych ===")
    data = {}
    poll_start = time.time()
    while "weather" not in data and (time.time() - poll_start) < 26:
        r = api({"action": "getResult"})
        if r.get("code") == 12:
            data[r["sourceFunction"]] = r
            print(f"  [✓] Otrzymano: {r.get('sourceFunction')} ({len(data)}/3)")
        elif r.get("code") == -805:
            print("  [!] Timeout sesji!")
            break
        time.sleep(0.15)

    if "weather" not in data:
        print("  [!] Brak prognozy pogody — nie mogę kontynuować!")
        return

    # ── Faza 3: Analiza danych ────────────────────────────────────────
    # Czysta logika kodu — ZERO AI. To ilustruje zasadę z lekcji:
    # "decyzja, aby nie stosować AI, na przykład tam, gdzie korzyści
    #  będą mniejsze albo nawet w ogóle ich nie odczujemy."
    #
    # Analiza pogody to:
    # 1. Filtrowanie wichur (wind > 14 m/s)
    # 2. Szukanie pierwszego okna produkcji (wystarczająca moc)
    # Model językowy byłby tu przerostem formy nad treścią.
    print(f"\n=== Faza 3: Analiza (elapsed: {time.time()-session_start:.1f}s) ===")

    # powerDeficitKw jest ZMIENNE — "Elektrownia raportuje jakie ma
    # obecnie niedobory prądu - jest to zmienne w czasie".
    # Fallback 5 kW jeśli powerplantcheck nie zdążył wrócić.
    deficit_str = data.get("powerplantcheck", {}).get("powerDeficitKw", "5")
    if isinstance(deficit_str, str) and "-" in deficit_str:
        deficit_kw = float(deficit_str.split("-")[1])  # górny zakres, bezpieczniej
    else:
        deficit_kw = float(deficit_str)
    print(f"  Deficyt mocy: {deficit_kw} kW")

    forecast = data.get("weather", {}).get("forecast", [])
    print(f"  Prognoza: {len(forecast)} punktów")

    configs = analyze_weather(forecast, deficit_kw)

    if not configs:
        print("  [!] Brak konfiguracji do wysłania!")
        return

    # ── Faza 4: Generowanie kodów unlock (RÓWNOLEGLE) ────────────────
    # Każda konfiguracja wymaga cyfrowego podpisu (unlockCode).
    # Generator kodów też jest ASYNCHRONICZNY — kolejkujemy i pollujemy.
    #
    # WAŻNA LEKCJA O MAPOWANIU WYNIKÓW:
    # Wyniki unlockCodeGenerator przychodzą w LOSOWEJ kolejności.
    # Mapujemy je po signedParams.startDate + signedParams.startHour.
    # Pierwsza wersja próbowała czytać startDate z top-level response
    # (nie istniało) → wszystkie kody miały klucz "None None" → fail.
    #
    # Lekcja: "oczekiwania vs rzeczywistość" — API nie zawsze działa
    # tak jak zakładamy. Trzeba sprawdzać rzeczywisty format odpowiedzi.
    print(f"\n=== Faza 4: Unlock codes ({len(configs)} kodów, elapsed: {time.time()-session_start:.1f}s) ===")

    code_map = {}  # "YYYY-MM-DD HH:MM:SS" → unlock code hash
    with ThreadPoolExecutor(max_workers=len(configs)) as pool:
        futures = {}
        for ts, cfg in configs.items():
            date_part, hour_part = ts.split(" ")
            f = pool.submit(api, {
                "action": "unlockCodeGenerator",
                "startDate": date_part,
                "startHour": hour_part,
                "windMs": cfg["windMs"],
                "pitchAngle": cfg["pitchAngle"],
            })
            futures[f] = ts
        for f in as_completed(futures):
            r = f.result()
            print(f"  Queued unlock for {futures[f]}: code={r.get('code')}")

    # Polluj kody unlock — agresywny interwał, bo czas ucieka
    code_results = poll_results(expected=len(configs), timeout=12, interval=0.15)
    for r in code_results:
        if "unlockCode" in r:
            # Mapowanie po signedParams — jedyny pewny sposób identyfikacji
            sp = r.get("signedParams", {})
            ts_key = f"{sp.get('startDate')} {sp.get('startHour')}"
            code_map[ts_key] = r["unlockCode"]
            print(f"  Code for {ts_key}: {r['unlockCode'][:20]}...")

    # ── Faza 5: Wysyłka BATCH CONFIG ─────────────────────────────────
    # Zamiast wysyłać konfiguracje pojedynczo, używamy batch endpoint.
    # Format: configs = {"YYYY-MM-DD HH:MM:SS": {pitchAngle, turbineMode, unlockCode}}
    #
    # To optymalizacja czasowa — jeden request zamiast N.
    # Lekcja: "jednorazowe zadania i pojedyncze akcje" — czasem jeden
    # dobrze zaprojektowany request zastępuje wiele mniejszych.
    print(f"\n=== Faza 5: Config (elapsed: {time.time()-session_start:.1f}s) ===")

    batch_configs = {}
    for ts, cfg in configs.items():
        unlock = code_map.get(ts, "")
        if not unlock:
            print(f"  [!] Brak unlock code dla {ts}!")
            continue
        batch_configs[ts] = {
            "pitchAngle": cfg["pitchAngle"],
            "turbineMode": cfg["turbineMode"],
            "unlockCode": unlock,
        }

    if batch_configs:
        config_resp = api({"action": "config", "configs": batch_configs})
        print(f"  Config response: code={config_resp.get('code')} msg={config_resp.get('message')}")
    else:
        print("  [!] Brak konfiguracji z kodami!")
        return

    # ── Faza 6: Turbine check (WYMAGANY przed done) ──────────────────
    # Dokumentacja API: "Run turbinecheck before done."
    # Kolejny asynchroniczny request — kolejkujemy i czekamy na wynik.
    print(f"\n=== Faza 6: Turbine check (elapsed: {time.time()-session_start:.1f}s) ===")
    api({"action": "get", "param": "turbinecheck"})
    tc_results = poll_results(expected=1, timeout=10, interval=0.15)
    for r in tc_results:
        print(f"  Turbine: {r.get('status')} battery={r.get('battery')}")

    # ── Faza 7: Done → flaga ─────────────────────────────────────────
    # Ostatnia akcja — API weryfikuje konfigurację i zwraca flagę.
    # Jeśli wszystko OK i zmieściliśmy się w 40s → sukces.
    print(f"\n=== Faza 7: Done (elapsed: {time.time()-session_start:.1f}s) ===")
    done_resp = api({"action": "done"})
    print(f"  Response: {json.dumps(done_resp, ensure_ascii=False)}")

    total = time.time() - session_start
    print(f"\n=== TOTAL TIME: {total:.1f}s ===")


if __name__ == "__main__":
    run()
