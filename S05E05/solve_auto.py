import os
from dotenv import load_dotenv
load_dotenv()

"""
S05E05 — CHRONOS-P1 — Pełny automat (bez interakcji z operatorem)
===================================================================
Skrypt automatycznie konfiguruje ZARÓWNO API (day/month/year/syncRatio/stabilization)
JAK I parametry UI (PT-A/PT-B/PWR/mode) przez odkryty endpoint /timetravel_backend.
Skok wykonywany jest przez {action: "timeTravel"} na /verify.

ODKRYCIE: Interfejs webowy komunikuje się z backendem przez POST /timetravel_backend,
wysyłając pola takie jak {PTA: true, PTB: false, PWR: 91, mode: "standby"}.
To pozwala na pełną automatyzację bez Playwright/Selenium.

LEKCJA S05E05 — koncepcje zastosowane:
1. Reverse engineering UI — analiza kodu źródłowego strony preview
   ujawniła endpointy i formaty danych (sendUpdate, pollConfig)
2. Dual API — podział na /verify (logika gry) i /timetravel_backend (UI)
   odzwierciedla architekturę client/server z lekcji
3. Parsowanie NLP — hinty stabilization w naturalnym języku
4. Polling + czekanie — internalMode zmienia się automatycznie
5. Walidacja — flux density jako health check
"""

import requests
import json
import time
import re
from openai import OpenAI

# ============================================================
# KONFIGURACJA
# ============================================================
API_KEY = os.environ["HUB_API_KEY"]
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
BASE = "https://hub.ag3nts.org"

# LLM do parsowania hintów stabilization (alternatywa dla ręcznego parsera)
llm = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

# Tryb parsowania: "llm" lub "regex"
PARSE_MODE = "llm"

# ============================================================
# KOMUNIKACJA — dwa endpointy
# ============================================================

def api(action, param=None, value=None):
    """
    Endpoint /verify — logika gry.
    Akcje: help, getConfig, reset, configure, timeTravel
    Configure wymaga trybu standby.
    """
    payload = {"apikey": API_KEY, "task": "timetravel", "answer": {"action": action}}
    if param is not None:
        payload["answer"]["param"] = param
        payload["answer"]["value"] = value
    for attempt in range(5):
        try:
            r = requests.post(f"{BASE}/verify", json=payload, timeout=15)
            return r.json()
        except:
            time.sleep(2)
    return None

def ui_update(fields):
    """
    Endpoint /timetravel_backend — kontrola interfejsu webowego.
    Obsługuje: PTA (bool), PTB (bool), PWR (int), mode ("standby"/"active")
    Odkryty przez analizę kodu JS strony preview.
    """
    fields["apikey"] = API_KEY
    for attempt in range(5):
        try:
            r = requests.post(f"{BASE}/timetravel_backend", json=fields, timeout=15)
            return r.json()
        except:
            time.sleep(2)
    return None

# ============================================================
# OBLICZENIA
# ============================================================

def calc_sync_ratio(day, month, year):
    """syncRatio = ((day*8 + month*12 + year*7) % 101) / 100"""
    return round((day * 8 + month * 12 + year * 7) % 101 / 100, 2)

def get_required_mode(year):
    """internalMode wymagany dla roku docelowego."""
    if year < 2000: return 1
    if year <= 2150: return 2
    if year <= 2300: return 3
    return 4

def polish_word_to_number(text):
    """
    Konwertuje polskie liczebniki na liczby.
    Obsługuje: "dziewięćset", "siedemset jedenaście", "pięćset dwadzieścia pięć" itp.
    Hinty z API używają słów zamiast cyfr — kluczowy element parsowania NLP.
    """
    ones = {
        "zero": 0, "jeden": 1, "dwa": 2, "trzy": 3, "cztery": 4, "pięć": 5,
        "sześć": 6, "siedem": 7, "osiem": 8, "dziewięć": 9,
        "jedenaście": 11, "dwanaście": 12, "trzynaście": 13, "czternaście": 14,
        "piętnaście": 15, "szesnaście": 16, "siedemnaście": 17, "osiemnaście": 18,
        "dziewiętnaście": 19, "dziesięć": 10
    }
    tens = {
        "dwadzieścia": 20, "trzydzieści": 30, "czterdzieści": 40, "pięćdziesiąt": 50,
        "sześćdziesiąt": 60, "siedemdziesiąt": 70, "osiemdziesiąt": 80, "dziewięćdziesiąt": 90
    }
    hundreds = {
        "sto": 100, "dwieście": 200, "trzysta": 300, "czterysta": 400, "pięćset": 500,
        "sześćset": 600, "siedemset": 700, "osiemset": 800, "dziewięćset": 900
    }

    words = text.lower().split()
    result = 0
    for w in words:
        if w in hundreds: result += hundreds[w]
        elif w in tens: result += tens[w]
        elif w in ones: result += ones[w]
    return result if result > 0 else None

def extract_numbers_from_hint(hint):
    """
    Wyciąga liczby z hintów — zarówno cyfry jak i polskie liczebniki.
    Zwraca listę znalezionych liczb.
    """
    numbers = []
    # Szukaj cyfr
    for m in re.finditer(r'\b(\d+)\b', hint):
        numbers.append((m.start(), int(m.group(1))))

    # Szukaj polskich liczebników — grupy kolejnych słów liczbowych
    polish_num_words = {
        "zero", "jeden", "dwa", "trzy", "cztery", "pięć", "sześć", "siedem", "osiem", "dziewięć",
        "dziesięć", "jedenaście", "dwanaście", "trzynaście", "czternaście", "piętnaście",
        "szesnaście", "siedemnaście", "osiemnaście", "dziewiętnaście",
        "dwadzieścia", "trzydzieści", "czterdzieści", "pięćdziesiąt",
        "sześćdziesiąt", "siedemdziesiąt", "osiemdziesiąt", "dziewięćdziesiąt",
        "sto", "dwieście", "trzysta", "czterysta", "pięćset",
        "sześćset", "siedemset", "osiemset", "dziewięćset"
    }
    words = hint.lower().split()
    i = 0
    while i < len(words):
        # Oczyść słowo z interpunkcji
        clean = re.sub(r'[.,;:!?]', '', words[i])
        if clean in polish_num_words:
            group = [clean]
            pos = hint.lower().find(clean)
            j = i + 1
            while j < len(words):
                c2 = re.sub(r'[.,;:!?]', '', words[j])
                if c2 in polish_num_words:
                    group.append(c2)
                    j += 1
                else:
                    break
            val = polish_word_to_number(" ".join(group))
            if val:
                numbers.append((pos, val))
            i = j
        else:
            i += 1
    # Sortuj po pozycji w tekście
    numbers.sort(key=lambda x: x[0])
    return [n[1] for n in numbers]

def parse_stabilization_llm(hint):
    """
    Parsuje hint stabilization za pomocą LLM.
    Zamiast ręcznego parsera polskich liczebników, LLM sam wyciąga
    wartość z naturalnego języka. To podejście jest bardziej odporne
    na zmiany formatu hintów i nowe wzorce tekstowe.

    Przykład: "sugerują 900 jednostek... obniżenie o 711" → LLM odpowiada "189"
    """
    try:
        resp = llm.chat.completions.create(
            model="anthropic/claude-haiku-4.5",
            messages=[{"role": "user", "content": f"""Extract the final stabilization value from this hint.
The hint describes a base value and an adjustment (increase or decrease).
Calculate the result and respond with ONLY the number.

Hint: "{hint}"

Answer (just the number):"""}],
            max_tokens=10, temperature=0
        )
        val = int(resp.choices[0].message.content.strip())
        print(f"  → stabilization (LLM) = {val}")
        return val
    except Exception as e:
        print(f"  LLM parse error: {e}, falling back to regex")
        return parse_stabilization_regex(hint)

def parse_stabilization_regex(hint):
    """
    Parsuje hint NLP z pola needConfig za pomocą regexów + parsera liczebników.
    Wzorce: "X jednostek... obniżenie o Y" → X-Y
             "X jednostek... zwiększyć o Y" → X+Y
    Obsługuje zarówno cyfry jak i polskie liczebniki.
    """
    if not hint:
        return None
    numbers = extract_numbers_from_hint(hint)
    if len(numbers) < 2:
        return None
    base, adj = numbers[0], numbers[1]
    if any(w in hint.lower() for w in ["zwiększ", "podwyż", "podnieś"]):
        result = base + adj
    else:
        result = base - adj
    print(f"  → stabilization (regex) = {base} {'+'if 'zwiększ' in hint.lower() else '-'} {adj} = {result}")
    return result

def parse_stabilization(hint):
    """Dispatcher — wybiera metodę parsowania na podstawie PARSE_MODE."""
    if not hint:
        return None
    if PARSE_MODE == "llm":
        return parse_stabilization_llm(hint)
    return parse_stabilization_regex(hint)

# Tabela PWR z dokumentacji urządzenia
PWR = {2024: 19, 2026: 28, 2238: 91}

# ============================================================
# PROCEDURA SKOKU — pełna automatyzacja
# ============================================================

def execute_jump(day, month, year, jump_type="jump"):
    """
    Wykonuje kompletny skok w czasie:
    1. Ustaw standby (UI)
    2. Skonfiguruj datę, syncRatio, stabilization (API)
    3. Ustaw PT-A/PT-B, PWR (UI)
    4. Przełącz na active (UI)
    5. Poczekaj na właściwy internalMode
    6. Wykonaj skok (API: timeTravel)
    """
    sync = calc_sync_ratio(day, month, year)
    mode = get_required_mode(year)
    pwr = PWR[year]

    # PT-A/PT-B
    if jump_type == "tunnel":
        pta, ptb = True, True
    elif year > 2026 or (year == 2026 and month > 4) or (year == 2026 and month == 4 and day > 11):
        pta, ptb = False, True   # przyszłość
    else:
        pta, ptb = True, False   # przeszłość

    print(f"\n{'='*60}")
    print(f"  SKOK → {day:02d}.{month:02d}.{year} [{jump_type}]")
    print(f"  sync={sync} pwr={pwr} mode={mode} PTA={pta} PTB={ptb}")
    print(f"{'='*60}")

    # 1. Standby
    ui_update({"mode": "standby"})
    time.sleep(0.5)

    # 2. Konfiguracja API
    for p, v in [("year", year), ("month", month), ("day", day)]:
        api("configure", p, v)

    r = api("configure", "syncRatio", sync)
    hint = r.get("needConfig", "")
    print(f"  hint: {hint[:150]}")

    stab = parse_stabilization(hint)
    if stab is not None:
        print(f"  stabilization = {stab}")
        api("configure", "stabilization", stab)

    # 3. Konfiguracja UI
    ui_update({"PTA": pta, "PTB": ptb})
    ui_update({"PWR": pwr})
    time.sleep(0.5)

    # 4. Przełącz na active
    ui_update({"mode": "active"})
    time.sleep(0.5)

    # 5. Czekaj na internalMode + flux=100%
    # Flux density = 100% dopiero gdy internalMode pasuje do roku docelowego.
    # internalMode zmienia się automatycznie co kilka sekund (cykl 1→2→3→4→1...)
    print(f"  Czekam na internalMode={mode} i flux=100%...")
    for i in range(120):
        cfg = api("getConfig")["config"]
        if cfg["internalMode"] == mode and cfg["fluxDensity"] == 100:
            print(f"  internalMode={mode} ✓  flux=100% ✓")
            break
        if i % 5 == 0:
            print(f"    [{i}] mode={cfg['internalMode']} flux={cfg['fluxDensity']}%")
        time.sleep(2)
    else:
        print(f"  ⚠ TIMEOUT! mode={cfg['internalMode']} flux={cfg['fluxDensity']}%")
        return cfg

    # 6. Skok!
    print(f"  >>> EXECUTING TIME TRAVEL <<<")
    result = api("timeTravel")
    print(f"  result: {json.dumps(result, ensure_ascii=False)[:300]}")

    # Sprawdź stan po skoku
    cfg = api("getConfig")
    c = cfg.get("config", {})
    print(f"  Po skoku: date={c.get('currentDate')} battery={c.get('batteryStatus')}")

    # Flaga?
    flag = cfg.get("flag", result.get("flag", ""))
    if flag:
        print(f"\n  *** FLAGA: {flag} ***")

    return cfg

# ============================================================
# MAIN — trzy skoki
# ============================================================

def main():
    print("CHRONOS-P1 — Pełny automat")
    print("=" * 60)

    # Reset
    r = api("reset")
    print(f"  Reset: {r['message']}")
    c = r["config"]
    print(f"  Battery: {c['batteryStatus']}, Date: {c['currentDate']}")

    # SKOK 1: → 2238 po baterie
    execute_jump(5, 11, 2238, "jump")

    # SKOK 2: → 2026 powrót
    execute_jump(11, 4, 2026, "jump")

    # SKOK 3: → 2024 tunel do Rafała
    cfg = execute_jump(12, 11, 2024, "tunnel")

    # Wynik
    flag = cfg.get("flag", "")
    if flag:
        print(f"\n{'='*60}")
        print(f"  SUKCES! Flaga: {flag}")
        print(f"{'='*60}")

if __name__ == "__main__":
    main()
