import os
from dotenv import load_dotenv
load_dotenv()

"""
S05E05 — Maszyna czasu CHRONOS-P1 (timetravel)
================================================
Asystent CLI do obsługi maszyny czasu. Konfiguruje parametry przez API,
oblicza syncRatio, parsuje hinty stabilization z odpowiedzi API,
i instruuje operatora co ustawić w interfejsie webowym.

LEKCJA S05E05 — "Nowa Rzeczywistość"
=====================================
Ostatnia lekcja AI_devs 4: Builders. Omawia budowanie produkcyjnych systemów
agentowych — projekt "Wonderlands" łączący czat, agentów, MCP, sandbox,
cyfrowy ogród (baza wiedzy jako pliki Markdown), i wiele innych koncepcji
z całego kursu.

KLUCZOWE KONCEPCJE Z LEKCJI:
1. Human-in-the-loop — nie wszystko da się zautomatyzować. Agent asystuje
   człowiekowi, podpowiadając co zrobić, ale niektóre akcje (PT-A, PT-B, PWR,
   kliknięcie sfery) musi wykonać operator w interfejsie webowym.

2. Podział odpowiedzialności API vs UI — parametry day/month/year/syncRatio/
   stabilization konfigurujemy przez API, ale PT-A/PT-B/PWR/active/standby
   tylko przez interfejs. To odzwierciedla realny podział w systemach
   produkcyjnych: część logiki jest automatyzowalna, część wymaga interakcji.

3. Dokumentacja jako baza wiedzy agenta — agent musi przeczytać dokumentację
   urządzenia aby znać wzory (syncRatio), tabele (PWR), zasady (internalMode).
   W produkcji to odpowiednik RAG / knowledge base.

4. Kontekstowe podpowiedzi z API — po ustawieniu daty, API zwraca pole
   "needConfig" z hintem o stabilization napisanym naturalnym językiem.
   Agent musi sparsować ten tekst (np. "900 jednostek, obniż o 711" → 189).

5. Parametry zmienne w czasie — internalMode zmienia się automatycznie
   co kilka sekund. Agent musi poczekać na właściwą fazę zamiast próbować
   ją wymusić. To analogia do systemów reaktywnych i event-driven.

6. Flux density jako walidacja — wskaźnik poprawności konfiguracji.
   100% = wszystko OK. Każdy poprawny parametr dodaje ~20%.
   To odpowiednik health check / readiness probe w systemach produkcyjnych.

ZADANIE:
=========
Trzy skoki w czasie:
  1. → 5.11.2238  (przyszłość, PT-B only)   — po nowe baterie
  2. → 11.04.2026 (przeszłość, PT-A only)   — powrót do teraźniejszości
  3. → 12.11.2024 (tunel, PT-A+PT-B)        — spotkanie z Rafałem

OBLICZENIA:
============
syncRatio = ((day*8 + month*12 + year*7) % 101) / 100
  - Skok 1: (40+132+15666) % 101 = 82 → 0.82
  - Skok 2: (88+48+14182)  % 101 = 77 → 0.77
  - Skok 3: (96+132+14168) % 101 = 54 → 0.54

stabilization — parsowany z naturalnego języka w polu needConfig:
  - Skok 1: "900 obniż o 711" → 189
  - Skok 2: "900 obniż o 178" → 722 (hint zmienia się przy każdym resecie!)
  - Skok 3: "600 zwiększ o 395" → 995

PWR z tabeli w dokumentacji:
  - 2238 → 91, 2026 → 28, 2024 → 19

internalMode (automatyczny, trzeba czekać):
  - mode 1: < 2000
  - mode 2: 2000-2150 (skoki do 2026 i 2024)
  - mode 3: 2151-2300 (skok do 2238)
  - mode 4: > 2300

PT-A/PT-B:
  - PT-A = przeszłość, PT-B = przyszłość
  - Oba = tunel czasowy (wymaga baterii ≥ 60%)

WYNIK: flaga {FLG:FIXTHEWORLD}
"""

import requests
import json
import time
import re

# ============================================================
# KONFIGURACJA
# ============================================================
HUB_API_KEY = os.environ["HUB_API_KEY"]
BASE = "https://hub.ag3nts.org"

# ============================================================
# KOMUNIKACJA Z API
# ============================================================

def api(action, param=None, value=None):
    """
    Wysyła komendę do API maszyny czasu CHRONOS-P1.
    Akcje: help, getConfig, reset, configure (+ param + value).
    Konfiguracja możliwa TYLKO w trybie standby.
    """
    payload = {
        "apikey": HUB_API_KEY,
        "task": "timetravel",
        "answer": {"action": action}
    }
    if param is not None:
        payload["answer"]["param"] = param
        payload["answer"]["value"] = value
    for attempt in range(5):
        try:
            r = requests.post(f"{BASE}/verify", json=payload, timeout=15)
            return r.json()
        except Exception as e:
            print(f"  [retry {attempt+1}] {e}")
            time.sleep(2)
    return None


def get_config():
    """Pobiera aktualną konfigurację urządzenia."""
    return api("getConfig")


def configure(param, value):
    """Ustawia parametr i wyświetla wynik + ewentualny hint needConfig."""
    result = api("configure", param, value)
    print(f"  {param}={value}: {result.get('message', '')}")
    # needConfig zawiera hint o stabilization — kluczowy element!
    nc = result.get("needConfig", "")
    if nc:
        print(f"  needConfig: {nc[:200]}")
    return result


# ============================================================
# OBLICZENIA Z DOKUMENTACJI
# ============================================================

def calc_sync_ratio(day, month, year):
    """
    Wskaźnik temporalny (syncRatio) z dokumentacji CHRONOS-P1.
    Wzór: ((day*8 + month*12 + year*7) % 101) / 100
    Zwraca float 0.00-1.00 z dokładnością do 2 miejsc.

    Wagi: dzień=8, miesiąc=12, rok=7
    Modulo 101 daje wynik 0-100, dzielimy przez 100 → zakres 0.00-1.00
    """
    raw = (day * 8 + month * 12 + year * 7) % 101
    return round(raw / 100, 2)


def get_required_internal_mode(year):
    """
    Zwraca wymagany internalMode dla roku docelowego.
    internalMode zmienia się automatycznie co kilka sekund —
    operator musi POCZEKAĆ na właściwą fazę.
    """
    if year < 2000: return 1
    if year <= 2150: return 2
    if year <= 2300: return 3
    return 4


def parse_stabilization_hint(hint_text):
    """
    Parsuje naturalny język z pola needConfig aby wyliczyć stabilization.

    API zwraca hinty w stylu:
      "sugerują 900 jednostek... obniżenie o 711" → 900 - 711 = 189
      "poziom 600... zwiększyć o 395" → 600 + 395 = 995

    Kluczowa koncepcja: API samo podpowiada wartość, ale w formie
    naturalnego języka — trzeba go sparsować. To analogia do tego,
    jak agent przetwarza kontekstowe podpowiedzi z narzędzi.
    """
    if not hint_text:
        return None

    # Szukamy słów kluczowych + liczb
    # Wzorce: "obniżenie/obniżyć o X" lub "zwiększyć/zwiększenie o X"
    numbers = re.findall(r'(\d+)', hint_text)
    if len(numbers) < 2:
        return None

    base = int(numbers[0])
    adjustment = int(numbers[1])

    # Sprawdź kierunek: obniżenie vs zwiększenie
    if any(w in hint_text.lower() for w in ["obniż", "zmniejsz", "reduk"]):
        result = base - adjustment
    elif any(w in hint_text.lower() for w in ["zwiększ", "podwyż", "podnieś"]):
        result = base + adjustment
    else:
        # Domyślnie odejmij (najczęstszy wzorzec)
        result = base - adjustment

    print(f"  → stabilization = {base} ± {adjustment} = {result}")
    return result


# ============================================================
# PROCEDURA SKOKU
# ============================================================

def configure_jump(day, month, year):
    """
    Konfiguruje wszystkie parametry API dla skoku do danej daty.
    Zwraca słownik z obliczonymi wartościami i instrukcjami dla operatora.

    Parametry API: day, month, year, syncRatio, stabilization
    Parametry UI:  PT-A, PT-B, PWR, standby/active, kliknięcie sfery
    """
    sync = calc_sync_ratio(day, month, year)
    mode = get_required_internal_mode(year)

    print(f"\n  Konfiguruję: {day:02d}.{month:02d}.{year}")
    print(f"  syncRatio = {sync}, wymagany internalMode = {mode}")

    # Ustaw datę i syncRatio
    configure("year", year)
    configure("month", month)
    r = configure("day", day)
    configure("syncRatio", sync)

    # Parsuj hint o stabilization z needConfig
    hint = r.get("needConfig", "")
    # Pobierz najnowszy hint (może się zmienić po ustawieniu syncRatio)
    r2 = api("getConfig")
    hint = r2.get("needConfig", hint)

    stab = parse_stabilization_hint(hint)
    if stab is not None:
        configure("stabilization", stab)

    return get_config()


def print_ui_instructions(day, month, year, jump_type="jump"):
    """
    Wyświetla instrukcje dla operatora — co ustawić w interfejsie webowym.

    PT-A = przeszłość, PT-B = przyszłość
    Oba = tunel czasowy (zużywa więcej energii, wymaga baterii ≥ 60%)
    """
    # PWR z tabeli dokumentacji
    pwr_table = {2024: 19, 2026: 28, 2238: 91}
    pwr = pwr_table.get(year, "???")
    mode = get_required_internal_mode(year)

    if jump_type == "tunnel":
        pt = "PT-A = ON, PT-B = ON (tunel czasowy)"
    elif year > 2026:
        pt = "PT-A = OFF, PT-B = ON (przyszłość)"
    else:
        pt = "PT-A = ON, PT-B = OFF (przeszłość)"

    print(f"\n{'='*60}")
    print(f"  INSTRUKCJE DLA OPERATORA — interfejs webowy:")
    print(f"  1. {pt}")
    print(f"  2. PWR (suwak) = {pwr}")
    print(f"  3. Sprawdź flux density = 100%")
    print(f"  4. Przełącz na ACTIVE")
    print(f"  5. Poczekaj na internalMode = {mode}")
    print(f"  6. Kliknij pulsującą sferę (zielona)")
    print(f"{'='*60}")


# ============================================================
# GŁÓWNA PROCEDURA — 3 skoki
# ============================================================

def main():
    """
    Procedura trzech skoków w czasie:
    1. Do 2238 po baterie (PT-B, skok w przyszłość)
    2. Powrót do 2026 (PT-A, skok w przeszłość)
    3. Tunel do 2024 (PT-A+PT-B, tunel czasowy do Rafała)

    Po każdym skoku operator musi:
    - Przełączyć na STANDBY (aby API mogło konfigurować)
    - Ustawić PT-A/PT-B i PWR w interfejsie
    - Przełączyć na ACTIVE i kliknąć sferę
    """
    print("=" * 60)
    print("  CHRONOS-P1 — Asystent podróży w czasie")
    print("  UI: https://hub.ag3nts.org/timetravel_preview")
    print("=" * 60)

    # Reset
    print("\n  Resetuję urządzenie...")
    print(f"  {api('reset').get('message', '')}")

    # ── SKOK 1: → 5.11.2238 po baterie ──
    print(f"\n{'#'*60}")
    print(f"# SKOK 1/3: → 5.11.2238 (po nowe baterie)")
    print(f"{'#'*60}")
    input("\nUpewnij się że UI jest w STANDBY. ENTER aby kontynuować...")

    configure_jump(5, 11, 2238)
    print_ui_instructions(5, 11, 2238, "jump")
    input("\nWykonaj skok w UI, potem ENTER...")

    cfg = get_config()
    print(f"  currentDate: {cfg['config']['currentDate']}")
    print(f"  batteryStatus: {cfg['config']['batteryStatus']}")

    # ── SKOK 2: → 11.04.2026 powrót ──
    print(f"\n{'#'*60}")
    print(f"# SKOK 2/3: → 11.04.2026 (powrót do teraźniejszości)")
    print(f"{'#'*60}")
    input("\nPrzełącz na STANDBY w UI, potem ENTER...")

    configure_jump(11, 4, 2026)
    print_ui_instructions(11, 4, 2026, "jump")
    input("\nWykonaj skok w UI, potem ENTER...")

    cfg = get_config()
    print(f"  currentDate: {cfg['config']['currentDate']}")
    print(f"  batteryStatus: {cfg['config']['batteryStatus']}")

    # ── SKOK 3: → 12.11.2024 tunel do Rafała ──
    print(f"\n{'#'*60}")
    print(f"# SKOK 3/3: → 12.11.2024 (tunel do Rafała)")
    print(f"{'#'*60}")
    input("\nPrzełącz na STANDBY w UI, potem ENTER...")

    configure_jump(12, 11, 2024)
    print_ui_instructions(12, 11, 2024, "tunnel")
    input("\nWykonaj skok w UI, potem ENTER...")

    # Sprawdź flagę
    cfg = get_config()
    print(json.dumps(cfg, indent=2, ensure_ascii=False))

    flag = cfg.get("flag", "")
    if flag:
        print(f"\n*** FLAGA: {flag} ***")
    else:
        print("\n  Brak flagi — sprawdź interfejs webowy.")


if __name__ == "__main__":
    main()
