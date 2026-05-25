import os
from dotenv import load_dotenv
load_dotenv()

import requests
import time
import json

# ============================================================
# S01E05 — Zarządzanie jawnymi i niejawnymi limitami modeli
# ============================================================
#
# Lekcja S01E05 skupia się na ograniczeniach produkcyjnych systemów AI:
# limitach API, kosztach tokenów, halucynacjach i odporności architektury.
# Zadanie "railway" demonstruje dwa z najważniejszych limitów jawnych:
#
# 1. HTTP 503 — serwer chwilowo przeciążony (Service Unavailable).
#    API celowo symuluje awarie. Kod MUSI je obsługiwać automatycznie
#    (retry z backoffem), inaczej zadanie jest nie do ukończenia.
#
# 2. HTTP 429 — przekroczono limit zapytań (Too Many Requests).
#    Każdy provider API (OpenAI, Anthropic, własne serwisy) ma limity
#    requests-per-minute i tokens-per-minute. Ignorowanie ich skutkuje
#    blokadą konta. Nagłówki HTTP mówią KIEDY można wznowić zapytania.
#
# Kluczowa zasada lekcji S01E05: aplikacja produkcyjna musi być zaprojektowana
# tak, żeby zewnętrzne ograniczenia nie powodowały awarii, lecz były obsługiwane
# gracefully — z informowaniem użytkownika i automatycznym wznawianiem.
#
# To samo API odkryliśmy metodą "help" — API samo-dokumentujące to wzorzec
# z lekcji (agent zaczyna od poznania dostępnych akcji, nie od zgadywania).

API_KEY = os.environ["HUB_API_KEY"]
URL = "https://hub.ag3nts.org/verify"
ROUTE = "X-01"


def call_api(action: str, extra: dict = {}) -> dict:
    """Wysyła akcję do API kolejowego z automatyczną obsługą błędów HTTP.

    Ta funkcja implementuje wzorzec "retry with backoff" — kluczowy dla
    produkcyjnych aplikacji AI. Lekcja S01E05 wyjaśnia, że przeciążenia
    i limity to codzienność na produkcji i musimy je adresować kodem,
    nie ręcznym ponawianem zapytań.
    """
    payload = {
        "apikey": API_KEY,
        "task": "railway",
        "answer": {"action": action, "route": ROUTE, **extra}
    }

    # Pętla retry — nie ograniczamy liczby prób, bo nie wiemy kiedy serwer
    # się zwolni. W produkcji dodalibyśmy max_retries z timeout-em całościowym.
    while True:
        resp = requests.post(URL, json=payload)
        headers = resp.headers

        # ============================================================
        # NAGŁÓWKI RATE LIMIT — informacje o stanie limitów API
        # ============================================================
        #
        # Lekcja S01E05: limity zapytań API to "jawne ograniczenia" —
        # provider explicite informuje o nich w nagłówkach HTTP:
        #
        # X-RateLimit-Limit     — maksymalna liczba zapytań w oknie czasowym
        # X-RateLimit-Remaining — ile zapytań zostało przed limitem
        # X-RateLimit-Reset     — kiedy (timestamp/sekundy) limit się zeruje
        # Retry-After           — ile sekund poczekać przed kolejną próbą
        #
        # Logowanie tych nagłówków to niezbędny element debugowania —
        # lekcja S01E05 podkreśla, że przy limitach "dobre logowanie to
        # podstawa". Bez tego nie wiemy czy jesteśmy blisko wyczerpania limitu.
        for h in ["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset", "Retry-After"]:
            if h in headers:
                print(f"  [{h}]: {headers[h]}")

        # ============================================================
        # OBSŁUGA 503 — przeciążenie serwera (celowe w tym zadaniu)
        # ============================================================
        #
        # HTTP 503 Service Unavailable = serwer żyje, ale nie może teraz
        # obsłużyć zapytania. To sytuacja przejściowa — za chwilę zasoby
        # się zwolnią. Właściwa reakcja: odczekaj Retry-After sekund i spróbuj
        # ponownie. BEZ tego retry, zadanie jest nie do ukończenia.
        #
        # W kontekście AI: dostawcy modeli (Anthropic, OpenAI) też zwracają
        # 503 przy przeciążeniu klastrów GPU. Produkcyjna aplikacja musi to
        # obsługiwać, bo model nie jest zawsze dostępny 24/7 bez przestojów.
        if resp.status_code == 503:
            wait = int(headers.get("Retry-After", 5))
            print(f"  503 — czekam {wait}s...")
            time.sleep(wait)
            continue  # wróć na początek pętli — ponów zapytanie

        # ============================================================
        # OBSŁUGA 429 — przekroczono limit zapytań
        # ============================================================
        #
        # HTTP 429 Too Many Requests = wysłaliśmy za dużo zapytań
        # w zbyt krótkim czasie. To "jawny limit" z lekcji S01E05.
        #
        # Kluczowa różnica vs 503:
        # - 503: serwer jest zajęty, za chwilę OK
        # - 429: my zbyt agresywnie pingujemy, musimy zwolnić
        #
        # Dlatego wait przy 429 jest domyślnie dłuższy (30s vs 5s przy 503).
        # X-RateLimit-Reset mówi kiedy "okno" limitu się resetuje.
        #
        # Lekcja S01E05: "zbyt agresywne odpytywanie spowoduje długie blokady"
        # — jeśli zignorujemy 429 i będziemy dalej pingować, blokada może
        # wzrosnąć wykładniczo (exponential backoff jest tu preferowany).
        if resp.status_code == 429:
            reset = headers.get("X-RateLimit-Reset", "?")
            wait = int(headers.get("Retry-After", 30))
            print(f"  429 rate limit — reset: {reset}, czekam {wait}s...")
            time.sleep(wait)
            continue  # wróć na początek pętli — ponów po odczekaniu

        # Sukces — logujemy pełną odpowiedź (ważne przy szukaniu flagi)
        data = resp.json()
        print(f"  -> {json.dumps(data, ensure_ascii=False)}")
        return data


# ============================================================
# GŁÓWNA LOGIKA — sekwencja akcji odkryta przez "help"
# ============================================================
#
# Zadanie wymaga najpierw wywołania akcji "help", która zwraca
# samo-dokumentację API: listę akcji, ich parametry i KOLEJNOŚĆ wywołań.
# To wzorzec "API samo-dokumentujące" — agent nie potrzebuje zewnętrznej
# dokumentacji, bo otrzymuje ją z samego API.
#
# Kolejność kroków odkryta przez "help":
# 1. reconfigure — resetuje trasę do stanu bazowego
# 2. setstatus RTOPEN — zmienia status na "otwarta"
# 3. getstatus — weryfikuje czy zmiana się powiodła
# 4. save — zapisuje zmiany trwale (bez tego reset po restarcie)
#
# Opóźnienia time.sleep(2) między krokami to prewencja 429 —
# szanujemy limity API nawet gdy nie dostaliśmy jeszcze błędu.
# Lekcja S01E05: "pilnuj limitów zapytań — monitoruj nagłówki po każdym żądaniu".

def main():
    print(f"\n=== 1. reconfigure {ROUTE} ===")
    r = call_api("reconfigure")
    time.sleep(2)   # prewencyjny backoff — nie czekamy na 429, działamy z wyprzedzeniem

    print(f"\n=== 2. setstatus {ROUTE} = RTOPEN ===")
    r = call_api("setstatus", {"value": "RTOPEN"})
    time.sleep(2)

    print(f"\n=== 3. getstatus {ROUTE} ===")
    r = call_api("getstatus")
    time.sleep(2)

    print(f"\n=== 4. save {ROUTE} ===")
    r = call_api("save")
    # Odpowiedź "save" zawiera flagę {FLG:...} — znak ukończenia zadania

    print("\nGotowe!")


if __name__ == "__main__":
    main()
