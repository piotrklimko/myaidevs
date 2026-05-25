import os
from dotenv import load_dotenv
load_dotenv()

import requests
import json

# --- CONFIG ---
HUB_API_KEY = os.environ["HUB_API_KEY"]
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
VERIFY_URL = "https://hub.ag3nts.org/verify"

def execute_cmd(cmd):
    payload = {
        "apikey": HUB_API_KEY,
        "task": "shellaccess",
        "answer": {
            "cmd": cmd
        }
    }
    response = requests.post(VERIFY_URL, json=payload)
    return response.json()

# KROK 1: Przeszukujemy pliki w /data w poszukiwaniu wzmianki o Rafale
# Używamy grep, aby znaleźć konkretną datę i lokalizację
# print("Szukam informacji o Rafale...")
# search_result = execute_cmd("grep -ri 'Rafał' /data")
# print("Wynik wyszukiwania:", json.dumps(search_result, indent=2, ensure_ascii=False))

# Najpierw: sprawdzamy strukturę /data
print("=== KROK 1: Struktura /data ===")
ls_result = execute_cmd("ls -la /data")
print("Zawartość /data:", json.dumps(ls_result, indent=2, ensure_ascii=False))

print("\n=== KROK 2: Szukam 'znaleziono' w pliku CSV ===")
search_result = execute_cmd("grep -i 'znaleziono' /data/time_logs.csv")
print("Wynik grep 'znaleziono':", json.dumps(search_result, indent=2, ensure_ascii=False))

print("\n=== KROK 3: Szukam 'Rafał' AND 'znaleziono' w CSV ===")
search_result2 = execute_cmd("grep -i 'Rafał' /data/time_logs.csv | grep -i 'znaleziono'")
print("Wynik grep 'Rafał' + 'znaleziono':", json.dumps(search_result2, indent=2, ensure_ascii=False))

print("\n=== KROK 4: Wydobywam miasto (grep -A1) ===")
city_grep = execute_cmd("grep -A1 '\"location_id\": 219' /data/locations.json")
print("Wpis location 219:", json.dumps(city_grep, indent=2, ensure_ascii=False))

print("\n=== KROK 5: Wydobywam GPS (grep -A5) ===")
gps_grep = execute_cmd("grep -B2 -A5 '954634' /data/gps.json")
print("Wpis GPS 954634:", json.dumps(gps_grep, indent=2, ensure_ascii=False)[:600])

print("\n=== KROK 6: Szukam dokładnie struktury dla entry_id 954634 ===")
# Sprawdzam linię nr 954634 i kontekst
gps_full = execute_cmd("grep -B10 -A10 '954634' /data/gps.json")
print("Full struktura 954634:\n", json.dumps(gps_full, indent=2, ensure_ascii=False)[:800])

print("\n=== KROK 7: Wydobywam dokładną nazwę miasta z locations.json ===")
# Próbuję wydobyć 'name' bezpośrednio
city_json = execute_cmd("grep -A1 '\"location_id\": 219' /data/locations.json | grep 'name'")
print("Nazwa z JSON:", json.dumps(city_json, indent=2, ensure_ascii=False))

# Spróbuję z sed/python-fu: używam python do parsowania
parse_cmd = execute_cmd("python3 -c \"import json; data=json.load(open('/data/locations.json')); print(next(l['name'] for l in data if l.get('location_id')==219))\"")
print("Python parse:", json.dumps(parse_cmd, indent=2, ensure_ascii=False))

print("\n=== KROK 8: Przygotowuję odpowiedź z prawidłową nazwą miasta ===")
final_answer = {
    "date": "2024-11-12",
    "city": "Grudziadz",
    "longitude": 18.968774,
    "latitude": 53.432303
}
print("Odpowiedź:", json.dumps(final_answer, indent=2, ensure_ascii=False))

print("\n=== KROK 9: Wysyłam ===")
cmd = f"echo '{json.dumps(final_answer)}'"
result = execute_cmd(cmd)
print("Wynik:", json.dumps(result, indent=2, ensure_ascii=False))

# --- TUTAJ ANALIZUJESZ WYNIK Z KROKU 1 ---
# Załóżmy, że znalazłeś: "2024-08-12: Znaleziono ciało Rafała w Warszawie (52.2297, 21.0122)"
# Pamiętaj, że musisz podać datę DZIEŃ WCZEŚNIEJ (2024-08-11).

# KROK 2: Wysłanie finalnej odpowiedzi (podstaw znalezione dane)
# Odkomentuj poniższe linie, gdy będziesz mieć już komplet danych:

# final_answer = {
#     "date": "RRRR-MM-DD", # dzień przed znalezieniem
#     "city": "Nazwa Miasta",
#     "longitude": 00.000000,
#     "latitude": 00.000000
# }
# cmd_final = f"echo '{json.dumps(final_answer)}'"
# result = execute_cmd(cmd_final)
# print("Odpowiedź serwera:", result)