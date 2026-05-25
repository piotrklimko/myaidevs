# S03E04 — Negotiations: Tutorial krok po kroku

## Cel zadania

Przygotować 1-2 narzędzia (endpointy HTTP), z których skorzysta zewnętrzny agent AI
z centrali. Agent szuka **3 przedmiotów** potrzebnych do budowy turbiny wiatrowej
i musi ustalić, **które miasta oferują WSZYSTKIE z nich jednocześnie**.

Agent wysyła zapytania w **języku naturalnym** (np. "potrzebuję turbiny wiatrowej"),
a narzędzie musi zwrócić listę miast.

**Wynik:** Domatowo i Skolwin — flaga `{FLG:WINDFARM}`.

---

## Krok 1: Pobranie i analiza danych

Pobieramy trzy pliki CSV z centrali:

```bash
mkdir -p S03E04 && cd S03E04
curl -O https://hub.ag3nts.org/dane/s03e04_csv/cities.csv
curl -O https://hub.ag3nts.org/dane/s03e04_csv/connections.csv
curl -O https://hub.ag3nts.org/dane/s03e04_csv/items.csv
```

### Co zawierają:

| Plik             | Wierszy | Opis                                           |
|------------------|---------|-------------------------------------------------|
| `cities.csv`     | 51      | Miasta handlarzy: `name,code` (np. Warszawa,A7K3QX) |
| `items.csv`      | 2137    | Przedmioty: `name,code` (np. Turbina wiatrowa 400W 48V,WITR48) |
| `connections.csv`| 5350    | Powiązania: `itemCode,cityCode` — który przedmiot jest w którym mieście |

### Kluczowe obserwacje z analizy:

- Przedmioty to **komponenty elektroniczne** (rezystory, kondensatory, inwertery, akumulatory, turbiny...)
- Są duplikaty kodów — np. `06OTEA` to zarówno "Akumulator AGM 48V 150Ah" jak i "Akumulator kwasowy 12V 200Ah"
- Przedmioty związane z turbiną wiatrową (grep "turbin|inwert|akumul"):
  - Turbina wiatrowa 400W 24V (WITR24) — 4 miasta
  - Turbina wiatrowa 400W 48V (WITR48) — 3 miasta
  - Inwerter DC/AC 12V 1500W (A94ZZ4) — 4 miasta
  - Inwerter DC/AC 48V 3000W (A94MAZ) — 3 miasta
  - Akumulator AGM 48V 150Ah (06OTEA) — 3 miasta
  - Akumulator kwasowy 12V 200Ah (06OTEA) — 3 miasta (ten sam kod!)

---

## Krok 2: Decyzje projektowe

### Ile narzędzi?

**Jedno.** Agent ma max 10 kroków i szuka 3 przedmiotów. Jedno narzędzie
łączące wyszukiwanie + lookup miast = 3 zapytania = 3 kroki. Proste.

Lekcja mówi: "łączenie akcji w jedno narzędzie" — nie ma sensu rozbijać
na osobne "szukaj przedmiot" i "podaj miasta", bo agent i tak zawsze
potrzebuje obu informacji naraz.

### Deterministycznie czy LLM?

**Hybrid.** To kluczowa decyzja architektoniczna zgodna z duchem lekcji:

```
Zapytanie agenta (język naturalny)
        │
        ▼
┌─────────────────────┐
│  Keyword search     │  ← DETERMINISTYCZNY: szybki, darmowy, 90% przypadków
│  (stemming + score) │
└────────┬────────────┘
         │
    ┌────┴────┐
    │ 1 wynik │──→ Użyj go (bez LLM!)
    │         │
    │ 2-3     │──→ Zwróć wszystkie warianty z miastami
    │ wyniki  │    (agent sam wybierze)
    │         │
    │ >3      │──→ LLM disambiguuje (wybiera 1 z wielu)
    │ wyników │
    │         │
    │ 0       │──→ LLM fallback (przeszukuje całą bazę)
    └─────────┘
         │
         ▼
┌─────────────────────┐
│  Lookup miast       │  ← DETERMINISTYCZNY: connections.csv
│  (code → cities)    │
└─────────────────────┘
         │
         ▼
    Odpowiedź JSON
```

### Dlaczego nie czyste LLM?

- **Koszt**: 2137 przedmiotów × 3 zapytania = niepotrzebne tokeny
- **Szybkość**: keyword search = <1ms, LLM = 1-2s
- **Niezawodność**: keyword search nigdy nie "halucynuje"
- **Lekcja**: "używaj LLM tam gdzie ma sens, resztę rób deterministycznie"

---

## Krok 3: Implementacja serwera (server.py)

### 3a. Ładowanie danych

Dane ładujemy RAZ przy starcie (wzorzec "load once, serve many"):

```python
ITEMS = load_csv("items.csv", "name", "code")      # [(nazwa, kod), ...]
CITY_BY_CODE = {code: name for name, code in ...}   # kod_miasta → nazwa
CONNECTIONS = {item_code: {city_codes}}              # kod_przedmiotu → set(kody_miast)
```

### 3b. Normalizacja tekstu

Agent może napisać "łącznik" a w CSV jest "Lacznik". Normalizujemy:
1. Lowercase
2. Usunięcie znaków diakrytycznych (ą→a, ć→c)
3. Usunięcie znaków specjalnych

```python
normalize("Turbina wiatrowa 400W") → "turbina wiatrowa 400w"
normalize("potrzebuję łącznika")   → "potrzebuje lacznika"
```

### 3c. Prosty stemmer polski

Problem: agent pisze "turbiny wiatrowej" (dopełniacz), a w CSV jest
"Turbina wiatrowa" (mianownik). Słowo "turbiny" ≠ "turbina".

Rozwiązanie: obcinamy końcówki fleksyjne:
```
turbiny  → turbin
turbina  → turbin   ✓ match!
wiatrowej → wiatrow
wiatrowa  → wiatrow  ✓ match!
```

Nie potrzebujemy pełnego stemmera (Stempel) — wystarczy lista ~30 końcówek.

### 3d. Scoring wyników

Dla każdego przedmiotu liczymy score = suma dopasowań tokenów:
- Token jest podciągiem nazwy → +2 pkt (najsilniejsze)
- Stem tokena = stem słowa w nazwie → +1.5 pkt
- Stem jest podciągiem stemu → +1 pkt
- Bonus za frazę (tokeny obok siebie) → +N pkt

### 3e. LLM disambiguation

Gdy keyword search zwrócił >3 wyników z tym samym score, pytamy Haiku:
```
System: "Wybierz JEDEN najlepiej pasujący przedmiot. Odpowiedz TYLKO kodem."
User: "Zapytanie: {query}\n\nDostępne:\n- Nazwa (kod: XXX)\n..."
```

Parametry: `temperature=0` (deterministyczność), `max_tokens=20` (sam kod).

### 3f. Endpoint API

```
POST /api/search
Body: {"params": "turbina wiatrowa"}
Response: {"output": "Turbina wiatrowa 400W 24V: Bydgoszcz, Jaworzno, ... | Turbina wiatrowa 400W 48V: Domatowo, ..."}
```

Ograniczenia:
- Odpowiedź: 4-500 bajtów
- Format: `Nazwa: Miasto1, Miasto2 | Nazwa2: Miasto3, Miasto4`

---

## Krok 4: Deployment na Azyl

### 4a. Połączenie SSH

```bash
ssh agent18356@azyl.ag3nts.org -p 5022
# hasło: <AZYL_SSH_PASSWORD>
```

**UWAGA:** Port to **5022**, nie 5222 (CLAUDE.md ma błędne dane!).

Bez `sshpass` użyliśmy triku z `SSH_ASKPASS`:
```bash
export SSH_ASKPASS_REQUIRE=force
export SSH_ASKPASS=$(mktemp)
echo '#!/bin/sh' > $SSH_ASKPASS
echo 'echo <AZYL_SSH_PASSWORD>' >> $SSH_ASKPASS
chmod +x $SSH_ASKPASS
ssh -o StrictHostKeyChecking=no agent18356@azyl.ag3nts.org -p 5022 "polecenie"
rm -f $SSH_ASKPASS
```

### 4b. Upload plików

```bash
scp -P 5022 server.py cities.csv connections.csv items.csv agent18356@azyl.ag3nts.org:~/s03e04/
```

### 4c. Uruchomienie serwera

```bash
ssh -f agent18356@azyl.ag3nts.org -p 5022 \
  "cd ~/s03e04 && python3 server.py > ~/s03e04/server.log 2>&1"
```

Serwer nasłuchuje na porcie **18356**. Nginx na Azylu mapuje:
`https://azyl-18356.ag3nts.org` → `localhost:18356`

### 4d. Test publicznego endpointu

```bash
curl -X POST https://azyl-18356.ag3nts.org/api/search \
  -H "Content-Type: application/json" \
  -d '{"params": "turbina wiatrowa"}'
```

Oczekiwana odpowiedź:
```json
{"output": "Turbina wiatrowa 400W 24V: Bydgoszcz, Jastrzebie-Zdroj, Jaworzno, Olsztyn | Turbina wiatrowa 400W 48V: Domatowo, Rzeszow, Skolwin"}
```

---

## Krok 5: Rejestracja narzędzia w centrali

```bash
curl -X POST https://hub.ag3nts.org/verify \
  -H "Content-Type: application/json" \
  -d '{
    "apikey": "<HUB_API_KEY>",
    "task": "negotiations",
    "answer": {
      "tools": [
        {
          "URL": "https://azyl-18356.ag3nts.org/api/search",
          "description": "Wyszukuje przedmioty w miastach handlarzy. W params podaj nazwe lub opis przedmiotu po polsku (np. turbina wiatrowa, akumulator, inwerter). Zwraca liste miast gdzie przedmiot jest dostepny."
        }
      ]
    }
  }'
```

**Ograniczenie:** opis narzędzia max **300 znaków**! Pierwsza próba z dłuższym opisem
została odrzucona.

---

## Krok 6: Weryfikacja asynchroniczna

Rejestracja zwraca "queued for verification". Agent centrali potrzebuje 30-60 sekund
na wykonanie swoich zapytań. Potem sprawdzamy:

```bash
curl -X POST https://hub.ag3nts.org/verify \
  -H "Content-Type: application/json" \
  -d '{
    "apikey": "<HUB_API_KEY>",
    "task": "negotiations",
    "answer": {"action": "check"}
  }'
```

Odpowiedź:
```json
{
  "code": 0,
  "message": "{FLG:WINDFARM}",
  "cities": ["Domatowo", "Skolwin"]
}
```

---

## Co zrobił agent centrali (rekonstrukcja z logów)

Agent wykonał 4 zapytania POST do naszego endpointu. Prawdopodobny scenariusz:

1. **"turbina wiatrowa"** → dostał obie wersje (24V i 48V) z miastami
2. **"inwerter"** → dostał oba inwertery z miastami
3. **"akumulator"** → dostał oba akumulatory z miastami
4. Może jedno dodatkowe zapytanie doprecyzowujące

Następnie agent sam obliczył przecięcie zbiorów miast i stwierdził,
że **Domatowo** i **Skolwin** oferują wszystkie 3 typy przedmiotów jednocześnie.

---

## Lekcje z tego zadania

### 1. Dopasuj narzędzie, nie rób generycznego
Jedno narzędzie "szukaj przedmiot → miasta" zamiast dwóch osobnych.
Agent ma 10 kroków — nie marnuj ich na pośrednie akcje.

### 2. Hybrid: deterministycznie + LLM
Keyword search obsłużył 100% zapytań agenta (nie trzeba było LLM fallbacku).
LLM jest "ubezpieczeniem" na wypadek, gdyby agent użył bardzo opisowego języka.

### 3. Zwracaj bogaty kontekst
Gdy mamy 2-3 warianty (np. turbina 24V i 48V), zwracamy WSZYSTKIE
z ich miastami. Agent sam decyduje. Gdybyśmy losowo wybrali jeden wariant,
agent mógłby przegapić prawidłowe miasto.

### 4. Opis narzędzia jest krytyczny
Agent nie widzi naszego kodu — jedyne co wie o narzędziu, to jego **opis**
(max 300 znaków). Musi jasno mówić:
- Co narzędzie robi
- Co podać w `params`
- Co dostanie w odpowiedzi

### 5. Ograniczenia kształtują architekturę
- 500 bajtów na odpowiedź → zwięzły format, bez JSONa wewnątrz
- 10 kroków agenta → jedno narzędzie zamiast dwóch
- 300 znaków opisu → precyzyjny, konkretny opis

### 6. Normalizacja i stemming > embeddingi
Dla bazy 2000+ przedmiotów z opisowymi nazwami, prosty keyword search
ze stemmingiem jest szybszy, tańszy i bardziej przewidywalny niż
wektorowe similarity search. Embeddingi to overengineering tutaj.

### 7. SSH na Azyl: port 5022
CLAUDE.md w repozytorium miał błędny port (5222). Poprawny: **5022**.
Bez `sshpass` można użyć `SSH_ASKPASS` trick.
