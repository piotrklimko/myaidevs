# AI_devs 4 (Builders) — workspace

## Setup
- Python 3.12, Ubuntu (projekt przeniesiony z Windows)
- Virtualenv: `.venv/` (Linux) — aktywuj: `source .venv/bin/activate`
- Zależności: `pip install -r requirements.txt`

## Klucze API
- **HUB_API_KEY**: w pliku `.env`
- **OPENROUTER_API_KEY**: w pliku `.env`

## LLM — OpenRouter
```python
from openai import OpenAI
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)
```
- **Model domyślny**: `anthropic/claude-haiku-4.5`
- **Structured Output (json_schema)**: `openai/gpt-4o-mini`

## Serwer Azyl (publiczny endpoint)
- URL: `https://azyl-18356.ag3nts.org`
- SSH: `ssh agent18356@azyl.ag3nts.org -p 5022` (hasło w `.env`: `AZYL_SSH_PASSWORD`)
- Serwer musi słuchać na porcie **18356** (nginx mapuje subdomenę → port)

## Struktura projektu
```
e01/
├── CLAUDE.md
├── requirements.txt
├── lekcje/          # materiały kursowe (.md)
├── S01E01/          # people — CSV + LLM tagging + Structured Output ✅
├── S01E02/          # findhim — Function Calling + Haversine ✅
├── S01E03/          # proxy — serwer HTTP + agent logistyczny ✅
└── S01E04/          # sendit — agent multimodalny (fetch_url + vision) ✅
    └── _scratch/    # pliki pośrednie i dokumenty
```

## Postęp zadań
| Zadanie | Nazwa | Techniki | Status |
|---------|-------|----------|--------|
| S01E01  | people | Filtrowanie CSV, LLM tagging, Structured Output | ✅ |
| S01E02  | findhim | Function Calling, Haversine, wynik: Wojciech Bielik | ✅ |
| S01E03  | proxy | HTTP server na Azylu, agent z podmianą celu | ✅ |
| S01E04  | sendit | Agent multimodalny, fetch_url + vision, deklaracja SPK | ✅ |
| S02E01  | categorize | Context Engineering, prompt caching, klasyfikator DNG/NEU | ✅ |
| S02E02  | electricity | Analiza PNG pixel-scan, wykrywanie połączeń kabli, obroty 3×3 | ✅ |
| S02E03  | failure | Kompresja logów: deduplikacja CRIT/ERRO do 36 unikalnych zdarzeń (<1500 tokenów) | ✅ |
| S02E04  | mailbox | Orchestrator: przeszukiwanie skrzynki zmail, dwuetapowe pobieranie danych, aktywna skrzynka | ✅ |
| S02E05  | drone | Vision (GPT-5.4) + reactive approach, DRN-BMB7 API, skanowanie siatki, tama (2,4) | ✅ |
| S03E01  | evaluation | Anomalie sensorów: programistyczne (zakresy + pola) + LLM klasyfikacja notatek, cache 2032 unikalnych notatek z 9999 plików | ✅ |
| S03E04  | negotiations | Narzędzie dla agenta: keyword search + stemming + LLM fallback, endpoint na Azylu, miasta z przedmiotami | ✅ |
| S04E04  | filesystem | Agent 2-etapowy (ekstrakcja + walidacja), batch API, baza wiedzy Markdown | ✅ |
| S04E05  | foodwarehouse | Agent magazynowy: SQLite + signatureGenerator + orders API, batch append | ✅ |
| S05E01  | radiomonitoring | Pipeline agentowy: router danych (tekst/obraz/audio/JSON/CSV/Morse), multimodalny nasłuch, synteza LLM | ✅ |
| S05E02  | phonecall | Voice agent: OpenAI gpt-audio (echo) TTS + Gemini STT, rozmowa z operatorem OKO, wyłączenie monitoringu RD820 | ✅ |
| S05E04  | goingthere | Nawigacja rakietą 3×12: skaner częstotliwości + disarm SHA1, hinty radiowe + LLM, omijanie skał | ✅ |
| S05E05  | timetravel | Pełny automat: dual API (/verify + /timetravel_backend), syncRatio, LLM parsing hintów PL, 3 skoki | ✅ |

## Styl pracy
- Krok po kroku, po polsku
- Kod do uruchomienia → czekaj na wyniki → dalej
- Klucze hardkodowane bezpośrednio w skryptach (projekt edukacyjny)
- Skrypty w Pythonie, styl "agentowy"
- Po rozwiązaniu zadania — obszerne komentarze w skrypcie (do nauki z lekcji)
- HTTP requests przez Python (`requests`), nie przez `curl` (żeby nie zatwierdzać każdego kroku)
- Środowisko: kontener Docker — venv nie jest potrzebny
