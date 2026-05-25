import os
from dotenv import load_dotenv
load_dotenv()

#!/usr/bin/env python3
"""
S05E01 — radiomonitoring
========================
Zadanie: przechwycić materiały z nasłuchu radiowego i wydobyć informacje
o mieście "Syjon": prawdziwą nazwę, powierzchnię, liczbę magazynów, nr telefonu.

ARCHITEKTURA AGENTOWA (z lekcji S05E01):
- Gateway LLM — centralny punkt komunikacji z AI przez OpenRouter
- Router danych — programistyczne decyzje co wymaga LLM, a co nie
- Pipeline zamiast jednego wielkiego prompta — oszczędność tokenów
- Multimodalność — obsługa tekstu, obrazów, audio, JSON, Morse'a
"""

import requests
import base64
import json
import re
import io
from openai import OpenAI

# =============================================================================
# KONFIGURACJA — Gateway (lekcja: scentralizowana komunikacja z AI)
# =============================================================================
HUB_API_KEY = os.environ["HUB_API_KEY"]
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
VERIFY_URL = "https://hub.ag3nts.org/verify"

# Gateway LLM — jeden klient, łatwe przełączanie modeli/providerów
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

TEXT_MODEL = "anthropic/claude-haiku-4.5"
VISION_MODEL = "anthropic/claude-haiku-4.5"

# =============================================================================
# DEKODER MORSE'A — deterministyczna logika zamiast LLM (oszczędność tokenów)
# =============================================================================
# Lekcja uczy: co da się zrobić kodem, rób kodem. LLM tylko tam, gdzie trzeba.
MORSE_CODE = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E',
    '..-.': 'F', '--.': 'G', '....': 'H', '..': 'I', '.---': 'J',
    '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O',
    '.--.': 'P', '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T',
    '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y',
    '--..': 'Z', '-----': '0', '.----': '1', '..---': '2', '...--': '3',
    '....-': '4', '.....': '5', '-....': '6', '--...': '7', '---..': '8',
    '----.': '9',
}

def decode_morse_tati(text: str) -> str:
    """
    Dekoduje Morse'a w formacie Ti/Ta (Ti=kropka, Ta=kreska).
    Separatory: spacja między literami, (stop) między słowami.
    """
    # Wyciągnij część z kodem Morse'a
    morse_match = re.search(r'((?:T[ai])+(?:\s+(?:T[ai])+|\s*\(stop\)\s*)*)', text)
    if not morse_match:
        return ""

    morse_part = morse_match.group(0)
    # Zamień (stop) na separator słów
    words = re.split(r'\(stop\)', morse_part)
    decoded_words = []

    for word in words:
        # Podziel na litery (grupy Ti/Ta oddzielone spacjami)
        letters = word.strip().split()
        decoded_letters = []
        for letter in letters:
            if not letter:
                continue
            # Zamień Ti→. Ta→-
            morse = letter.replace('Ti', '.').replace('Ta', '-')
            decoded_letters.append(MORSE_CODE.get(morse, '?'))
        if decoded_letters:
            decoded_words.append(''.join(decoded_letters))

    return ' '.join(decoded_words)


# =============================================================================
# KOMUNIKACJA Z HUB
# =============================================================================
def send_to_hub(action_data: dict) -> dict:
    """Centralny gateway do API huba."""
    payload = {
        "apikey": HUB_API_KEY,
        "task": "radiomonitoring",
        "answer": action_data,
    }
    print(f"  -> action={action_data.get('action', '?')}")
    resp = requests.post(VERIFY_URL, json=payload, timeout=30)
    data = resp.json()
    print(f"  <- code={data.get('code')}, msg={data.get('message', '')[:80]}")
    return data


# =============================================================================
# DETEKCJA TYPU BINARNEGO — programistyczny router (nie angażujemy LLM)
# =============================================================================
def detect_binary_type(raw_bytes: bytes) -> str:
    """Magic bytes detection — prymityw wielokrotnego użytku."""
    if raw_bytes[:8] == b'\x89PNG\r\n\x1a\n':
        return "image/png"
    if raw_bytes[:3] == b'\xff\xd8\xff':
        return "image/jpeg"
    if raw_bytes[:4] == b'%PDF':
        return "application/pdf"
    if raw_bytes[:4] == b'RIFF' and raw_bytes[8:12] == b'WAVE':
        return "audio/wav"
    if raw_bytes[:3] == b'ID3' or raw_bytes[:2] == b'\xff\xfb':
        return "audio/mp3"
    if raw_bytes[:4] == b'OggS':
        return "audio/ogg"
    if raw_bytes[:4] == b'fLaC':
        return "audio/flac"
    try:
        json.loads(raw_bytes)
        return "application/json"
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass
    try:
        raw_bytes.decode('utf-8')
        return "text/plain"
    except UnicodeDecodeError:
        pass
    return "unknown"


# =============================================================================
# ANALIZA OBRAZÓW — model vision (multimodalność z lekcji)
# =============================================================================
def analyze_image(image_bytes: bytes, mime_type: str) -> str:
    """Wysyła obraz do modelu vision przez gateway OpenRouter."""
    b64 = base64.b64encode(image_bytes).decode()
    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{b64}"},
                    },
                    {
                        "type": "text",
                        "text": (
                            "Opisz dokładnie co widzisz na tym obrazie. "
                            "Wypisz WSZYSTKIE widoczne teksty, liczby, nazwy, numery telefonów. "
                            "Bądź bardzo dokładny."
                        ),
                    },
                ],
            },
        ],
        temperature=0,
        max_tokens=1000,
    )
    return response.choices[0].message.content


# =============================================================================
# TRANSKRYPCJA AUDIO — OpenAI Whisper przez OpenRouter
# =============================================================================
def transcribe_audio(audio_bytes: bytes, mime_type: str) -> str:
    """Transkrybuje audio — zapisuje do pliku tymczasowego i używa Whisper."""
    import tempfile, os

    # Określ rozszerzenie
    ext_map = {"audio/mp3": ".mp3", "audio/mpeg": ".mp3", "audio/wav": ".wav",
               "audio/ogg": ".ogg", "audio/flac": ".flac"}
    ext = ext_map.get(mime_type, ".mp3")

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        # Używamy OpenAI API do Whisper (bezpośrednio, nie przez OpenRouter)
        whisper_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
        # OpenRouter nie obsługuje Whisper - użyjmy innego podejścia
        # Groq oferuje darmowy Whisper
        groq_client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key="placeholder",  # nie mamy klucza Groq
        )
        # Alternatywnie: wyślij audio jako base64 do modelu multimodalnego
        b64 = base64.b64encode(audio_bytes).decode()

        # Gemini obsługuje audio - spróbujmy przez OpenRouter
        response = client.chat.completions.create(
            model="google/gemini-2.0-flash-001",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": b64,
                                "format": "mp3",
                            },
                        },
                        {
                            "type": "text",
                            "text": "Transkrybuj to nagranie audio. Wypisz dokładnie co słyszysz. Bądź bardzo dokładny.",
                        },
                    ],
                },
            ],
            temperature=0,
            max_tokens=1000,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"    [Błąd transkrypcji audio: {e}]")
        # Fallback: spróbuj inny format
        try:
            response = client.chat.completions.create(
                model="google/gemini-2.0-flash-001",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime_type};base64,{b64}"},
                            },
                            {
                                "type": "text",
                                "text": "Transkrybuj to nagranie audio. Wypisz dokładnie co słyszysz.",
                            },
                        ],
                    },
                ],
                temperature=0,
                max_tokens=1000,
            )
            return response.choices[0].message.content
        except Exception as e2:
            return f"[Nie udało się transkrybować audio: {e2}]"
    finally:
        os.unlink(tmp_path)


# =============================================================================
# PRZETWARZANIE BINAREK — router decydujący o ścieżce analizy
# =============================================================================
def process_binary(b64_data: str, meta: str, filesize: int) -> str:
    """
    Router binarny: dekoduj → rozpoznaj typ → skieruj do specjalisty.
    Kluczowy element oszczędzania tokenów (lekcja o routingu danych).
    """
    try:
        raw = base64.b64decode(b64_data)
    except Exception as e:
        return f"[Błąd dekodowania base64: {e}]"

    detected = detect_binary_type(raw)
    print(f"    Typ: meta={meta}, detected={detected}, size={len(raw)}B")

    # JSON — parsuj programistycznie
    if detected == "application/json" or meta == "application/json":
        try:
            data = json.loads(raw)
            return f"[JSON] {json.dumps(data, ensure_ascii=False, indent=2)}"
        except:
            pass

    # Tekst (w tym XML)
    if detected == "text/plain" or meta.startswith("text/"):
        text = raw.decode('utf-8', errors='replace')
        return f"[TEKST/{meta}] {text}"

    # Obrazy
    if detected.startswith("image/"):
        return analyze_image(raw, detected)

    # Audio
    if detected.startswith("audio/"):
        return transcribe_audio(raw, detected)

    return f"[Nieznany: meta={meta}, detected={detected}, {len(raw)}B]"


# =============================================================================
# GŁÓWNA PĘTLA AGENTOWA
# =============================================================================
def main():
    print("=" * 60)
    print("S05E01 — RADIOMONITORING")
    print("=" * 60)

    # KROK 1: Start sesji
    print("\n[1] Start sesji...")
    start_resp = send_to_hub({"action": "start"})

    # KROK 2: Nasłuch — zbieramy WSZYSTKO (Blackboard pattern)
    # Kluczowa zmiana: nie filtrujemy na tym etapie, zbieramy surowe dane.
    # Filtrowanie robi końcowa synteza LLM (wie czego szukać).
    all_materials = []
    listen_count = 0

    print("\n[2] Nasłuch...")

    while True:
        listen_count += 1
        print(f"\n--- #{listen_count} ---")

        resp = send_to_hub({"action": "listen"})
        code = resp.get("code")

        if code != 100:
            print(f"    KONIEC: {resp.get('message', '')}")
            break

        # Transkrypcja tekstowa
        if "transcription" in resp:
            text = resp["transcription"]
            preview = text[:120].replace('\n', ' ')
            print(f"    TXT: {preview}...")

            # Sprawdź czy to kod Morse'a (Ti/Ta pattern)
            if re.search(r'T[ai]T[ai]', text):
                decoded = decode_morse_tati(text)
                if decoded:
                    print(f"    MORSE => {decoded}")
                    all_materials.append(f"[MORSE zdekodowany] {decoded}")

            all_materials.append(f"[Transkrypcja] {text}")

        # Załącznik binarny
        elif "attachment" in resp:
            meta = resp.get("meta", "unknown")
            filesize = resp.get("filesize", 0)
            print(f"    BIN: meta={meta}, size={filesize}")

            result = process_binary(resp["attachment"], meta, filesize)
            all_materials.append(result)
            print(f"    => {result[:150]}...")

        else:
            all_materials.append("[Pusty sygnał]")

    print(f"\n{'=' * 60}")
    print(f"Zebrano materiałów: {len(all_materials)}")
    print(f"{'=' * 60}")

    # KROK 3: Synteza — LLM analizuje WSZYSTKIE zebrane dane
    # Orchestrator pattern: jeden model widzi cały kontekst i syntetyzuje
    print("\n[3] Synteza...")

    # Wypisz wszystkie materiały
    for i, m in enumerate(all_materials):
        print(f"\n--- Materiał {i+1} ---")
        print(m[:500])

    combined = "\n\n===\n\n".join(f"Materiał {i+1}:\n{m}" for i, m in enumerate(all_materials))

    # Jeśli za dużo tekstu, przycinamy mniej istotne części
    if len(combined) > 50000:
        print(f"    Tekst za długi ({len(combined)} znaków), przycinam...")
        combined = combined[:50000]

    synthesis = client.chat.completions.create(
        model="anthropic/claude-sonnet-4",  # lepszy model do syntezy
        messages=[
            {
                "role": "system",
                "content": (
                    "Jesteś analitykiem wywiadu. Na podstawie zebranych materiałów z nasłuchu "
                    "radiowego musisz ustalić 4 informacje o mieście zwanym 'Syjon':\n\n"
                    "1. cityName — prawdziwa nazwa miasta (NIE 'Syjon', to kryptonim)\n"
                    "2. cityArea — powierzchnia miasta w km², zaokrąglona do 2 miejsc po przecinku\n"
                    "3. warehousesCount — liczba magazynów w mieście\n"
                    "4. phoneNumber — numer telefonu osoby kontaktowej\n\n"
                    "Przeanalizuj WSZYSTKIE materiały. Informacje mogą być rozproszone "
                    "po różnych fragmentach. Zwróć uwagę na:\n"
                    "- nazwy miast i ich opisy\n"
                    "- zdekodowane wiadomości Morse'a\n"
                    "- dane z obrazów i plików\n"
                    "- numery telefonów, liczby, powierzchnie\n"
                    "- kontekst: Syjon to miasto wymazane z mapy, z dostępem do wody, polami, bydłem\n\n"
                    "Odpowiedz WYŁĄCZNIE w formacie JSON:\n"
                    '{"cityName": "...", "cityArea": "XX.XX", "warehousesCount": N, "phoneNumber": "..."}\n'
                    "Dodaj krótkie uzasadnienie po JSON."
                ),
            },
            {
                "role": "user",
                "content": combined,
            },
        ],
        temperature=0,
        max_tokens=2000,
    )

    synthesis_text = synthesis.choices[0].message.content.strip()
    print(f"\nSynteza:\n{synthesis_text}")

    # Parsuj JSON
    json_match = re.search(r'\{[^{}]*\}', synthesis_text)
    if json_match:
        report = json.loads(json_match.group())
    else:
        report = json.loads(synthesis_text)

    print(f"\nRaport: {json.dumps(report, ensure_ascii=False, indent=2)}")

    # KROK 4: Transmit
    print("\n[4] Wysyłam raport...")

    final = {
        "action": "transmit",
        "cityName": str(report["cityName"]),
        "cityArea": str(report["cityArea"]),
        "warehousesCount": int(report["warehousesCount"]),
        "phoneNumber": str(report["phoneNumber"]),
    }

    print(f"Payload: {json.dumps(final, ensure_ascii=False, indent=2)}")
    result = send_to_hub(final)
    print(f"\n{'=' * 60}")
    print(f"WYNIK: {json.dumps(result, ensure_ascii=False)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
