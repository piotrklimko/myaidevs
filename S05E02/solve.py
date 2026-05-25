import os
from dotenv import load_dotenv
load_dotenv()

"""
S05E02 — phonecall
Rozmowa głosowa z operatorem systemu OKO.

=== OPIS ZADANIA ===
Musimy przeprowadzić wieloetapową rozmowę głosową (audio base64 MP3) z operatorem
systemu monitoringu dróg "OKO". Cel:
1. Przedstawić się jako Tymon Gajewski (tożsamość weryfikowana po głosie/imieniu)
2. Zapytać o status trzech dróg: RD224, RD472, RD820
3. Poprosić o wyłączenie monitoringu na przejezdnej drodze (bo transport do bazy Zygfryda)
4. Podać hasło autoryzacyjne "BARBAKAN" gdy operator o nie zapyta

=== ARCHITEKTURA ===
Pipeline: Tekst → TTS (OpenAI gpt-audio) → base64 MP3 → Hub API → odpowiedź audio → STT (Gemini) → tekst

Kluczowe decyzje:
- TTS przez OpenAI gpt-audio na OpenRouter (jedyny model z wyjściem audio na OR)
- Streaming PCM16 24kHz → konwersja ffmpeg → MP3 (bo streaming nie wspiera MP3 bezpośrednio)
- STT przez Gemini 2.5 Flash (input_audio modality) — szybkie i dokładne
- Głos "echo" — najlepiej przechodzi detekcję syntetycznej mowy operatora
- Krótkie, kolokwialne zdania — formalne/długie wypowiedzi są wykrywane jako sztuczne

=== KLUCZOWE WNIOSKI (co działało, co nie) ===

1. DETEKCJA TTS — operator wykrywa syntetyczną mowę (code -790/-810).
   - Głosy sage, shimmer, alloy, coral — wykrywane jako sztuczne
   - Głos "echo" — przechodzi detekcję (brzmi bardziej naturalnie po polsku)
   - gTTS i edge-tts — natychmiast wykrywane

2. NATURALNOŚĆ TEKSTU — równie ważna jak jakość głosu!
   - ZŁE: "Proszę o wyłączenie monitoringu na drodze RD820" (za formalnie)
   - DOBRE: "Możesz wyłączyć monitoring na RD820?" (jak normalna rozmowa)
   - ZŁE: "Jasne, hasło to BARBAKAN" (za rozwlekle)
   - DOBRE: "BARBAKAN!" (krótko, jak człowiek odpowiadający na pytanie)

3. KONWERSJA AUDIO — OpenAI streaming wymaga format: "pcm16" (MP3 niedostępne).
   PCM16 raw 24kHz mono → ffmpeg → MP3 44.1kHz 128kbps. Bez tego hub nie przyjmie audio.

4. FLOW KONWERSACJI — sesja jest stanowa, każda wiadomość zwraca code:
   - 110: start sesji
   - 120: tożsamość potwierdzona
   - 150: status dróg dostarczony
   - -790/-810: wykryto sztuczną mowę (rozmowa spalona, trzeba restart)
   - Ujemne kody = failure, trzeba zacząć nową sesję od "start"

5. RETRY — rozmowa może się "spalić" na dowolnym kroku. Mechanizm ponawiania
   startuje całą sesję od nowa (action: start).

=== FLAGA ===
{FLG:CANYOUHEARME}
"""

import base64
import io
import json
import subprocess
import requests
from pydub import AudioSegment

# --- CONFIG ---
HUB_API_KEY = os.environ["HUB_API_KEY"]
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
VERIFY_URL = "https://hub.ag3nts.org/verify"
STT_MODEL = "google/gemini-2.5-flash"      # STT — transkrypcja odpowiedzi operatora
TTS_MODEL = "openai/gpt-audio"             # TTS — jedyny model z audio output na OpenRouter
TTS_VOICE = "echo"                          # Najlepiej przechodzi detekcję po polsku (testowane: sage, shimmer, alloy, coral)
MAX_RETRIES = 1


def text_to_mp3_base64(text: str) -> str:
    """
    Generuje mowę z tekstu za pomocą OpenAI gpt-audio via OpenRouter.

    Flow: tekst → OpenAI streaming (PCM16 24kHz) → ffmpeg → MP3 base64

    Streaming jest WYMAGANY przez model (bez stream=True zwraca błąd).
    Format PCM16 to jedyny obsługiwany w streaming — MP3/WAV/FLAC nie działają.
    Dlatego potrzebna jest konwersja ffmpeg na końcu.

    System prompt każe modelowi czytać DOKŁADNIE podany tekst — bez tego
    model dodaje od siebie "Oczywiście, oto treść:" itp.
    """
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": TTS_MODEL,
            "modalities": ["text", "audio"],       # Wymagane do aktywacji audio output
            "audio": {"voice": TTS_VOICE, "format": "pcm16"},  # pcm16 = jedyny format w streaming
            "stream": True,                         # Wymagane przez model
            "messages": [
                {"role": "system", "content":
                    "Jesteś telefonistą. Przeczytaj na głos DOKŁADNIE tekst użytkownika. "
                    "Nie dodawaj nic od siebie. Mów po polsku, naturalnie."},
                {"role": "user", "content": f"Przeczytaj: {text}"}
            ],
        },
        timeout=120,
        stream=True
    )

    # Zbieramy chunki audio ze streamu SSE (Server-Sent Events)
    # Każdy chunk zawiera kawałek base64-encoded PCM16 w delta.audio.data
    audio_chunks = []
    transcript = ""
    for line in resp.iter_lines():
        if not line:
            continue
        line = line.decode("utf-8")
        if not line.startswith("data: "):
            continue
        data_str = line[6:]
        if data_str == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            audio = delta.get("audio", {})
            if "data" in audio:
                audio_chunks.append(audio["data"])      # Kawałki PCM16 base64
            if "transcript" in audio:
                transcript += audio["transcript"]       # Tekst rozpoznany przez model
        except Exception:
            pass

    full_pcm_b64 = "".join(audio_chunks)
    if not full_pcm_b64:
        raise RuntimeError("No audio data from TTS")

    pcm_data = base64.b64decode(full_pcm_b64)

    # Konwersja PCM16 24kHz mono → MP3 44.1kHz 128kbps przez ffmpeg
    # Hub API wymaga MP3, a streaming daje tylko surowe PCM16
    proc = subprocess.run(
        ["ffmpeg", "-y", "-f", "s16le", "-ar", "24000", "-ac", "1", "-i", "pipe:0",
         "-ar", "44100", "-ab", "128k", "-f", "mp3", "pipe:1"],
        input=pcm_data, capture_output=True
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {proc.stderr.decode()[:200]}")

    mp3_b64 = base64.b64encode(proc.stdout).decode("utf-8")
    print(f"    [TTS] \"{transcript}\" ({len(mp3_b64)} chars)")
    return mp3_b64


def transcribe_audio(audio_b64: str) -> str:
    """
    Transkrybuje audio odpowiedzi operatora za pomocą Gemini 2.5 Flash.

    Używa input_audio modality — wysyłamy base64 MP3 jako część wiadomości,
    a Gemini zwraca transkrypcję tekstową. Szybkie i dokładne dla polskiego.
    """
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": STT_MODEL,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Transkrybuj nagranie po polsku. Zwróć TYLKO transkrypcję."},
                {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "mp3"}}
            ]}],
            "max_tokens": 1000,
        },
        timeout=120,
    )
    data = resp.json()
    if "error" in data:
        print(f"    [STT ERROR] {data['error']}")
        return ""
    return data["choices"][0]["message"]["content"].strip()


def send_hub(payload: dict) -> dict:
    """Wysyła request do Hub API (hub.ag3nts.org/verify)."""
    resp = requests.post(VERIFY_URL, json=payload, timeout=60)
    return resp.json()


_msg_counter = 0

def send_audio(text: str, step: str) -> dict:
    """
    Pełny cykl jednej wypowiedzi:
    1. Generuje audio TTS z tekstu
    2. Zapisuje nasze audio do pliku MP3 (do odsłuchu/debugowania)
    3. Wysyła do Hub API jako base64
    4. Odbiera odpowiedź operatora (audio + metadata)
    5. Transkrybuje odpowiedź operatora przez Gemini STT
    6. Zapisuje audio operatora do pliku MP3

    Zwraca dict z code, message, transcript i raw response.
    """
    global _msg_counter
    _msg_counter += 1
    print(f"\n  [{step}] \"{text}\"")

    audio_b64 = text_to_mp3_base64(text)

    # Zapisz nasze audio do pliku do odsłuchu — kluczowe przy debugowaniu
    # jakości TTS (czy brzmi naturalnie? czy nie dodaje zbędnych słów?)
    mp3_bytes = base64.b64decode(audio_b64)
    fname = f"msg_{_msg_counter:02d}_{step.replace(' ', '_')}.mp3"
    with open(fname, "wb") as f:
        f.write(mp3_bytes)
    print(f"    [SAVED] {fname} ({len(mp3_bytes)} bytes)")

    data = send_hub({
        "apikey": HUB_API_KEY,
        "task": "phonecall",
        "answer": {"audio": audio_b64}
    })

    code = data.get("code", 0)
    msg = data.get("message", "")
    print(f"    Code: {code} | {msg}")

    transcript = ""
    if "audio" in data:
        # Operator odpowiedział audio — zapisz i transkrybuj
        op_bytes = base64.b64decode(data["audio"])
        op_fname = f"msg_{_msg_counter:02d}_{step.replace(' ', '_')}_operator.mp3"
        with open(op_fname, "wb") as f:
            f.write(op_bytes)
        print(f"    [SAVED] {op_fname}")

        transcript = transcribe_audio(data["audio"])
        print(f"    Operator: \"{transcript}\"")

    return {"code": code, "message": msg, "transcript": transcript, "raw": data}


def find_passable_road(transcript: str) -> str:
    """
    Wyciąga numer przejezdnej drogi z odpowiedzi operatora za pomocą LLM.

    Operator mówi np. "RD472 i RD224 są nieprzejezdne, zostaje RD820".
    Prosty regex by nie wystarczył (różne sformułowania), więc używamy
    claude-haiku do ekstrakcji — szybkie i niezawodne.
    """
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": "anthropic/claude-haiku-4.5",
            "messages": [{"role": "user", "content":
                f"Z wypowiedzi operatora wyciągnij numer drogi która JEST przejezdna (nie nieprzejezdna). "
                f"Odpowiedz TYLKO numerem np. RD820.\n\nWypowiedź: {transcript}"}],
            "max_tokens": 20,
        },
        timeout=30,
    )
    return resp.json()["choices"][0]["message"]["content"].strip()


def is_failed(resp: dict) -> bool:
    """Ujemny code = rozmowa spalona (np. -790 = wykryto sztuczną mowę)."""
    return resp.get("code", 0) < 0


def has_flag(resp: dict) -> str | None:
    """Sprawdza czy w odpowiedzi jest flaga (format FLG:...)."""
    msg = resp.get("message", "")
    if isinstance(msg, str) and "FLG:" in msg.upper():
        return msg
    return None


def step(text: str, label: str) -> dict | None:
    """
    Wysyła wiadomość i obsługuje wynik:
    - Flaga → zwraca dict z kluczem "FLAG"
    - Failure (ujemny code) → zwraca None (sygnał do restartu sesji)
    - OK → zwraca pełny response do dalszego przetwarzania
    """
    resp = send_audio(text, label)
    if f := has_flag(resp["raw"]):
        print(f"\n{'*'*60}\n  FLAGA: {f}\n{'*'*60}")
        return {"FLAG": f}
    if is_failed(resp):
        return None
    return resp


def attempt_conversation() -> str | None:
    """
    Jedna pełna próba rozmowy z operatorem OKO.

    Scenariusz (z lekcji):
    1. Start sesji (action: start)
    2. Przedstawienie — "Halo, tu Tymon Gajewski" → operator potwierdza tożsamość (code 120)
    3. Pytanie o drogi — wymień RD224, RD472, RD820 + wspomnij transport do bazy Zygfryda
       → operator podaje status (code 150), zwykle RD820 jest przejezdna
    4. Prośba o wyłączenie monitoringu na przejezdnej drodze
       → operator pyta o hasło autoryzacyjne
    5. Podanie hasła "BARBAKAN"
       → jeśli OK, dostajemy flagę

    WAŻNE: Każdy krok może się spalić (code < 0) jeśli operator
    wykryje sztuczną mowę. Wtedy zwracamy None i retry startuje od nowa.
    """

    # Start sesji — resetuje stan rozmowy
    start = send_hub({
        "apikey": HUB_API_KEY,
        "task": "phonecall",
        "answer": {"action": "start"}
    })
    print(f"\n  Start: code={start.get('code')}")

    # 1. Przedstawienie — krótkie, naturalne, jak człowiek dzwoniący do biura
    resp = step("Halo, dzień dobry. Tu Tymon Gajewski.", "Przedstawienie")
    if resp is None: return None
    if "FLAG" in resp: return resp["FLAG"]

    # 2. Pytanie o drogi — wspomnij wszystkie trzy + cel (transport do bazy Zygfryda)
    #    Operator sprawdza status i informuje które są przejezdne/nieprzejezdne
    resp = step(
        "Słuchaj, muszę sprawdzić jak wygląda sytuacja na drogach RD224, RD472 i RD820. "
        "Organizujemy transport do jednej z baz Zygfryda i potrzebuję wiedzieć, którędy możemy jechać.",
        "Drogi"
    )
    if resp is None: return None
    if "FLAG" in resp: return resp["FLAG"]

    # Wyciągnij przejezdną drogę z odpowiedzi operatora (LLM extraction)
    road = find_passable_road(resp["transcript"])
    print(f"    >>> Przejezdna: {road}")

    # 3. Prośba o wyłączenie monitoringu — krótko, z uzasadnieniem w jednym zdaniu
    #    To kluczowy moment — operator często wykrywa sztuczną mowę właśnie tu
    resp = step(
        f"Możesz wyłączyć monitoring na {road}? Bo transport do bazy Zygfryda jest tajny.",
        "Monitoring"
    )
    if resp is None: return None
    if "FLAG" in resp: return resp["FLAG"]

    # 4. Odpowiedzi na pytania operatora — adaptacyjne (max 5 rund)
    #    Operator może pytać o: hasło, uzasadnienie, potwierdzenie
    for i in range(5):
        t = resp["transcript"].lower()
        code = resp["code"]
        print(f"    [REACT] code={code}, text={resp['transcript'][:100]}")

        # Rozpoznaj czego operator chce na podstawie słów kluczowych w transkrypcji
        if "hasło" in t or "autoryz" in t or "potwierdz" in t or "uwierzyteln" in t or "kod" in t:
            # Hasło autoryzacyjne — krótko, stanowczo
            resp = step("BARBAKAN!", "Hasło")
        elif "dlaczego" in t or "powód" in t or "po co" in t or "czemu" in t:
            # Uzasadnienie — dlaczego wyłączamy monitoring
            resp = step(
                "Bo to jest transport żywności do jednej z tajnych baz Zygfryda. "
                "Nie mogę zdradzić gdzie dokładnie, więc ta misja nie może być w logach.",
                "Uzasadnienie"
            )
        else:
            # Domyślnie — podaj hasło (najczęściej o to chodzi)
            resp = step("Tak, hasło to BARBAKAN.", "Hasło")

        if resp is None: return None
        if "FLAG" in resp: return resp["FLAG"]

    return None


def main():
    """
    Główna pętla z retry. Rozmowa może się spalić na dowolnym kroku
    (wykrycie TTS), więc ponawiamy całą sesję od nowa.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\n{'='*60}")
        print(f"  PRÓBA {attempt}/{MAX_RETRIES}")
        print(f"{'='*60}")

        try:
            flag = attempt_conversation()
            if flag:
                print(f"\n{'*'*60}")
                print(f"  FLAGA: {flag}")
                print(f"{'*'*60}")
                return
        except Exception as e:
            print(f"  Błąd: {e}")

        print(f"  Ponawiam...")

    print("\nWyczerpano limit prób.")


if __name__ == "__main__":
    main()
