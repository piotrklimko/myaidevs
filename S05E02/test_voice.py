import os
from dotenv import load_dotenv
load_dotenv()

"""Test nagrań TTS — porównanie różnych stylów."""

import base64
import json
import subprocess
import requests

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
TTS_MODEL = "openai/gpt-audio"

samples = [
    ("sage_natural", "sage",
     "Jesteś zwykłym Polakiem rozmawiającym przez telefon. Mów naturalnie, spokojnie, z drobnymi pauzami.",
     "Halo, dzień dobry. Tu Tymon Gajewski."),
    ("sage_casual", "sage",
     "Mów po polsku, bardzo swobodnie, jakbyś gadał ze znajomym. Nie bądź perfekcyjny.",
     "Ej, słuchaj, muszę sprawdzić co tam z drogami. RD224, RD472 i RD820. Wiecie coś?"),
    ("shimmer_natural", "shimmer",
     "Jesteś zwykłym Polakiem rozmawiającym przez telefon. Mów naturalnie, spokojnie.",
     "Halo, dzień dobry. Tu Tymon Gajewski."),
    ("alloy_natural", "alloy",
     "Jesteś zwykłym Polakiem rozmawiającym przez telefon. Mów naturalnie, spokojnie.",
     "Halo, dzień dobry. Tu Tymon Gajewski."),
]

for name, voice, system_prompt, text in samples:
    print(f"\nGeneruję: {name} (voice={voice})...")
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": TTS_MODEL,
            "modalities": ["text", "audio"],
            "audio": {"voice": voice, "format": "pcm16"},
            "stream": True,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Przeczytaj: {text}"}
            ],
        },
        timeout=120,
        stream=True
    )

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
                audio_chunks.append(audio["data"])
            if "transcript" in audio:
                transcript += audio["transcript"]
        except Exception:
            pass

    full_pcm_b64 = "".join(audio_chunks)
    if not full_pcm_b64:
        print(f"  BRAK AUDIO!")
        continue

    pcm_data = base64.b64decode(full_pcm_b64)
    proc = subprocess.run(
        ["ffmpeg", "-y", "-f", "s16le", "-ar", "24000", "-ac", "1", "-i", "pipe:0",
         "-ar", "44100", "-ab", "128k", "-f", "mp3", "pipe:1"],
        input=pcm_data, capture_output=True
    )
    if proc.returncode != 0:
        print(f"  ffmpeg error: {proc.stderr.decode()[:200]}")
        continue

    fname = f"test_{name}.mp3"
    with open(f"/workspaces/aidevs/S05E02/{fname}", "wb") as f:
        f.write(proc.stdout)
    print(f"  OK: {fname} ({len(proc.stdout)} bytes) — transcript: \"{transcript}\"")

print("\nGotowe! Pliki w S05E02/")
