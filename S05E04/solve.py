import os
from dotenv import load_dotenv
load_dotenv()

import requests
import hashlib
import json
import re
import time
from openai import OpenAI

HUB_API_KEY = os.environ["HUB_API_KEY"]
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
BASE = "https://hub.ag3nts.org"

llm = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

def http(method, url, body=None, retries=30):
    for i in range(retries):
        try:
            r = requests.request(method, url, json=body, timeout=15)
            if r.status_code == 200:
                return r
            if r.status_code == 429:
                time.sleep(5)
                continue
            if r.status_code == 400:
                return r
            time.sleep(3)
        except:
            time.sleep(3)
    return None

def send_cmd(cmd):
    r = http("POST", f"{BASE}/verify",
             {"apikey": HUB_API_KEY, "task": "goingthere", "answer": {"command": cmd}})
    if r:
        d = r.json()
        print(f"  [{cmd}] st={r.status_code} {json.dumps(d, ensure_ascii=False)[:300]}")
        return d
    print(f"  [{cmd}] NO RESPONSE")
    return None

def check_and_disarm():
    """Check freq scanner, disarm if tracked. Returns True if safe to proceed."""
    for attempt in range(10):
        r = http("GET", f"{BASE}/api/frequencyScanner?key={HUB_API_KEY}", retries=3)
        if not r or r.status_code != 200:
            time.sleep(3)
            continue

        text = r.text.strip()
        # Print more for debug
        print(f"  scan[{len(text)}]: {text[:300]}")

        # Check if clear (handles mangled spelling)
        if re.search(r'cl[e]+a?r', text.lower().replace(" ", "")):
            return True

        # Extract frequency - find any standalone number (not in a string)
        freq = None
        m = re.search(r'["\']?\w*[Ff][Rr][Ee3]\w{0,5}[Nn][Cc][Yy]\w*["\']?\s*:\s*(\d+)', text)
        if m:
            freq = int(m.group(1))
        if not freq:
            # Fallback: find any number that looks like a frequency (3-4 digits)
            nums = re.findall(r':\s*(\d{2,4})\b', text)
            if nums:
                freq = int(nums[0])

        # Extract detectionCode - look for value after a key containing "tect" + "code" pattern
        code = None
        # Match key:value where key resembles detectionCode (with d/b, o/0 substitutions)
        # Key pattern: something with "tect" and "c0de"/"code" in it
        m = re.search(r'[`"\']?\w*[Tt][Ee3][Cc][Tt]\w*[Cc][0Oo][BbDd][Ee3]\w*[`"\']?\s*:\s*[`"\'](\w+)[`"\']', text)
        if m:
            code = m.group(1)
        if not code:
            # Fallback: find all key:"value" pairs, pick short random-looking value
            pairs = re.findall(r'[`"\']?\w+[`"\']?\s*:\s*[`"\']([\w]{3,15})[`"\']', text)
            skip = {"self", "guided", "missile", "surface", "air", "true", "false"}
            for v in pairs:
                if v.lower() in skip or any(w in v.lower() for w in skip):
                    continue
                # Detection codes are short (4-8 chars) mixed case
                if 4 <= len(v) <= 10 and re.search(r'[A-Z]', v) and re.search(r'[a-z]', v):
                    code = v
                    break

        if freq and code:
            h = hashlib.sha1((code + "disarm").encode()).hexdigest()
            print(f"  DISARM freq={freq} code={code} hash={h}")
            dr = http("POST", f"{BASE}/api/frequencyScanner",
                      {"apikey": HUB_API_KEY, "frequency": freq, "disarmHash": h})
            if dr:
                print(f"  disarm resp: {dr.text[:200]}")
                if dr.status_code == 200:
                    return True
            # If disarm failed, retry scanner
            time.sleep(2)
            continue

        print(f"    parse fail: freq={freq} code={code}")
        time.sleep(3)

    print("  WARNING: scanner failed after retries")
    return False

def get_hint():
    r = http("POST", f"{BASE}/api/getmessage", {"apikey": HUB_API_KEY})
    if r and r.status_code == 200:
        try:
            hint = r.json().get("hint", "")
            if hint:
                print(f"  hint: {hint}")
                return hint
        except: pass
    return None

def interpret_hint(hint):
    """Use LLM with few-shot examples to determine rock direction."""
    prompt = """You help navigate a rocket. Each hint describes where a ROCK is in the next column.
Directions: port/left = UP, starboard/right = DOWN, ahead/bow/nose/front/center/middle/straight/trajectory = SAME ROW.

The hint says which direction is DANGEROUS (has rock) or which are SAFE (no rock).
Answer with ONE word for where the ROCK is: LEFT, RIGHT, or AHEAD.

Examples:
"Watch the starboard side" → RIGHT
"The obstacle is resting beside starboard" → RIGHT
"The rock is crowding starboard" → RIGHT
"Port and bow are clear, starboard has risk" → RIGHT
"Watch the port side instead" → LEFT
"The rock has taken up position by port" → LEFT
"The danger is posted on the port side" → LEFT
"The route through the middle ends at a rock" → AHEAD
"The problem is sitting straight ahead" → AHEAD
"The forward corridor is closed by a rock" → AHEAD
"The nose of the rocket is pointed at trouble" → AHEAD
"The bow faces a rock" → AHEAD
"The solid mass is planted directly before the nose" → AHEAD
"The center is occupied by a rock" → AHEAD
"The rock is on the same line as your trajectory" → AHEAD
"The hazard is waiting in the exact path of the bow" → AHEAD
"Continuing without turning would send you into the rock" → AHEAD
"No warning lights show on the sides. The central path is blocked" → AHEAD

Now answer for this hint:
\"""" + hint + """\"
Answer:"""

    try:
        resp = llm.chat.completions.create(
            model="anthropic/claude-haiku-4.5",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5, temperature=0)
        ans = resp.choices[0].message.content.strip().upper()
        for w in ["LEFT", "RIGHT", "AHEAD"]:
            if w in ans:
                print(f"    rock={w}")
                return w.lower()
    except Exception as e:
        print(f"    llm err: {e}")
    return "ahead"

def pick_move(rock, row):
    if rock == "ahead":
        return ("left", row - 1) if row > 1 else ("right", row + 1)
    elif rock == "left":  # rock at row-1
        return ("right", row + 1) if row < 3 else ("go", row)
    elif rock == "right":  # rock at row+1
        return ("left", row - 1) if row > 1 else ("go", row)
    return ("go", row)

def play():
    result = send_cmd("start")
    if not result or result.get("code", 0) < 0:
        return False

    row = result["player"]["row"]
    col = result["player"]["col"]
    base_col = result["base"]["col"]
    print(f"  start: row={row} col={col} target={base_col}")

    while col < base_col:
        print(f"\n--- Col {col}, Row {row} ---")
        time.sleep(0.5)

        # 1. Scanner
        ok = check_and_disarm()
        if not ok:
            print("  scanner fail, continuing anyway...")
        time.sleep(0.5)

        # 2. Hint
        hint = get_hint()
        if not hint:
            print("  no hint!")
            return False
        time.sleep(0.3)

        # 3. Choose move
        rock = interpret_hint(hint)
        move, new_row = pick_move(rock, row)

        # Bounds safety
        if new_row < 1 or new_row > 3:
            move, new_row = "go", row

        print(f"  => {move} (row {row}->{new_row})")
        time.sleep(0.5)

        result = send_cmd(move)
        if not result:
            return False

        code = result.get("code", 0)
        if code < 0 or result.get("crashed"):
            print(f"  CRASH: {json.dumps(result)[:200]}")
            return False

        if "flag" in json.dumps(result).lower():
            print(f"\n*** FLAG: {json.dumps(result)}")
            return True

        row = result.get("player", {}).get("row", new_row)
        col = result.get("player", {}).get("col", col + 1)

    return True

print("=== Rocket Navigation ===")
for attempt in range(50):
    print(f"\n{'='*50} ATTEMPT {attempt+1} {'='*50}")
    if play():
        break
    time.sleep(3)
