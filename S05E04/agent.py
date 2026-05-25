import os
from dotenv import load_dotenv
load_dotenv()

import requests
import hashlib
import time
import json
import re

# Configuration
API_KEY = os.environ["HUB_API_KEY"]
BASE_URL = "https://hub.ag3nts.org"

class RocketAgent:
    def __init__(self):
        self.state = {"row": 2, "col": 1, "target_row": None}

    def log(self, msg):
        print(f"[*] {msg}")

    def call_api(self, method, endpoint, payload=None):
        """Standardized API handler with retries to overcome random errors."""
        for attempt in range(15):  # Increased retries for stability
            try:
                url = f"{BASE_URL}{endpoint}"
                if method == "GET":
                    r = requests.get(url, params={"key": API_KEY}, timeout=10)
                else:
                    r = requests.post(url, json=payload, timeout=10)
                
                if r and r.status_code == 200:
                    return r
                time.sleep(1)
            except Exception:
                time.sleep(2)
        return None

    def perceive_scanner(self):
        """Extracts data from zniekształcone (distorted) scanner output."""
        r = self.call_api("GET", "/api/frequencyScanner")
        if not r: return None
        
        text = r.text
        if "It's clear!" in text:
            return None
        
        # Using regex to find frequency (numbers) and detectionCode (mixed case string)
        freq_match = re.search(r'(\d{3,4})', text)
        # Look for a string that looks like a code (alphanumeric, 4-10 chars)
        code_match = re.search(r'["\'`]([a-zA-Z0-9]{4,12})["\'`]', text)
        
        if freq_match and code_match:
            return {"frequency": int(freq_match.group(1)), "code": code_match.group(1)}
        return None

    def interpret_hint(self, hint):
        """Maps nautical/radio hints to danger rows."""
        h = hint.lower()
        if any(w in h for w in ["port", "left"]): return 1
        if any(w in h for w in ["starboard", "right"]): return 3
        if any(w in h for w in ["ahead", "bow", "nose", "center", "middle"]): return 2
        return None

    def execute_mission(self):
        self.log("Initializing systems...")
        r = self.call_api("POST", "/verify", {"apikey": API_KEY, "task": "goingthere", "answer": {"command": "start"}})
        if not r: return self.execute_mission() # Recovery: Restart if start fails
        
        start_data = r.json()
        self.state["row"] = start_data["player"]["row"]
        
        while True:
            # 1. Frequency Scanning & Neutralization
            scan = self.perceive_scanner()
            if scan:
                self.log(f"Active tracking detected! Frequency: {scan['frequency']}")
                # SHA1 calculation: detectionCode + "disarm"
                d_hash = hashlib.sha1((scan['code'] + "disarm").encode()).hexdigest()
                self.call_api("POST", "/api/frequencyScanner", {
                    "apikey": API_KEY, 
                    "frequency": scan['frequency'], 
                    "disarmHash": d_hash
                })
            
            # 2. Perception: Radio Hints
            hint_r = self.call_api("POST", "/api/getmessage", {"apikey": API_KEY})
            if not hint_r: continue
            
            rock_row = self.interpret_hint(hint_r.json().get('hint', ''))
            
            # 3. Reasoning: Movement Logic
            curr = self.state["row"]
            if rock_row == curr:
                # Avoid rock while staying in bounds 1-3
                if curr == 1: move, next_row = "right", 2
                elif curr == 3: move, next_row = "left", 2
                else: move, next_row = "left", 1
            else:
                move, next_row = "go", curr

            # 4. Action & Persistence
            self.log(f"Action: {move} | Current Row: {curr} | Rock Row: {rock_row}")
            res_r = self.call_api("POST", "/verify", {"apikey": API_KEY, "task": "goingthere", "answer": {"command": move}})
            
            if not res_r: 
                self.log("API Lost. Attempting recovery...")
                continue
                
            res = res_r.json()
            if "flag" in str(res):
                self.log(f"GRUDZIĄDZ REACHED: {res.get('flag')}")
                break
            if res.get("crashed"):
                self.log("Rocket destroyed. Initializing new run...")
                return self.execute_mission()
            
            self.state["row"] = res["player"]["row"]
            time.sleep(0.1)

if __name__ == "__main__":
    agent = RocketAgent()
    agent.execute_mission()