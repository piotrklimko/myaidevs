import os
from dotenv import load_dotenv
load_dotenv()

import requests

HUB_API_KEY = os.environ["HUB_API_KEY"]

resp = requests.post(
    "https://hub.ag3nts.org/verify",
    json={
        "apikey": HUB_API_KEY,
        "task": "proxy",
        "answer": {
            "url": "https://azyl-18356.ag3nts.org",
            "sessionID": "sesja123"
        }
    }
)
print(resp.text)