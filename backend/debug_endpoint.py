import requests
import sys

URL = "http://127.0.0.1:8000/api/speak"
TEXT = "Testing endpoint audio."

try:
    print(f"Sending request to {URL}...")
    resp = requests.post(URL, json={"text": TEXT})
    
    print(f"Status Code: {resp.status_code}")
    if resp.status_code == 200:
        content_len = len(resp.content)
        print(f"Received {content_len} bytes")
        if content_len > 1000:
            with open("endpoint_test.wav", "wb") as f:
                f.write(resp.content)
            print("✅ Saved endpoint_test.wav")
        else:
            print(f"❌ Response too small! Content: {resp.content[:100]}")
    else:
        print(f"❌ Request failed: {resp.text}")

except Exception as e:
    print(f"❌ Error: {e}")
