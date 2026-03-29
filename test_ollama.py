import requests
import sys

def test_ollama():
    print("1. Testing connection to http://localhost:11434/ ...")
    try:
        response = requests.get("http://localhost:11434/", timeout=2)
        print(f"   Status Code: {response.status_code}")
        if response.status_code == 200:
            print("   ✅ Ollama is running!")
        else:
            print("   ⚠️ Ollama responded with unexpected status.")
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        print("\n   PLEASE RUN 'ollama serve' IN A TERMINAL.")
        return

    print("\n2. Checking available models...")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            if models:
                print(f"   found {len(models)} models:")
                for m in models:
                    print(f"   - {m['name']}")
                
                # Check for 'phi'
                has_phi = any('phi' in m['name'] for m in models)
                if has_phi:
                    print("\n   ✅ 'phi' model is available.")
                else:
                    print("\n   ⚠️ 'phi' model NOT found. Falling back to first available model?")
            else:
                print("   ⚠️ No models found. Please run 'ollama pull phi'.")
        else:
            print(f"   Failed to list models. Status: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Failed to list models: {e}")

    print("\n3. Testing generation with 'phi'...")
    try:
        payload = {
            "model": "phi",
            "prompt": "Say hello!",
            "stream": False
        }
        response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=10)
        if response.status_code == 200:
            print(f"   ✅ Response receive: {response.json().get('response')}")
        else:
            print(f"   ❌ Generation failed: {response.text}")
    except Exception as e:
        print(f"   ❌ Generation error: {e}")

if __name__ == "__main__":
    test_ollama()
