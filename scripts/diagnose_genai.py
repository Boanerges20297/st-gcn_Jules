import sys
import os

print("=== Google GenAI SDK Diagnostic ===\n")

# Check what's installed
print("1. Checking installed packages...")
try:
    import google
    print(f"   google package: {google.__version__ if hasattr(google, '__version__') else 'installed'}")
    print(f"   google.__path__: {google.__path__}")
except ImportError:
    print("   google: NOT installed")

try:
    import google.genai
    print(f"   google.genai: installed")
    print(f"   Attributes: {dir(google.genai)[:10]}...")
except ImportError:
    print("   google.genai: NOT installed")

try:
    import google.generativeai
    print(f"   google.generativeai: installed")
    print(f"   Attributes: {dir(google.generativeai)[:10]}...")
except ImportError:
    print("   google.generativeai: NOT installed")

print("\n2. Trying different import patterns...")

# Try new SDK
try:
    from google import genai
    print(f"   from google import genai: OK")
    print(f"   genai.Client: {hasattr(genai, 'Client')}")
    print(f"   genai.models: {hasattr(genai, 'models')}")
    print(f"   dir(genai): {[x for x in dir(genai) if not x.startswith('_')][:15]}")
except Exception as e:
    print(f"   from google import genai: FAILED - {e}")

# Try old SDK
try:
    import google.generativeai as genai_old
    print(f"   import google.generativeai as genai_old: OK")
    print(f"   genai_old.GenerativeModel: {hasattr(genai_old, 'GenerativeModel')}")
    print(f"   genai_old.configure: {hasattr(genai_old, 'configure')}")
    print(f"   dir(genai_old)[:15]: {[x for x in dir(genai_old) if not x.startswith('_')][:15]}")
except Exception as e:
    print(f"   import google.generativeai: FAILED - {e}")

print("\n3. Testing API key...")
api_key = os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY')
if api_key:
    print(f"   API key found: {api_key[:20]}...")
else:
    print("   No API key found")

print("\n4. Attempting actual call...")
try:
    from google import genai
    client = genai.Client(api_key=api_key)
    print(f"   Client created: {client}")
    print(f"   client.models: {hasattr(client, 'models')}")
    if hasattr(client, 'models'):
        print(f"   client.models.generate_content: {hasattr(client.models, 'generate_content')}")
except Exception as e:
    print(f"   Failed: {e}")

print("\nDone.")
