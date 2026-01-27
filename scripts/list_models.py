#!/usr/bin/env python
import os
from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai

api_key = os.environ.get('GEMINI_API_KEY')
genai.configure(api_key=api_key)

print("Available models:")
for m in genai.list_models():
    print(f"  - {m.name}")
    print(f"    Display name: {m.display_name}")
    print(f"    Version: {m.version}")
    if hasattr(m, 'supported_generation_methods'):
        print(f"    Methods: {m.supported_generation_methods}")
    print()
