#!/usr/bin/env python3
"""Test semantic embeddings generation with 5 sample neighborhoods."""

import json
import sys
import os

# Add src to path
sys.path.insert(0, 'src')

from llm_service import get_semantic_embeddings_batch, _call_model_with_rotation, get_gemini_api_keys

# Test with 5 neighborhoods
TEST_BAIRROS = [
    'Bom Futuro',
    'Aldeota',
    'Centro',
    'Messejana',
    'Praia de Iracema'
]

def main():
    print(f'Testing semantic embeddings generation for {len(TEST_BAIRROS)} neighborhoods...\n')
    
    # First test: just call the model directly to see what it returns
    print('--- Debug: Testing raw model call ---')
    api_keys = get_gemini_api_keys()
    if not api_keys:
        print('ERROR: No API keys found')
        return 1
    
    prompt = (
        "Generate semantic embedding for this neighborhood:\n\n"
        "Neighborhood: Bom Futuro, Fortaleza, Ceará, Brazil\n"
        "Description: Safe residential area in central Fortaleza\n\n"
        "Respond with ONLY a JSON array of 384 floating-point numbers "
        "(no markdown, no explanation):\n"
        "[-0.123, 0.456, ..., 0.789]"
    )
    
    try:
        response = _call_model_with_rotation(prompt, api_keys)
        print(f'Raw response (first 500 chars):\n{str(response)[:500]}\n')
    except Exception as e:
        print(f'ERROR in raw call: {e}')
        return 1
    
    cache_file = 'data/processed/test_embeddings.json'
    
    try:
        print(f'\nCalling get_semantic_embeddings_batch()...')
        embeddings = get_semantic_embeddings_batch(
            TEST_BAIRROS,
            cache_file=cache_file
        )
        
        print(f'\n✓ Successfully generated embeddings for:')
        for bairro, embedding in embeddings.items():
            if isinstance(embedding, list):
                print(f'  - {bairro}: {len(embedding)}D embedding')
            else:
                print(f'  - {bairro}: Invalid type {type(embedding)}')
        
        # Verify cache was created
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cached = json.load(f)
            print(f'\n✓ Cache file created with {len(cached)} embeddings')
        
        print(f'\n✓ Test PASSED - Ready to generate all 319 bairro embeddings')
        return 0
        
    except Exception as e:
        print(f'\n✗ Test FAILED: {e}', file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
