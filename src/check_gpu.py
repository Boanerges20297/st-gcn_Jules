"""
Diagnóstico de suporte a GPU Intel
"""
import torch
import sys

print("=" * 60)
print("GPU INTEL DIAGNÓSTICO")
print("=" * 60)

print(f"\n1. PyTorch versão: {torch.__version__}")
print(f"2. CUDA disponível: {torch.cuda.is_available()}")

# Tentar IPEX
try:
    import intel_extension_for_pytorch as ipex
    print(f"3. IPEX instalada: SIM")
    print(f"   IPEX versão: {ipex.__version__}")
    
    # Tentar XPU
    try:
        if hasattr(torch, 'xpu'):
            print(f"4. torch.xpu disponível: SIM")
            if torch.xpu.is_available():
                print(f"5. GPU XPU detectada: SIM")
                print(f"   Devices: {torch.xpu.device_count()}")
            else:
                print(f"5. GPU XPU detectada: NÃO")
                print("   -> Driver Intel Arc GPU pode não estar instalado corretamente")
        else:
            print(f"4. torch.xpu disponível: NÃO")
    except Exception as e:
        print(f"4. Erro ao verificar XPU: {e}")
        
except ImportError as e:
    print(f"3. IPEX instalada: NÃO ({e})")
    print("   -> Instale com: pip install intel-extension-for-pytorch")

print("\n" + "=" * 60)
print("RECOMENDAÇÃO:")
print("=" * 60)
print("GPU Intel Iris não detectada via IPEX/XPU.")
print("Usando CPU com batch_size=512 para treino rápido.")
print("=" * 60 + "\n")
