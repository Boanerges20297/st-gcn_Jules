#!/usr/bin/env python3
"""
Health Check Script para Container Docker
Valida saúde da aplicação e endpoints críticos
"""

import sys
import json
import time
import requests
from pathlib import Path
from datetime import datetime

class HealthChecker:
    """Verifica saúde da aplicação"""
    
    def __init__(self, base_url="http://localhost:5000", timeout=5):
        self.base_url = base_url
        self.timeout = timeout
        self.checks = {
            "app_online": False,
            "api_endpoints": False,
            "data_accessible": False,
            "model_loaded": False
        }
        self.errors = []
    
    def check_app_online(self):
        """Verifica se app está online"""
        try:
            response = requests.get(f"{self.base_url}/", timeout=self.timeout)
            if response.status_code == 200:
                self.checks["app_online"] = True
                return True
        except Exception as e:
            self.errors.append(f"App offline: {e}")
        return False
    
    def check_api_endpoints(self):
        """Verifica endpoints críticos"""
        endpoints = [
            "/api/metrics",
            "/api/anomaly_status",
            "/api/explain/1"
        ]
        
        for endpoint in endpoints:
            try:
                response = requests.get(
                    f"{self.base_url}{endpoint}",
                    timeout=self.timeout
                )
                if response.status_code in [200, 400, 503]:
                    self.checks["api_endpoints"] = True
                else:
                    self.errors.append(f"Endpoint {endpoint} retornou {response.status_code}")
                    return False
            except Exception as e:
                self.errors.append(f"Erro em {endpoint}: {e}")
                return False
        
        return True
    
    def check_data_accessible(self):
        """Verifica se dados estão acessíveis"""
        try:
            # Tentar acessar /api/metrics que carrega dados
            response = requests.get(
                f"{self.base_url}/api/metrics",
                timeout=self.timeout
            )
            if response.status_code == 200:
                data = response.json()
                if "metrics" in data:
                    self.checks["data_accessible"] = True
                    return True
        except Exception as e:
            self.errors.append(f"Dados não acessíveis: {e}")
        
        return False
    
    def check_model_loaded(self):
        """Verifica se modelo está carregado"""
        try:
            response = requests.get(
                f"{self.base_url}/api/explain/1",
                timeout=self.timeout
            )
            # Modelo carregado se conseguir gerar explicação
            if response.status_code == 200:
                data = response.json()
                if "summary" in data:
                    self.checks["model_loaded"] = True
                    return True
        except Exception as e:
            self.errors.append(f"Modelo não carregado: {e}")
        
        return False
    
    def run_all_checks(self):
        """Executa todos os checks"""
        print("\n" + "="*60)
        print("HEALTH CHECK - ST-GCN Crime Prediction System")
        print("="*60)
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Target: {self.base_url}")
        print("-" * 60)
        
        # App online é crítico
        print("1. Verificando se app está online...", end=" ", flush=True)
        if self.check_app_online():
            print("✓ OK")
        else:
            print("✗ FALHOU")
            return False
        
        # Endpoints
        print("2. Verificando endpoints API...", end=" ", flush=True)
        if self.check_api_endpoints():
            print("✓ OK")
        else:
            print("✗ FALHOU")
        
        # Dados
        print("3. Verificando acessibilidade de dados...", end=" ", flush=True)
        if self.check_data_accessible():
            print("✓ OK")
        else:
            print("⚠ AVISO")
        
        # Modelo
        print("4. Verificando carregamento do modelo...", end=" ", flush=True)
        if self.check_model_loaded():
            print("✓ OK")
        else:
            print("⚠ AVISO")
        
        print("-" * 60)
        
        # Resumo
        total = len(self.checks)
        passed = sum(1 for v in self.checks.values() if v)
        
        print(f"\nResumo: {passed}/{total} checks passou")
        print(f"Status: {'HEALTHY' if passed >= 3 else 'DEGRADED'}")
        
        if self.errors:
            print(f"\nErros/Avisos:")
            for error in self.errors:
                print(f"  - {error}")
        
        print("="*60 + "\n")
        
        # Retorna sucesso se pelo menos 3 checks passar
        return passed >= 3


def main():
    """Main entry point"""
    
    # Configurar timeout
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            checker = HealthChecker()
            success = checker.run_all_checks()
            
            if success:
                sys.exit(0)  # Sucesso
            
            if attempt < max_retries - 1:
                print(f"Tentativa {attempt + 1} falhou. Aguardando {retry_delay}s...")
                time.sleep(retry_delay)
        
        except Exception as e:
            print(f"Erro no health check: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    
    print("Health check falhou após todas as tentativas")
    sys.exit(1)  # Falha


if __name__ == "__main__":
    main()
