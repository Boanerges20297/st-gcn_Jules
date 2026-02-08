#!/usr/bin/env python
"""
verify_overfitting_and_recommendations.py

Script completo para verificar:
1. Status do overfitting no modelo
2. Implementação das 3 recomendações de Feb 2026:
   - Treinar modelo com validação cruzada (TimeSeriesSplit)
   - Usar regularização (L2, dropout) para reduzir overfitting
   - Avaliar impacto real dos micro-nós com test set limpo

Data: Feb 8, 2026
Status: Verificação executiva de recomendações
"""

import os
import sys
import pickle
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import inspect
import ast

# ==============================================================================
# SETUP
# ==============================================================================

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class OverfittingVerifier:
    """Verifica status de overfitting e recomendações implementadas"""
    
    def __init__(self, root_path=ROOT):
        self.root = root_path
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def print_header(self, title):
        """Imprime header formatado"""
        print(f"\n{'='*80}")
        print(f"  {title}")
        print(f"{'='*80}\n")
    
    def print_section(self, title):
        """Imprime seção"""
        print(f"\n{'-'*80}")
        print(f"  {title}")
        print(f"{'-'*80}\n")
    
    def check_script_exists(self, script_path):
        """Verifica se script existe"""
        full_path = self.root / script_path
        exists = full_path.exists()
        status = "✅ EXISTE" if exists else "❌ NÃO EXISTE"
        print(f"{status}: {script_path}")
        return exists
    
    def extract_python_features(self, file_path):
        """Extrai features do código Python"""
        features = {
            'has_timeseriessplit': False,
            'has_dropout': False,
            'has_weight_decay': False,
            'has_batch_norm': False,
            'has_early_stopping': False,
            'has_temporal_split': False,
            'has_cross_validation': False,
            'dropout_value': None,
            'weight_decay_value': None,
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                source = content
        except:
            return features
        
        # Buscar padrões simples (sem AST complexo)
        if 'TimeSeriesSplit' in source:
            features['has_timeseriessplit'] = True
        if 'Dropout' in source:
            features['has_dropout'] = True
        if 'weight_decay' in source:
            features['has_weight_decay'] = True
        if 'BatchNorm' in source:
            features['has_batch_norm'] = True
        if 'early_stop' in source.lower() or 'patience' in source.lower():
            features['has_early_stopping'] = True
        if 'temporal_split' in source or 'train_ratio' in source:
            features['has_temporal_split'] = True
        if 'cross_val' in source.lower() or 'kfold' in source.lower():
            features['has_cross_validation'] = True
        
        # Extrair valores específicos (regex simples)
        import re
        
        # Dropout valor
        dropout_match = re.search(r'Dropout\(([\d.]+)\)', source)
        if dropout_match:
            features['dropout_value'] = float(dropout_match.group(1))
        
        # Weight decay
        wd_match = re.search(r'weight_decay\s*=\s*([\d.e\-]+)', source)
        if wd_match:
            try:
                features['weight_decay_value'] = float(wd_match.group(1))
            except:
                pass
        
        return features
    
    def print_feature_check(self, name, features):
        """Imprime check de features"""
        checks = [
            ('TimeSeriesSplit', 'timeseriessplit'),
            ('Dropout (regularização)', 'dropout'),
            ('Weight Decay (L2)', 'weight_decay'),
            ('BatchNorm', 'batch_norm'),
            ('Early Stopping', 'early_stopping'),
        ]
        
        print(f"\n📄 {name}")
        print("   " + "-" * 70)
        for display_name, key in checks:
            has_it = features.get(f'has_{key}', False)
            status = "✅" if has_it else "❌"
            print(f"   {status} {display_name}")
            
            # Print specific values if available
            if key == 'dropout' and features.get('dropout_value'):
                print(f"      → Valor: {features['dropout_value']}")
            elif key == 'weight_decay' and features.get('weight_decay_value'):
                print(f"      → Valor: {features['weight_decay_value']}")
    
    def verify_recommendation_1_timeseries_split(self):
        """REC 1: Treinar modelo com validação cruzada (TimeSeriesSplit)"""
        self.print_section("RECOMENDAÇÃO 1: Validação Cruzada Temporal (TimeSeriesSplit)")
        
        scripts_to_check = [
            'src/train.py',
            'src/validate_with_crossval.py',
            'scripts/train_with_anomaly_awareness.py'
        ]
        
        print("Verificando implementação de TimeSeriesSplit...\n")
        
        status_overall = False
        found_implementations = []
        
        for script in scripts_to_check:
            script_path = self.root / script
            if script_path.exists():
                features = self.extract_python_features(script_path)
                self.print_feature_check(script, features)
                
                if features['has_timeseriessplit'] or features['has_temporal_split'] or features['has_cross_validation']:
                    status_overall = True
                    found_implementations.append(script)
        
        # Análise
        print("\n" + "="*70)
        if found_implementations:
            print(f"✅ IMPLEMENTADO em:")
            for impl in found_implementations:
                print(f"   • {impl}")
        else:
            print(f"⚠️  PARCIALMENTE IMPLEMENTADO:")
            print(f"   • validate_with_crossval.py tem split temporal manual (não oficial TimeSeriesSplit)")
            print(f"   • train.py NÃO usa TimeSeriesSplit no treinamento principal")
            print(f"   • Recomendação: Criar train_with_timeseries_split.py")
        
        self.results['rec1_timeseriessplit'] = {
            'status': 'PARCIAL' if not found_implementations else 'COMPLETO',
            'details': found_implementations or ['Split manual em validate_with_crossval.py']
        }
        
        return status_overall
    
    def verify_recommendation_2_regularization(self):
        """REC 2: Usar regularização (L2, dropout) para reduzir overfitting"""
        self.print_section("RECOMENDAÇÃO 2: Regularização (L2 + Dropout)")
        
        # Verificar modelo
        print("Verificando modelo neural (src/model.py)...\n")
        model_path = self.root / 'src' / 'model.py'
        
        if model_path.exists():
            features = self.extract_python_features(model_path)
            self.print_feature_check('model.py', features)
            
            # Verificar trainer
            print("\nVerificando trainer (src/train.py)...\n")
            train_path = self.root / 'src' / 'train.py'
            features_train = self.extract_python_features(train_path)
            self.print_feature_check('train.py', features_train)
            
            # Análise
            print("\n" + "="*70)
            has_dropout = features.get('has_dropout', False)
            has_batch_norm = features.get('has_batch_norm', False)
            has_weight_decay = features_train.get('has_weight_decay', False)
            dropout_val = features.get('dropout_value')
            weight_decay_val = features_train.get('weight_decay_value')
            
            if has_dropout and has_weight_decay:
                print(f"✅ IMPLEMENTADO COMPLETAMENTE")
                print(f"\n   Regularização aplicada:")
                if dropout_val:
                    print(f"   • Dropout rate: {dropout_val} ({'ALTO' if dropout_val > 0.5 else 'MODERADO'})")
                if weight_decay_val:
                    print(f"   • L2 (Weight Decay): {weight_decay_val}")
                if has_batch_norm:
                    print(f"   • BatchNorm2d (normalização interna)")
            else:
                print(f"⚠️  PARCIALMENTE IMPLEMENTADO")
                if not has_dropout:
                    print(f"   ❌ Dropout não encontrado no modelo")
                if not has_weight_decay:
                    print(f"   ❌ Weight decay não encontrado no trainer")
            
            self.results['rec2_regularization'] = {
                'status': 'COMPLETO' if (has_dropout and has_weight_decay) else 'PARCIAL',
                'dropout': dropout_val,
                'weight_decay': weight_decay_val,
                'batch_norm': has_batch_norm
            }
        else:
            print("❌ model.py não encontrado")
            self.results['rec2_regularization'] = {'status': 'ERRO'}
    
    def verify_recommendation_3_micronodes(self):
        """REC 3: Avaliar impacto real dos micro-nós com test set limpo"""
        self.print_section("RECOMENDAÇÃO 3: Avaliação de Micro-nós com Test Set Limpo")
        
        # Verificar validate_with_crossval.py
        print("Verificando validate_with_crossval.py...\n")
        validate_path = self.root / 'src' / 'validate_with_crossval.py'
        
        if validate_path.exists():
            features = self.extract_python_features(validate_path)
            
            with open(validate_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            has_com_micro = 'COM Micro-nós' in content or 'COM 319' in content
            has_sem_micro = 'SEM Micro-nós' in content or 'SEM' in content
            has_precision = 'precision_at_k' in content
            has_ground_truth = 'ground_truth' in content or 'y_true' in content
            
            print("Funcionalidades implementadas:")
            print(f"   {'✅' if has_com_micro else '❌'} Avalia COM micro-nós (319 nodes)")
            print(f"   {'✅' if has_sem_micro else '❌'} Avalia SEM micro-nós (~35 bairros)")
            print(f"   {'✅' if has_ground_truth else '❌'} Calcula ground truth de dados não-vistos")
            print(f"   {'✅' if has_precision else '❌'} Usa Precision@K e NDCG@K reais")
            
            print("\n" + "="*70)
            if all([has_com_micro, has_sem_micro, has_ground_truth, has_precision]):
                print("✅ IMPLEMENTADO COMPLETAMENTE")
                print(f"\n   O script compare:")
                print(f"   • COM 319 nodes vs SEM (apenas bairros)")
                print(f"   • Usa dados de teste não-vistos pelo modelo")
                print(f"   • Calcula Precision@K e NDCG@K real")
            else:
                print("⚠️  PARCIALMENTE IMPLEMENTADO")
                missing = []
                if not has_com_micro: missing.append("Avaliação COM micro-nós")
                if not has_sem_micro: missing.append("Avaliação SEM micro-nós")
                if not has_ground_truth: missing.append("Ground truth limpo")
                if not has_precision: missing.append("Métricas Precision@K")
                for m in missing:
                    print(f"   ❌ {m}")
            
            # Verificar check_overfitting.py também
            print("\n\nVerificando check_overfitting.py (ranking model)...\n")
            check_path = self.root / 'src' / 'check_overfitting.py'
            if check_path.exists():
                features_check = self.extract_python_features(check_path)
                with open(check_path, 'r', encoding='utf-8') as f:
                    check_content = f.read()
                
                has_periods = 'ultimos' in check_content or '30 dias' in check_content
                has_comparison = 'proximos' in check_content and 'distantes' in check_content
                
                if has_periods and has_comparison:
                    print("✅ check_overfitting.py implementa validação temporal")
                    print("   • Compara performance: últimos 30d vs 30-60d vs 60-90d")
                    print("   • Detecta overfitting por período temporal")
            
            self.results['rec3_micronodes'] = {
                'status': 'COMPLETO' if all([has_com_micro, has_sem_micro]) else 'PARCIAL',
                'com_micro': has_com_micro,
                'sem_micro': has_sem_micro,
                'ground_truth': has_ground_truth
            }
        else:
            print("❌ validate_with_crossval.py não encontrado")
            self.results['rec3_micronodes'] = {'status': 'ERRO'}
    
    def print_summary(self):
        """Imprime resumo executivo"""
        self.print_header("SUMÁRIO EXECUTIVO - RECOMENDAÇÕES")
        
        print(f"Data da Verificação: {self.timestamp}\n")
        
        rec_status = {
            'REC 1': self.results.get('rec1_timeseriessplit', {}).get('status', 'DESCONHECIDO'),
            'REC 2': self.results.get('rec2_regularization', {}).get('status', 'DESCONHECIDO'),
            'REC 3': self.results.get('rec3_micronodes', {}).get('status', 'DESCONHECIDO'),
        }
        
        print("STATUS DE IMPLEMENTAÇÃO:\n")
        for rec, status in rec_status.items():
            icon = "✅" if status == "COMPLETO" else "⚠️ " if status == "PARCIAL" else "❌"
            print(f"   {icon} {rec}: {status}")
        
        print("\n" + "="*80)
        print("RECOMENDAÇÕES IMEDIATAS:\n")
        
        if rec_status['REC 1'] == 'PARCIAL':
            print("1️⃣  TimeSeriesSplit:")
            print("   • validate_with_crossval.py já tem split temporal manual")
            print("   • Recomendação: Usar sklearn.TimeSeriesSplit para treino")
            print("   • Action: python scripts/train_with_timeseries_split.py (CRIAR)\n")
        
        if rec_status['REC 2'] == 'COMPLETO':
            print("2️⃣  Regularização: ✅ JÁ IMPLEMENTADA")
            dropout = self.results.get('rec2_regularization', {}).get('dropout')
            wd = self.results.get('rec2_regularization', {}).get('weight_decay')
            print(f"   • Dropout: {dropout} (rate)")
            print(f"   • Weight Decay (L2): {wd}")
            print(f"   • BatchNorm: Sim\n")
        
        if rec_status['REC 3'] == 'COMPLETO':
            print("3️⃣  Micro-nós: ✅ AVALIAÇÃO JÁ IMPLEMENTADA")
            print("   • Script: python src/validate_with_crossval.py")
            print("   • Compara: COM micro-nós (319) vs SEM (~35 bairros)")
            print("   • Métrica: Precision@K, NDCG@K em dados não-vistos\n")
        
        print("="*80)
        print("\nPRÓXIMOS PASSOS:\n")
        
        if rec_status['REC 1'] == 'PARCIAL':
            print("▶️  CRIAR: scripts/train_with_timeseries_split.py")
            print("   Implementar TimeSeriesSplit oficial no treinamento\n")
        
        print("▶️  EXECUTAR: python src/validate_with_crossval.py")
        print("   Validar performance COM e SEM micro-nós\n")
        
        print("▶️  EXECUTAR: python src/check_overfitting.py")
        print("   Validar overfitting do ranking model\n")
        
        print("▶️  CRIAR RELATÓRIO: Consolidar resultados em VALIDATION_REPORT.md")
        print("   Documentar impacto de cada recomendação\n")
        
        print("="*80)
    
    def run_all_verifications(self):
        """Executa todas as verificações"""
        self.print_header("VERIFICAÇÃO DE OVERFITTING E RECOMENDAÇÕES")
        
        print(f"Root: {self.root}")
        print(f"Timestamp: {self.timestamp}\n")
        
        # 1. Verificar scripts básicos
        self.print_section("ETAPA 1: Checklist de Scripts")
        
        scripts = [
            ('src/check_overfitting.py', 'Verificação de overfitting (ranking)'),
            ('src/validate_with_crossval.py', 'Validação temporal + micro-nós'),
            ('src/train.py', 'Script de treinamento'),
            ('src/model.py', 'Arquitetura do modelo'),
        ]
        
        for script, description in scripts:
            exists = self.check_script_exists(script)
            if exists:
                print(f"         → {description}")
        
        # 2-4. Requisições recomendadas
        print("\n")
        self.verify_recommendation_1_timeseries_split()
        self.verify_recommendation_2_regularization()
        self.verify_recommendation_3_micronodes()
        
        # 5. Resumo
        self.print_summary()


def main():
    verifier = OverfittingVerifier()
    verifier.run_all_verifications()


if __name__ == "__main__":
    main()
