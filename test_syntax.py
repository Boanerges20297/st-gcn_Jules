#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Teste de sintaxe do app.py"""
import sys
sys.path.insert(0, r'C:\Users\Boanerges\Desktop\Projetos\st-gcn_jules')

try:
    # Test import - não vamos executar, só verificar sintaxe
    import py_compile
    result = py_compile.compile(r'C:\Users\Boanerges\Desktop\Projetos\st-gcn_jules\app.py', doraise=True)
    print("✅ app.py syntax OK")
except py_compile.PyCompileError as e:
    print(f"❌ Syntax Error: {e}")
    sys.exit(1)
