# 🎯 ST-GCN Jules: Spatial-Temporal Crime Prediction System

**Versão**: 2.0 (Production-Ready with Real-Time Ranking Validation)  
**Status**: ✅ Complete | 🔄 Live | 📚 Documented  
**Data**: Fevereiro 2026 | Fortaleza, Ceará

---

## 📚 Documentação

Toda a documentação está centralizada em `/docs/`:

- **[📖 Start Here](docs/README.md)** - Documentation Index (navegue por role/tópico)
- **[🚀 Quick Start](docs/QUICK_START.md)** - Setup em 15 minutos
- **[📊 Full Reference](docs/README.md)** - Documentação completa (1205 linhas)
- **[🏗️ Technical Details](docs/TECHNICAL_SUMMARY.md)** - Arquitetura e decisões técnicas
- **[📁 Project Structure](docs/STRUCTURE.md)** - Organização de diretórios

---

## ⚡ Quick Commands

```bash
# Setup & Run
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py

# Access Dashboard
# http://127.0.0.1:5000/map

# Test System
python scripts/tests/demo_ranking_validation.py

# Train New Model
python scripts/training/train_ranking_window30_final.py
```

---

## 🎯 Sistema em Operação

**Performance**:
- ST-GCN: P@5 = 0.70
- RankingModel: P@5 = 0.80
- **Combined: P@5 = 0.80 (100% Top-5 concordância)**

**Cobertura**: 319 bairros × 1491 dias históricos + Real-time validation

**Modelos Ativos**: 
- `models/stgcn_model_v2.pth` (279 KB)
- `models/ranking_model_window30_final.pkl` (2.2 MB)

---

## 📂 Estrutura Organizada

```
st-gcn_Jules/
├── 📚 docs/                  ← TODA A DOCUMENTAÇÃO
│   ├── README.md            (índice)
│   ├── QUICK_START.md
│   ├── TECHNICAL_SUMMARY.md
│   ├── STRUCTURE.md
│   └── ...
├── 🤖 app.py               (Flask principal)
├── 📊 models/              (apenas modelos ativos)
├── 📂 scripts/             (organizados em 5 categorias)
│   ├── training/
│   ├── tuning/
│   ├── tests/
│   ├── debug/
│   └── utilities/
└── 📂 src/                 (código-fonte)
```

---

## 📖 Por Onde Começar?

1. **Primeira vez usando?** → [QUICK_START.md](docs/QUICK_START.md)
2. **Quer entender tudo?** → [docs/README.md](docs/README.md) (índice completo)
3. **Precisa deploy?** → [TECHNICAL_SUMMARY.md](docs/TECHNICAL_SUMMARY.md)
4. **Mexer no código?** → [STRUCTURE.md](docs/STRUCTURE.md)

---

## ✅ Status Produção

- ✅ Modelos treinados e validados
- ✅ Real-time ranking validation (100% concordância)
- ✅ API endpoints funcionando
- ✅ Dashboard interativo
- ✅ Documentação completa (2100+ linhas)
- ✅ Scripts organizados (67 total)
- ✅ Criticidade 3-níveis (CRÍTICO/ALERTA/MONITORADO)
- ✅ Integração de eventos exógenos

---

**Versão**: 2.0.0 | **Data**: 04/02/2026 | **Status**: Production-Ready ✅

Para detalhes, consulte [docs/README.md](docs/README.md)
