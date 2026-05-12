# Requisitos: Exportação de Top 30 Micronodos

## 📋 Escopo
Implementar a funcionalidade de extração dos 30 micronodos com maior probabilidade de ocorrência (risco) predita pelo modelo Sentinela, formatando-os para inclusão no pacote de screenshots/relatórios.

## ✅ Requisitos Funcionais
1. **Identificação de Top Risco:** Filtrar a inferência do Sentinela para obter os 30 maiores scores.
2. **Enriquecimento Geográfico:** Associar o ID do micronodo ao Bairro e Regional correspondente.
3. **Formatação de Saída:** Gerar arquivo `top_30_micronodes.csv` ou JSON no diretório de exportação.
4. **Integração com Package:** Garantir que o script de screenshot/exportação capture este novo arquivo.
5. **Automação:** O script `sentinela_inference.py` deve gerar este arquivo automaticamente após a inferência.

## 🛠️ Critérios de Aceite
- [ ] O arquivo exportado deve conter exatamente 30 entradas (ou menos, se o total for inferior).
- [ ] Colunas obrigatórias: `micronode_id`, `score`, `bairro`, `regional`.
- [ ] O arquivo deve estar presente na pasta `exports/` ou similar definida para screenshots.
- [ ] A ordenação deve ser decrescente pelo `score`.

## 📌 Restrições
- Não deve impactar o tempo de inferência em mais de 2 segundos.
- Deve utilizar os pesos ativos em `models/active/`.
