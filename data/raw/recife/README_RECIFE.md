# 🌴 Missão Recife: Teste de Generalização Hardcore

Este diretório prepara o terreno para testar a arquitetura **DeepSTGAT** em uma nova capital, provando que o modelo não é "viciado" em Fortaleza.

## 📂 Estrutura Atual
*   `bairros_recife.geojson`: Malha geográfica oficial (Já baixado ✅).
*   `cvli_recife.csv`: **[PENDENTE]** Arquivo de crimes que você precisa providenciar.

## 🚀 Passo a Passo

### 1. Obter os Dados de Crimes
Acesse o site da SDS-PE ou dados abertos e baixe a série histórica de CVLI (2023-2025).
*   **Formato ideal:** CSV ou Excel.
*   **Colunas Obrigatórias:**
    *   `Data` (Data do fato)
    *   `Bairro` (Nome do bairro onde ocorreu)
    *   `Municipio` (Deve ser "Recife")

### 2. Salvar o Arquivo
Renomeie o arquivo baixado para **`cvli_recife.csv`** e salve-o nesta pasta:
`C:\Users\STI01\Desktop\Projetos\st-gcn_Jules\dataawecife`

### 3. Rodar a Ingestão
Assim que o arquivo estiver na pasta, peça para o Gemini CLI:
> "Processe os dados de Recife agora"

O sistema irá automaticamente:
1.  Ler o CSV e padronizar os nomes dos bairros.
2.  Cruzar com o GeoJSON para criar a Matriz de Adjacência ($A$).
3.  Gerar o arquivo processado `processed_recife.pkl`.

### 4. Treinar o Modelo
Com os dados processados, o Gemini irá disparar o treino:
> "Treine o especialista Recife"

O resultado será um modelo `recife_model.pth` capaz de prever riscos na capital pernambucana com a mesma tecnologia usada no Ceará.
