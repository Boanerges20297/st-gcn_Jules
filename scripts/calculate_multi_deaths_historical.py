import pandas as pd
import os

# Caminho do arquivo
CSV_FILE = 'data/raw/dados_status_ocorrencias_gerais_ENRIQUECIDO.csv'

def calculate_qtd_mortes():
    print(f"🚀 Iniciando cálculo de múltiplas mortes para {CSV_FILE}")
    
    if not os.path.exists(CSV_FILE):
        print("❌ Erro: Arquivo não encontrado.")
        return

    # 1. Carregar CSV
    print("⏳ Carregando CSV...")
    df = pd.read_csv(CSV_FILE, low_memory=False)
    print(f"✅ CSV carregado: {len(df)} registros.")

    # 2. Criar chave de evento robusta
    print("🔄 Criando chaves de evento para agrupamento...")
    # Garantir que lat/lng sejam numéricos
    df['lat_num'] = pd.to_numeric(df['latitude'], errors='coerce').fillna(0).round(3)
    df['lng_num'] = pd.to_numeric(df['longitude'], errors='coerce').fillna(0).round(3)
    
    # Chave: Data + Hora + Bairro + Coordenadas Arredondadas
    df['event_key'] = (
        df['data'].astype(str) + "_" + 
        df['hora'].astype(str) + "_" + 
        df['bairro'].fillna('').astype(str).str.upper() + "_" +
        df['lat_num'].astype(str) + "_" +
        df['lng_num'].astype(str)
    )

    # 3. Convergir múltiplas mortes em um único registro (Collapse)
    print("📊 Convergindo múltiplas vítimas em eventos únicos...")
    cvli_mask = df['tipo'].str.lower() == 'cvli'
    
    if cvli_mask.any():
        df_cvli = df[cvli_mask].copy()
        df_others = df[~cvli_mask].copy()
        
        # Agrupar CVLIs
        # Definimos como agregar cada coluna importante
        agg_dict = {col: 'first' for col in df_cvli.columns if col not in ['event_key', 'qtd_mortes']}
        
        # Ajustes específicos de agregação
        if 'id' in df_cvli.columns:
            agg_dict['id'] = lambda x: ' | '.join([str(v) for v in x if pd.notna(v)])
        
        # Remover colunas de privacidade (LGPD)
        for col_priv in ['nome_vitima', 'vitima']:
            if col_priv in agg_dict: del agg_dict[col_priv]
            if col_priv in df_others.columns: df_others = df_others.drop(columns=[col_priv])
            if col_priv in df_cvli.columns: df_cvli = df_cvli.drop(columns=[col_priv])
        
        # Contagem de mortes
        df_cvli_collapsed = df_cvli.groupby('event_key').agg(agg_dict).reset_index()
        df_cvli_collapsed['qtd_mortes'] = df_cvli.groupby('event_key').size().values
        
        # Recombinar
        df_final = pd.concat([df_others, df_cvli_collapsed], ignore_index=True)
        
        print(f"✅ Convergência concluída:")
        print(f"   - Registros CVLI originais: {len(df_cvli)}")
        print(f"   - Registros CVLI convergidos: {len(df_cvli_collapsed)}")
        print(f"   - Redução de redundância: {len(df_cvli) - len(df_cvli_collapsed)} linhas removidas.")
        
        df = df_final
    else:
        print("⚠ Nenhuma ocorrência de CVLI encontrada para agrupar.")
        df['qtd_mortes'] = 1

    # 4. Ajustar ordem das colunas (para manter o padrão solicitado)
    cols = list(df.columns)
    # Remover colunas auxiliares
    for c in ['lat_num', 'lng_num', 'event_key']:
        if c in cols: cols.remove(c)
    
    # Garantir que qtd_mortes esteja perto das outras colunas de enriquecimento
    if 'qtd_mortes' in cols: cols.remove('qtd_mortes')
    
    if 'version' in cols:
        idx = cols.index('version') + 1
    elif 'id' in cols:
        idx = cols.index('id') + 1
    else:
        idx = len(cols)
        
    final_cols = cols[:idx] + ['qtd_mortes'] + cols[idx:]
    df = df[final_cols]

    # 5. Salvar
    print(f"💾 Salvando arquivo atualizado...")
    df.to_csv(CSV_FILE, index=False, encoding='utf-8-sig')
    print("✅ Concluído com sucesso!")

if __name__ == "__main__":
    calculate_qtd_mortes()
