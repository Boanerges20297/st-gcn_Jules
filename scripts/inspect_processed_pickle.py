"""Quick inspector for data/processed/processed_graph_data.pkl
Prints type, length, columns, dtypes, CRS (if GeoDataFrame), geometry types, and sample rows.
"""
import os
import pickle
import pandas as pd
import geopandas as gpd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PICKLE = os.path.join(BASE_DIR, 'data', 'processed', 'processed_graph_data.pkl')


def summarize(obj):
    print('OBJECT TYPE:', type(obj))
    try:
        if isinstance(obj, gpd.GeoDataFrame):
            g = obj
            print('GeoDataFrame length:', len(g))
            print('CRS:', g.crs)
            print('Columns:', list(g.columns))
            print('Dtypes:\n', g.dtypes)
            if 'geometry' in g.columns:
                print('Geometry types sample:', g.geometry.geom_type.value_counts().to_dict())
            print('\nFirst 10 rows:')
            with pd.option_context('display.max_columns', None, 'display.width', 200):
                print(g.head(10))
        elif isinstance(obj, pd.DataFrame):
            df = obj
            print('DataFrame length:', len(df))
            print('Columns:', list(df.columns))
            print('Dtypes:\n', df.dtypes)
            print('\nFirst 10 rows:')
            with pd.option_context('display.max_columns', None, 'display.width', 200):
                print(df.head(10))
        else:
                print('Non-DataFrame object; repr:')
                try:
                    s = repr(obj)
                    print(s[:1000])
                except Exception:
                    try:
                        print(str(obj)[:1000])
                    except Exception:
                        pass
                # If it's a dict, list keys and types
                try:
                    if isinstance(obj, dict):
                        print('\nTop-level keys and types:')
                        for k, v in obj.items():
                            tname = type(v).__name__
                            info = ''
                            try:
                                if hasattr(v, '__len__') and not isinstance(v, (str, bytes, dict)):
                                    info = f' len={len(v)}'
                            except Exception:
                                info = ''
                            print(f" - {k}: {tname}{info}")
                except Exception:
                    pass
    except Exception as e:
        print('Summarize failed:', e)


def main():
    if not os.path.exists(PICKLE):
        print('Pickle not found at', PICKLE)
        return
    try:
        with open(PICKLE, 'rb') as fh:
            obj = pickle.load(fh)
    except Exception as e:
        print('Failed to load pickle:', e)
        return
    summarize(obj)
    # check for common columns
    try:
        df = obj if isinstance(obj, (pd.DataFrame, gpd.GeoDataFrame)) else None
        if df is not None:
            for col in ['faction', 'AIS', 'bairro', 'bairro_node', 'bairro_name', 'Bairro', 'name']:
                print(f"Has column '{col}':", col in df.columns)
    except Exception:
        pass

if __name__ == '__main__':
    main()
