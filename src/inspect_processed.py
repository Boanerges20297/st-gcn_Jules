import pickle, os
p = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'processed_graph_data.pkl')
if not os.path.exists(p):
    print('MISSING')
else:
    d = pickle.load(open(p, 'rb'))
    ng = d.get('nodes_gdf')
    print('nodes_gdf type:', type(ng))
    try:
        print('columns:', list(ng.columns))
        print('sample rows:')
        print(ng.head(5).to_dict(orient='records'))
    except Exception as e:
        print('error reading nodes_gdf:', e)
    nf = d.get('node_features')
    print('node_features shape:', None if nf is None else nf.shape)
    dates = d.get('dates')
    print('dates len:', None if dates is None else len(dates))
