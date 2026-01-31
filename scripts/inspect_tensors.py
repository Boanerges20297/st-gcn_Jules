import numpy as np, pickle, os
p='data/processed'
files=['tensor_cvli_prisoes_faccoes.npy','tensor_cvli_univariado.npy','tensor_multivariado.npy','tensor_prisoes.npy']
for f in files:
    fp=os.path.join(p,f)
    print(f, 'exists' if os.path.exists(fp) else 'missing')
    if os.path.exists(fp):
        a=np.load(fp, allow_pickle=True)
        try:
            print('  shape', a.shape)
        except Exception as e:
            print('  load error', e)
pg=os.path.join(p,'processed_graph_data.pkl')
if os.path.exists(pg):
    with open(pg,'rb') as fh:
        d=pickle.load(fh)
    print('processed_graph_data keys', list(d.keys())[:20])
    print('dates in pg?', 'dates' in d)
else:
    print('processed_graph_data.pkl missing')
gd=os.path.join(p,'graph_data','dates.pkl')
if os.path.exists(gd):
    with open(gd,'rb') as fh:
        dates=pickle.load(fh)
    print('graph_data/dates len', len(dates), 'sample', dates[:3])
else:
    print('graph_data/dates.pkl missing')
