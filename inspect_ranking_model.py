import pickle, os
p = os.path.join('models','ranking_model_v2.pkl')
print('File:',p,'exists',os.path.exists(p))
if os.path.exists(p):
    m = pickle.load(open(p,'rb'))
    print('Type:',type(m))
    print('Dir sample:',[k for k in dir(m) if not k.startswith('__')][:50])
    # Try predict signature
    for attr in ['predict','predict_proba','score','forward']:
        if hasattr(m,attr):
            print('Has',attr,'->',getattr(m,attr))
