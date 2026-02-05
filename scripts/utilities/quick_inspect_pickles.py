import pickle, pprint, os
files = [
    os.path.join('models','backup','ranking_model_best_Config_01_Small.pkl'),
    os.path.join('models','ranking_model_v2.pkl'),
    os.path.join('models','backup','ranking_model_best_Config_02_SmallLR.pkl')
]
for f in files:
    print('\n===',f,'===')
    if not os.path.exists(f):
        print('MISSING')
        continue
    try:
        with open(f,'rb') as fh:
            obj = pickle.load(fh)
    except Exception as e:
        print('UNPICKLE ERROR:',e)
        continue
    print('TYPE:',type(obj))
    if isinstance(obj,dict):
        print('KEYS:',list(obj.keys()))
        if 'meta' in obj:
            print('META SAMPLE:')
            pprint.pprint({k:obj['meta'].get(k) for k in list(obj['meta'].keys())[:10]})
    else:
        try:
            print('DIR sample:', [k for k in dir(obj) if not k.startswith('__')][:50])
        except Exception as e:
            print('DIR ERROR',e)
    try:
        import numpy as np
        arrs = [x for x in obj.__dict__.values() if isinstance(x, (list,tuple)) or (hasattr(x,'shape') and hasattr(x,'dtype'))]
        if arrs:
            print('Found array-like attributes examples (repr truncated):')
            for a in arrs[:3]:
                try:
                    print(str(type(a)), getattr(a,'shape',None), str(getattr(a,'dtype',None)))
                except:
                    print(type(a))
    except Exception:
        pass
print('\nDone')
