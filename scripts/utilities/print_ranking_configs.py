import pickle, pprint, os
files = [
    os.path.join('models','backup','ranking_model_best_Config_01_Small.pkl'),
    os.path.join('models','backup','ranking_model_best_Config_02_SmallLR.pkl'),
    os.path.join('models','ranking_model_v2.pkl')
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
    if isinstance(obj,dict):
        print('Keys:',list(obj.keys()))
        if 'config' in obj:
            print('CONFIG:')
            pprint.pprint(obj['config'])
        if 'best_val_p5' in obj:
            print('best_val_p5:',obj.get('best_val_p5'))
        if 'eval_p5' in obj:
            print('eval_p5:',obj.get('eval_p5'))
        if 'scores' in obj:
            s = obj['scores']
            try:
                import numpy as np
                print('scores shape:', getattr(s,'shape',None), 'min/max:', float(np.min(s)), float(np.max(s)))
            except Exception:
                print('scores repr length',len(repr(s)))
    else:
        print('Non-dict pickle type:',type(obj))
print('\nDone')
