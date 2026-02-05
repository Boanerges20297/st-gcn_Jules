import numpy as np
p='data/processed/adjacency_matrix_backup_20260123_105747.npy'
arr=np.load(p,allow_pickle=True)
print(type(arr), arr.shape)
print('dtype', arr.dtype)
for i,a in enumerate(arr[:2]):
    print(i,type(a), getattr(a,'shape',None))
