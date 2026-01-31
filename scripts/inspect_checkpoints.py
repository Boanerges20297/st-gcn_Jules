import torch, os
for name in ['models/stgcn_cvli.pth','models/stgcn_cvp.pth']:
    if os.path.exists(name):
        print('---',name)
        sd=torch.load(name,map_location='cpu')
        if isinstance(sd,dict):
            for k in sorted(sd.keys()):
                if 'conv_final.weight' in k or 'layer' in k or 'fc.weight' in k:
                    print(k, sd[k].shape)
        else:
            print('checkpoint not dict, type', type(sd))
    else:
        print(name, 'missing')
