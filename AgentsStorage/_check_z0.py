import yaml, os
ds_dir = r'D:\GitRepository\cable-defect-location-method-based-on-BIS\AI_TEST\DataSet\raw'
ok = 0; bad = 0
for split in ['train','val','test']:
    d = os.path.join(ds_dir, split)
    for f in os.listdir(d):
        if f.endswith('.yaml'):
            with open(os.path.join(d, f)) as fh:
                meta = yaml.safe_load(fh)
            for defect in meta.get('defects', []):
                z0 = defect.get('z0_ohm', 1)
                if z0 <= 0:
                    bad += 1
                    print(f'BAD: {f} z0={z0}')
                else:
                    ok += 1
print(f'All defects z0>0: {ok}, bad: {bad}')
