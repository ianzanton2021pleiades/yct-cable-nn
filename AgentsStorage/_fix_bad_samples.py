"""Remove old Phase 1 samples with negative Z0 from dataset."""
import yaml, os, glob

ds_dir = r'D:\GitRepository\cable-defect-location-method-based-on-BIS\AI_TEST\DataSet'
removed = 0

for split in ['train', 'val', 'test']:
    raw_dir = os.path.join(ds_dir, 'raw', split)
    label_dir = os.path.join(ds_dir, 'labels', split)
    
    for yaml_path in glob.glob(os.path.join(raw_dir, '*.yaml')):
        with open(yaml_path) as f:
            meta = yaml.safe_load(f)
        
        has_bad = any(d.get('z0_ohm', 1) <= 0 for d in meta.get('defects', []))
        if has_bad:
            sid = meta['sample_id']
            # Remove yaml, csv, npy
            for ext_dir, ext in [(raw_dir, '.yaml'), (raw_dir, '.csv'), (label_dir, '.npy')]:
                fpath = os.path.join(ext_dir, sid + ext)
                if os.path.exists(fpath):
                    os.remove(fpath)
            removed += 1
            print(f"Removed: {sid}")

print(f"Total removed: {removed}")

# Update manifest
manifest_path = os.path.join(ds_dir, 'manifest.yaml')
with open(manifest_path) as f:
    manifest = yaml.safe_load(f)

# Filter out removed samples
bad_ids = set()
for split in ['train', 'val', 'test']:
    raw_dir = os.path.join(ds_dir, 'raw', split)
    existing = set(f.replace('.csv','').replace('.yaml','') for f in os.listdir(raw_dir))
    # Just rebuild from what's on disk

# Simpler: just recount
all_samples = manifest['samples']
valid_samples = []
for s in all_samples:
    sid = s['sample_id']
    split = s['split']
    csv_path = os.path.join(ds_dir, 'raw', split, sid + '.csv')
    if os.path.exists(csv_path):
        valid_samples.append(s)

manifest['samples'] = valid_samples
manifest['n_total'] = len(valid_samples)
manifest['n_train'] = sum(1 for s in valid_samples if s['split'] == 'train')
manifest['n_val'] = sum(1 for s in valid_samples if s['split'] == 'val')
manifest['n_test'] = sum(1 for s in valid_samples if s['split'] == 'test')

with open(manifest_path, 'w') as f:
    yaml.dump(manifest, f, allow_unicode=True, default_flow_style=False)

print(f"Manifest updated: {manifest['n_total']} total "
      f"(train={manifest['n_train']}, val={manifest['n_val']}, test={manifest['n_test']})")
