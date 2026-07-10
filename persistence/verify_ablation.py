"""Quick verification: AblationDataset(mode='persistence') == PersistenceDataset"""
import sys, os, torch, numpy as np

# Ensure project root and parent are on path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Find persistence data
base = 'E:/912_cyan/Jamming_signal_simulation/output/2D_8X7'
split, jnr = 'train', 0
data_folder = f'{base}/JNR_+{jnr}'
persistence_file = f'{data_folder}/{split}_echo_persistences.mat'
metadata_file = f'{data_folder}/{split}_echo_metadata.json'

if not os.path.exists(persistence_file):
    print(f'Data not found: {persistence_file}')
    print('Searching for persistence .mat files...')
    for root, dirs, files in os.walk('E:/912_cyan/Jamming_signal_simulation/output'):
        for f in files:
            if 'echo_persistences.mat' in f:
                pf = os.path.join(root, f)
                mf = pf.replace('persistences.mat', 'metadata.json')
                print(f'  Found: {pf}')
                if os.path.exists(mf):
                    persistence_file = pf
                    metadata_file = mf
                    break
        else:
            continue
        break
    else:
        print('No persistence .mat files found, skipping verification.')
        sys.exit(0)

from persistence.data import PersistenceDataset, AblationDataset

# Test class names
class_names = ['DFTJ','ISRJ','ISDJ','ISCJ','MISRJ','AJ','BJ','SJ',
               'NCJ','NPJ','SMSPJ','C&IJ','NFMJ','NPMJ','NAMJ','CSJ','PJ']

print(f'Using: {persistence_file}')
print(f'Metadata: {os.path.exists(metadata_file)}')

# Load both datasets (no CLIP norm for raw comparison)
ds_persist = PersistenceDataset(
    persistence_file=persistence_file,
    metadata_file=metadata_file,
    class_names=class_names,
    image_size=224,
    apply_clip_norm=False,
)

ds_ablation = AblationDataset(
    persistence_file=persistence_file,
    metadata_file=metadata_file,
    class_names=class_names,
    image_size=224,
    apply_clip_norm=False,
    ablation_mode='persistence',
)

print(f'PersistenceDataset: {len(ds_persist)} samples')
print(f'AblationDataset:    {len(ds_ablation)} samples')

# Compare all samples (up to 20)
all_close = True
n_check = min(20, len(ds_persist))
for i in range(n_check):
    t1, l1, m1 = ds_persist[i]
    t2, l2, m2 = ds_ablation[i]

    t_close = torch.allclose(t1, t2, atol=1e-7)
    l_close = torch.allclose(l1, l2)
    if not (t_close and l_close):
        print(f'  Sample {i}: tensor={t_close}, label={l_close}, '
              f'max_diff={float((t1 - t2).abs().max()):.2e}')
        all_close = False

# Also test with CLIP norm
ds_persist_norm = PersistenceDataset(
    persistence_file=persistence_file,
    metadata_file=metadata_file,
    class_names=class_names,
    image_size=224,
    apply_clip_norm=True,
)
ds_ablation_norm = AblationDataset(
    persistence_file=persistence_file,
    metadata_file=metadata_file,
    class_names=class_names,
    image_size=224,
    apply_clip_norm=True,
    ablation_mode='persistence',
)
for i in range(min(5, len(ds_persist_norm))):
    t1, l1, m1 = ds_persist_norm[i]
    t2, l2, m2 = ds_ablation_norm[i]
    t_close = torch.allclose(t1, t2, atol=1e-7)
    if not t_close:
        print(f'  CLIP-norm sample {i}: tensor={t_close}, '
              f'max_diff={float((t1 - t2).abs().max()):.2e}')
        all_close = False

if all_close:
    print('SUCCESS: All samples identical between PersistenceDataset and AblationDataset(mode=persistence)!')
else:
    print('FAILURE: Mismatch detected!')
