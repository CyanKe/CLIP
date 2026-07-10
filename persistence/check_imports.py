"""Quick import check for ablation module"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from persistence.data import (
    PersistenceDataset, AblationDataset,
    create_persistence_dataloaders, create_ablation_dataloaders,
    collate_fn_ablation,
)
from persistence import AblationDataset as AD, create_ablation_dataloaders, collate_fn_ablation as cfa

print('All imports OK!')
print('AblationDataset:', AD)
print('create_ablation_dataloaders:', create_ablation_dataloaders)
print('collate_fn_ablation:', cfa)
