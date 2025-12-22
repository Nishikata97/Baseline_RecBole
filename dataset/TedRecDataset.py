import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from recbole.data.dataset import SequentialDataset


class TedRecDataset(SequentialDataset):
    def __init__(self, config):
        super().__init__(config)

        self.plm_size = config['plm_size']
        self.plm_suffix = config['plm_suffix']
        plm_embedding_weight = self.load_plm_embedding()
        self.plm_embedding = self.weight2emb(plm_embedding_weight)

    def load_plm_embedding(self):
        """Load PLM embeddings from either .feat1CLS or .npy format"""
        # First try the original .feat1CLS format
        feat_path = osp.join(self.config['data_path'], f'{self.dataset_name}.{self.plm_suffix}')
        
        use_npy_format = False
        if os.path.exists(feat_path):
            # Load from .feat1CLS format (original format)
            loaded_feat = np.fromfile(feat_path, dtype=np.float32).reshape(-1, self.plm_size)
        else:
            # Try to load from .npy format (new Amazon datasets format)
            # Support both: {dataset}_{suffix} and direct relative path (e.g., subdir/file.npy)
            npy_path = osp.join(self.config['data_path'], f'{self.dataset_name}_{self.plm_suffix}')
            
            # If not found, try as a direct relative path under data_path
            if not os.path.exists(npy_path):
                npy_path = osp.join(self.config['data_path'], self.plm_suffix)
            
            if os.path.exists(npy_path):
                print(f"Loading embeddings from .npy format: {npy_path}")
                loaded_feat = np.load(npy_path)
                print(f"Loaded embedding shape: {loaded_feat.shape}")
                use_npy_format = True
                
                # Check if plm_size matches the loaded embedding dimension
                if loaded_feat.shape[1] != self.plm_size:
                    print(f"Warning: plm_size in config ({self.plm_size}) doesn't match "
                          f"loaded embedding dimension ({loaded_feat.shape[1]}). "
                          f"Using loaded dimension.")
                    self.plm_size = loaded_feat.shape[1]
            else:
                raise FileNotFoundError(
                    f"Neither {feat_path} nor {npy_path} exists. "
                    f"Please check your dataset path and format."
                )

        mapped_feat = np.zeros((self.item_num, self.plm_size))
        
        if use_npy_format:
            # For .npy format: need to map from RecBole's internal ID to original item_id in .item file
            import pandas as pd
            item_file = osp.join(self.config['data_path'], f'{self.dataset_name}.item')
            item_df = pd.read_csv(item_file, sep='\t')
            original_item_ids = item_df['item_id:token'].tolist()
            
            # Create mapping from original item_id to embedding index
            item_id_to_emb_idx = {item_id: idx for idx, item_id in enumerate(original_item_ids)}
            
            # Map from RecBole's internal ID to embedding
            for internal_id, token in enumerate(self.field2id_token['item_id']):
                if token == '[PAD]': 
                    continue
                if token in item_id_to_emb_idx:
                    emb_idx = item_id_to_emb_idx[token]
                    mapped_feat[internal_id] = loaded_feat[emb_idx]
                else:
                    print(f"Warning: item_id {token} not found in embeddings")
        else:
            # For .feat1CLS format: original logic (assumes integer item_ids)
            for i, token in enumerate(self.field2id_token['item_id']):
                if token == '[PAD]': 
                    continue
                mapped_feat[i] = loaded_feat[int(token)]
        
        return mapped_feat

    def weight2emb(self, weight):
        plm_embedding = nn.Embedding(self.item_num, self.plm_size, padding_idx=0)
        plm_embedding.weight.requires_grad = False
        plm_embedding.weight.data.copy_(torch.from_numpy(weight))
        return plm_embedding
