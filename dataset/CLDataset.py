"""
Custom Dataset for DuoRec and CL4SRec models
Support semantic augmentation for DuoRec
"""

import os
import numpy as np
import copy
from recbole.data.dataset import SequentialDataset
from logging import getLogger


class CLDataset(SequentialDataset):
    """
    Custom Sequential Dataset for contrastive learning based models (DuoRec, CL4SRec)
    Support semantic augmentation for DuoRec
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.logger = getLogger()
        self.ssl_aug = config['SSL_AUG'] if 'SSL_AUG' in config.final_config_dict else None
        
    def semantic_augmentation(self, target_index):
        """
        Generate same_target_index for DuoRec semantic augmentation
        For each sequence, find all other sequences with the same target item
        
        Args:
            target_index: indices of target items in interaction
            
        Returns:
            same_target_index: array of arrays, each containing indices of sequences with same target
        """
        aug_path = os.path.join(self.config['data_path'], self.dataset_name, 'semantic_augmentation.npy')
        
        if os.path.exists(aug_path):
            self.logger.info(f"Loading semantic augmentation from {aug_path}")
            same_target_index = np.load(aug_path, allow_pickle=True)
        else:
            self.logger.info(f"Computing semantic augmentation...")
            same_target_index = []
            target_item = self.inter_feat['item_id'][target_index].numpy()
            
            for index, item_id in enumerate(target_item):
                # Find all sequences with the same target item
                all_index_same_id = np.where(target_item == item_id)[0]
                # Remove the current sequence itself
                delete_index = np.argwhere(all_index_same_id == index)
                all_index_same_id_wo_self = np.delete(all_index_same_id, delete_index)
                same_target_index.append(all_index_same_id_wo_self)
                
            same_target_index = np.array(same_target_index, dtype=object)
            
            # Save for future use
            os.makedirs(os.path.dirname(aug_path), exist_ok=True)
            np.save(aug_path, same_target_index)
            self.logger.info(f"Saved semantic augmentation to {aug_path}")
        
        return same_target_index
    
    def leave_one_out(self, group_by, leave_one_num=1):
        """
        Override leave_one_out to add semantic augmentation for DuoRec
        """
        self.logger.debug(f'Leave one out, group_by=[{group_by}], leave_one_num=[{leave_one_num}].')
        
        if group_by is None:
            raise ValueError('Leave one out strategy require a group field.')
        if group_by != self.uid_field:
            raise ValueError('Sequential models require group by user.')
        
        self.prepare_data_augmentation()
        grouped_index = self._grouped_index(self.uid_list)
        next_index = self._split_index_by_leave_one_out(grouped_index, leave_one_num)
        
        self._drop_unused_col()
        next_ds = []
        for index in next_index:
            ds = copy.copy(self)
            for field in ['uid_list', 'item_list_index', 'target_index', 'item_list_length']:
                setattr(ds, field, np.array(getattr(ds, field)[index]))
            next_ds.append(ds)
        
        # Apply mask to avoid data leakage
        next_ds[0].mask = np.ones(len(self.inter_feat), dtype=bool)
        next_ds[1].mask = np.ones(len(self.inter_feat), dtype=bool)
        next_ds[2].mask = np.ones(len(self.inter_feat), dtype=bool)
        
        next_ds[0].mask[self.target_index[next_index[1] + next_index[2]]] = False
        next_ds[1].mask[self.target_index[next_index[2]]] = False
        
        # Semantic augmentation for DuoRec (only for training set)
        if self.ssl_aug == 'DuoRec':
            self.logger.info("Generating semantic augmentation for DuoRec...")
            self.same_target_index = self.semantic_augmentation(next_ds[0].target_index)
            setattr(next_ds[0], 'same_target_index', self.same_target_index)
            self.logger.info(f"Semantic augmentation completed. Total sequences: {len(self.same_target_index)}")
        
        return next_ds

