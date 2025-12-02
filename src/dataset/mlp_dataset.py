import torch
import numpy as np

from .base_depth_dataset import BaseDepthDataset, DepthFileNameMode


class MLPDepthDataset(BaseDepthDataset):
    def __init__(
        self,
        eigen_valid_mask: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            # MLP prosthesis dataset parameters
            min_depth=0.1,  # Minimum depth in meters for your dataset
            max_depth=10.0,  # Maximum depth in meters for your dataset
            has_filled_depth=False,  # You don't have filled depth
            name_mode=DepthFileNameMode.id,  # Use 'id' naming mode
            **kwargs,
        )
        
        self.eigen_valid_mask = eigen_valid_mask
        
        # Optional: Filter out files where depth doesn't exist
        filtered_filenames = []
        for line in self.filenames:
            if len(line) >= 2:  # Must have at least rgb and depth paths
                filtered_filenames.append(line)
        self.filenames = filtered_filenames

    def _read_depth_file(self, rel_path):
        """Read depth file for MLP prosthesis dataset"""
        depth_in = self._read_image(rel_path)
        
        # Debug: Print depth file info
        # print(f"Reading depth: {rel_path}, shape: {depth_in.shape}, dtype: {depth_in.dtype}")
        
        # MLP dataset specific depth decoding
        if depth_in.dtype == np.uint16:
            # 16-bit depth - assuming metric depth in millimeters
            depth_decoded = depth_in.astype(np.float32) / 1000.0  # mm to meters
        elif depth_in.dtype == np.uint8:
            # 8-bit depth - normalize to metric range
            depth_decoded = depth_in.astype(np.float32) / 255.0 * self.max_depth
        else:
            # Assume depth is already in float32
            depth_decoded = depth_in.astype(np.float32)
        
        # Clip to valid range
        depth_decoded = np.clip(depth_decoded, self.min_depth, self.max_depth)
        
        return depth_decoded

    def _get_valid_mask(self, depth: torch.Tensor):
        """Create valid mask for MLP dataset"""
        valid_mask = super()._get_valid_mask(depth)

        # Apply Eigen-style crop if specified in config
        if self.eigen_valid_mask:
            # Custom crop for prosthesis dataset
            eval_mask = torch.zeros_like(valid_mask.squeeze()).bool()
            H, W = eval_mask.shape
            
            # For prosthesis view, you might want to keep central region
            # Adjust based on what part of the image contains valid depth
            h_start = int(0.0 * H)  # Start from top
            h_end = int(1.0 * H)    # Go to bottom  
            w_start = int(0.0 * W)  # Start from left
            w_end = int(1.0 * W)    # Go to right
            
            eval_mask[h_start:h_end, w_start:w_end] = 1
            eval_mask = eval_mask.reshape(valid_mask.shape)
            valid_mask = torch.logical_and(valid_mask, eval_mask)

        return valid_mask