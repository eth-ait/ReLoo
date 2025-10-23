import torch
import torch.nn as nn
from pytorch3d import ops
from .networks import ImplicitNet
from ..utils.deformation_utils import skinning, batch_rodrigues, to_transform_mat

class VirtualBone(nn.Module):
    """
    Virtual bone deformation module using implicit networks and skinning.
    """

    def __init__(self, opt=None, K=5):
        super().__init__()
        self.deformation_field = ImplicitNet(opt)
        self.K = K
        self.virtual_bones_pos = None

    def update_virtual_bones(self, virtual_bones_pos):
        """
        Update the positions of virtual bones (from a new cloth template).
        """
        self.virtual_bones_pos = torch.from_numpy(virtual_bones_pos).float().cuda().detach()

    def update_K(self, K):
        """Update the number of nearest neighbors for KNN weighting."""
        self.K = K

    def _compute_transform(self, nodes, cond, time_enc, smpl_root_orient, smpl_trans, scale):
        """
        Compute the final transformation matrices for nodes.
        """
        # Predict 6DoF transformation from implicit network
        transformation = self.deformation_field(nodes, cond, time_enc)
        rot = transformation[:, :, :3]
        trans = transformation[0, :, 3:].unsqueeze(-1)

        rot_mat = batch_rodrigues(rot[0])  # axis-angle -> rotation matrix
        transform_mat = to_transform_mat(rot_mat, trans).unsqueeze(0)

        # SMPL root orientation
        smpl_root_mat = batch_rodrigues(smpl_root_orient)
        smpl_root_mat = to_transform_mat(
            smpl_root_mat, torch.zeros([smpl_root_mat.shape[0], 3, 1], device=smpl_root_mat.device)
        ).unsqueeze(0).detach()

        # Combine SMPL root transform with deformation
        transform_mat = torch.matmul(
            smpl_root_mat.expand(-1, transform_mat.shape[1], -1, -1),
            transform_mat
        )

        # Apply scaling and translation
        transform_mat[:, :, :3, :] *= scale.unsqueeze(1).unsqueeze(1)
        transform_mat[:, :, :3, 3] += smpl_trans.unsqueeze(1) * scale.unsqueeze(1)

        return transform_mat

    def forward(self, x, cond, smpl_root_orient, smpl_trans, scale, time_enc, return_nodes_only=False):
        """
        Deform points x using virtual bones.

        Args:
            x (Tensor): Points to deform (N, 3)
            cond (Tensor): Conditioning input for deformation field
            smpl_root_orient (Tensor): SMPL root rotation
            smpl_trans (Tensor): SMPL root translation
            scale (Tensor): Scale factors
            time_enc (Tensor): Temporal encoding for deformation
            return_nodes_only (bool): If True, return deformed virtual bones only

        Returns:
            xc (Tensor): Deformed points (or deformed virtual bones if return_nodes_only=True)
            T (Tensor): Transformation matrices
            outlier_mask (Tensor): Outlier mask for original points (optional)
        """
        if self.virtual_bones_pos is None:
            raise ValueError("Virtual bones not initialized. Call update_virtual_bones() first.")

        # Compute transformation matrices
        transform_mat = self._compute_transform(
            self.virtual_bones_pos, cond, time_enc, smpl_root_orient, smpl_trans, scale
        )

        # Deform virtual bones
        skinning_weights_self = torch.eye(self.virtual_bones_pos.shape[0], device=self.virtual_bones_pos.device)
        nodes_deformed = skinning(
            self.virtual_bones_pos[None],
            skinning_weights_self[None],
            transform_mat,
            inverse=False,
            return_T=False
        ).squeeze(0)

        if return_nodes_only:
            return nodes_deformed

        # Compute KNN weights for input points x
        dist2, nn_index, _ = ops.knn_points(x.unsqueeze(0), nodes_deformed[None], K=self.K, return_nn=False)
        dist = torch.sqrt(dist2)
        least_distance = dist[0, :, 0]

        # Compute normalized weights
        dist_clamped = torch.clamp(dist, max=1.0)
        weights = -torch.log(dist_clamped - 1e-6)[0]
        weights /= weights.sum(dim=-1, keepdim=True)

        # Scatter weights to nearest neighbors
        skinning_weights = torch.zeros((x.shape[0], self.virtual_bones_pos.shape[0]), device=x.device)
        skinning_weights.scatter_(1, nn_index[0], weights)

        # Apply skinning to points
        xc, T = skinning(x[None], skinning_weights[None], transform_mat, inverse=True, return_T=True)

        outlier_mask = least_distance > 0.2
        return xc.squeeze(0), T, outlier_mask