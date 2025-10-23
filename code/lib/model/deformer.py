import torch
import torch.nn as nn
import torch.nn.functional as F
from .smpl import SMPLServer
from pytorch3d import ops
import cubvh
from ..utils.deformation_utils import skinning

class MeshDeformer(nn.Module):
    def __init__(self, opt, max_dist=0.1, K=1, gender="female", betas=None):
        super().__init__()

        self._deformer = SMPLDeformer(max_dist, K, gender, betas)

        # vertices and faces of the template mesh need to be registered as buffers
        # so that they can be retrieved when loading the model from a checkpoint
        self.register_buffer("verts", self._deformer.smpl_verts)
        self.register_buffer("faces", self._deformer.smpl.smpl.faces_tensor)
        self.register_buffer("weights", self._deformer.smpl_weights)
        self.register_buffer("features", torch.zeros((*self.verts.shape[:-1], 256)).squeeze(0))

        self.bvh = cubvh.cuBVH(self._deformer.smpl_verts[0], self._deformer.smpl.smpl.faces_tensor)
       
    def update(self, verts=None):
        """update the deformer with the current mesh"""
        self.verts = verts
        self.bvh = cubvh.cuBVH(self.verts[0], self._deformer.smpl.smpl.faces_tensor)
    
    def signed_distance(self, x):
        shape = x.shape[:-1]
        signed_distance = self.bvh.signed_distance(x.reshape(-1, 3))[0].reshape(shape)
        return signed_distance


class SMPLDeformer:
    """
    SMPL-based spatial deformer that applies linear blend skinning
    to deform 3D points using SMPL transformations.
    """

    def __init__(self, max_dist=0.2, K=1, gender='female', betas=None):
        """
        Args:
            max_dist (float): Maximum distance threshold for valid correspondences.
            K (int): Number of nearest SMPL vertices to use for skinning weight estimation.
            gender (str): Gender for the SMPL model.
            betas (Tensor or list): Shape parameters for SMPL (length 10).
        """
        super().__init__()
        self.max_dist = max_dist
        self.K = K

        # Initialize SMPL server
        self.smpl = SMPLServer(gender=gender)

        # Construct canonical SMPL parameters
        smpl_params_canonical = self.smpl.param_canonical.clone()
        betas_tensor = torch.tensor(betas, dtype=torch.float32, device=self.smpl.param_canonical.device)
        smpl_params_canonical[:, 76:] = betas_tensor

        # Split SMPL parameters
        cano_scale, cano_transl, cano_thetas, cano_betas = torch.split(
            smpl_params_canonical, [1, 3, 72, 10], dim=1
        )

        # Forward pass to get canonical vertices and weights
        smpl_output = self.smpl(cano_scale, cano_transl, cano_thetas, cano_betas)
        self.smpl_verts = smpl_output['smpl_verts']
        self.smpl_weights = smpl_output['smpl_weights']

    def forward(self, x, smpl_tfs,
                inverse=False, smpl_verts=None,
                return_T=False, body_verts=None):
        """
        Applies deformation to a set of 3D points x.

        Args:
            x (Tensor): Input points (N, 3)
            smpl_tfs (Tensor): SMPL transformations (B, J, 4, 4)
            inverse (bool): If True, apply inverse deformation.
            smpl_verts (Tensor, optional): Override SMPL vertices.
            return_T (bool): Whether to return transformation matrices.
            body_verts (Tensor, optional): Body vertices to refine outlier detection.

        Returns:
            (Tensor, optional Tensor, Tensor):
                - Transformed points
                - (Optional) Transform matrices T
                - Outlier mask
        """
        if x.numel() == 0:
            return x
        smpl_verts = smpl_verts if smpl_verts is not None else self.smpl_verts

        # Optionally deform body vertices for filtering
        body_verts_deformed = (
            self.forward_skinning(body_verts.unsqueeze(0), smpl_tfs)
            if body_verts is not None else None
        )

        # Compute skinning weights & outliers
        weights, outlier_mask = self.query_skinning_weights(
            x[None], smpl_verts=smpl_verts[0],
            smpl_weights=self.smpl_weights,
            body_verts_d=body_verts_deformed
        )

        # Apply skinning transform
        x_transformed, T = skinning(x.unsqueeze(0), weights, smpl_tfs, inverse=inverse, return_T=True)

        if return_T:
            return x_transformed.squeeze(0), T, outlier_mask
        return x_transformed.squeeze(0), outlier_mask

    def forward_skinning(self, x_canonical, smpl_tfs):
        """Forward skinning of canonical-space points."""
        weights, _ = self.query_skinning_weights(
            x_canonical, smpl_verts=self.smpl_verts[0],
            smpl_weights=self.smpl_weights
        )
        return skinning(x_canonical, weights, smpl_tfs, inverse=False)

    def query_skinning_weights(self, pts, smpl_verts, smpl_weights, body_verts_d=None):
        """
        Query skinning weights from nearest SMPL vertices and detect outliers.

        Args:
            pts (Tensor): Query points (1, N, 3)
            smpl_verts (Tensor): SMPL vertices (V, 3)
            smpl_weights (Tensor): SMPL LBS weights (B, V, J)
            body_verts_d (Tensor, optional): Deformed body vertices for outlier filtering.

        Returns:
            weights (Tensor): (B, N, J) Skinning weights
            outlier_mask (BoolTensor): (N,) Outlier mask
        """
        # Nearest neighbors from SMPL mesh
        dists, idx, _ = ops.knn_points(pts, smpl_verts[None], K=self.K, return_nn=True)
        dists = torch.clamp(dists, max=4)
        weights_conf = torch.exp(-dists)
        weights_conf /= weights_conf.sum(-1, keepdim=True)

        # Blend vertex weights
        weights = smpl_weights[:, idx[0], :]
        weights = torch.sum(weights * weights_conf.unsqueeze(-1), dim=-2).detach()

        # Compute distances for outlier detection
        if body_verts_d is not None:
            dists, _, _ = ops.knn_points(pts, body_verts_d, K=1, return_nn=False)
        dists = torch.sqrt(torch.clamp(dists, max=4))
        outlier_mask = (dists[..., 0] > self.max_dist)[0]

        return weights, outlier_mask

    def update_K(self, K):
        """Update the number of nearest neighbors used for weighting."""
        self.K = K