from .networks import ImplicitNet, RenderingNet
from .virtual_bone import VirtualBone
from .simple_density import AbsDensity
from .simple_ray_sampler import UniformSampler
from .deformer import MeshDeformer
from .smpl import SMPLServer
from .sampler import PointInSpace
from ..utils import utils
from .sdfstudio.models.volsdf import VolSDFModel
from ..utils.meshing import simplify_mesh
import numpy as np
import trimesh
import torch
import torch.nn as nn
import hydra
import os, glob

class ReLoo(nn.Module):
    def __init__(self, opt, betas_path, gender, num_training_frames, num_clothes):
        super().__init__()

        # Network Initialization
        self.num_clothes = num_clothes
        self.gender = gender
        self.sdf_bounding_sphere = opt.body_network.scene_bounding_sphere

        # Inner body networks
        self.body_network = ImplicitNet(opt.body_network)
        self.rendering_network = RenderingNet(opt.rendering_network)

        # Outer clothing networks
        if num_clothes == 2:
            self.clothing_network1 = ImplicitNet(opt.clothing_implicit_network)
            self.clothing_network2 = ImplicitNet(opt.clothing_implicit_network)
            self.clothing_network_list = [self.clothing_network1, self.clothing_network2]
        elif num_clothes == 1:
            self.clothing_network = ImplicitNet(opt.clothing_implicit_network)
            self.clothing_network_list = [self.clothing_network]

        # Background networks
        self.bg_implicit_network = ImplicitNet(opt.bg_implicit_network)
        self.bg_rendering_network = RenderingNet(opt.bg_rendering_network)

        # Frame latent encoder
        self.frame_latent_encoder = nn.Embedding(num_training_frames, opt.bg_rendering_network.dim_frame_encoding)
        self.sampler = PointInSpace()

        # Load SMPL betas
        betas = np.load(betas_path)
        checkpoints = sorted(glob.glob("checkpoints/*.ckpt"))
        if checkpoints:
            checkpoint = torch.load(checkpoints[-1])
            betas = checkpoint['state_dict']['body_model_params.betas.weight'].detach().cpu().numpy().squeeze()

        self.deformer = MeshDeformer(opt=opt.deformer, betas=betas, gender=self.gender)
        self.virtual_bone = VirtualBone(opt=opt.virtual_bone)
        self.num_vb = opt.virtual_bone.num_vb
        self.epoch_start_vb = opt.virtual_bone.epoch_start_vb

        # Background + Sampling
        self.bg_density = AbsDensity()
        self.inverse_sphere_sampler = UniformSampler(1.0, 0.0, 32, False, far=1.0)
        self.smpl_server = SMPLServer(gender=self.gender, betas=betas)
        self.model_fg = VolSDFModel()

        # Optional SMPL Initialization
        if opt.smpl_init:
            smpl_model_state = torch.load(hydra.utils.to_absolute_path('../assets/smpl_init.pth'))
            self._copy_smpl_init(smpl_model_state['model_state_dict'])

        # Build Canonical SMPL Mesh
        smpl_v_cano = self.smpl_server.verts_c.squeeze(0).detach().cpu().numpy()
        smpl_f_cano = self.smpl_server.smpl.faces
        self.smpl_mesh = trimesh.Trimesh(smpl_v_cano, smpl_f_cano)

        # Load and Update Templates for continuing training
        body_template = self.load_template("simplified/body.ply", simplify_mesh(self.smpl_mesh, 2048))
        if num_clothes == 2:
            cloth_template1 = self.load_template("simplified/cloth1.ply")
            cloth_template2 = self.load_template("simplified/cloth2.ply")
            self.update_template(body_template, cloth_template1, cloth_template2)
        elif num_clothes == 1:
            cloth_template = self.load_template("simplified/cloth1.ply")
            self.update_template(body_template, cloth_template)

    # Helper function for SMPL init
    def _copy_smpl_init(self, checkpoint):
        """Copy SMPL pre-trained weights into implicit and clothing networks."""
        def copy_params(target_net, checkpoint):
            state_dict = target_net.state_dict()
            for name, param in checkpoint.items():
                if name in state_dict:
                    tparam = state_dict[name]
                    if tparam.shape == param.shape:
                        tparam.copy_(param)
                    else:
                        # handle mismatched shapes gracefully
                        shape = param.shape
                        print(f"[Warning] Shape mismatch {name}: {shape} -> {tparam.shape}")
                        if len(shape) == 1:
                            tparam[:shape[0]] = param
                        elif len(shape) == 2:
                            tparam[:shape[0], :shape[1]] = param

        # copy to implicit network
        copy_params(self.body_network, checkpoint)
        # copy to all clothing networks
        for cloth_net in self.clothing_network_list:
            copy_params(cloth_net, checkpoint)

    def load_template(self, mesh_file, default_simplify_mesh=None):
        """Load a mesh template from a given file path.

        Args:
            mesh_file (str): Path to the template mesh file (e.g., 'simplified/body.ply').
            default_simplify_mesh (trimesh.Trimesh, optional): Fallback mesh if the file does not exist.

        Returns:
            trimesh.Trimesh or None: The loaded mesh, the default simplified mesh, or None.
        """
        if mesh_file and os.path.exists(mesh_file):
            return trimesh.load(mesh_file, process=False)

        if default_simplify_mesh is not None:
            return default_simplify_mesh

        return None

    def update_template(self, mesh_body, mesh_cloth1=None, mesh_cloth2=None):
        device = self.smpl_server.verts_c.device

        # Body mesh
        self.mesh_v_body = torch.tensor(mesh_body.vertices, dtype=torch.float32, device=device)
        self.mesh_f_body = torch.tensor(mesh_body.faces, dtype=torch.long, device=device)

        # Clothing meshes
        if self.num_clothes == 2 and mesh_cloth1 is not None and mesh_cloth2 is not None:
            self.mesh_v_cloth1 = torch.tensor(mesh_cloth1.vertices, dtype=torch.float32, device=device)
            self.mesh_f_cloth1 = torch.tensor(mesh_cloth1.faces, dtype=torch.long, device=device)
            self.mesh_v_cloth2 = torch.tensor(mesh_cloth2.vertices, dtype=torch.float32, device=device)
            self.mesh_f_cloth2 = torch.tensor(mesh_cloth2.faces, dtype=torch.long, device=device)
        elif self.num_clothes == 1 and mesh_cloth1 is not None:
            self.mesh_v_cloth = torch.tensor(mesh_cloth1.vertices, dtype=torch.float32, device=device)
            self.mesh_f_cloth = torch.tensor(mesh_cloth1.faces, dtype=torch.long, device=device)

        # Virtual bone update
        if mesh_cloth1 is not None and hasattr(self, 'virtual_bone'):
            virtual_bones_files = sorted(glob.glob('virtual_bones/*.npy'))
            if virtual_bones_files:
                virtual_bones = np.load(virtual_bones_files[-1])
                self.virtual_bone.update_virtual_bones(virtual_bones)
        else:
            print("[Warning] No virtual bone files found in 'virtual_bones/*.npy'.")

    def sdf_func_with_deformer(
        self, x, cond, smpl_params, smpl_outputs, time_enc, net, using_virtual_bone=False
    ):
        smpl_tfs = smpl_outputs['smpl_tfs']
        smpl_verts = smpl_outputs['smpl_verts']
        body_verts = smpl_outputs['body_verts']
        scale = smpl_params['scale']
        smpl_root_orient = smpl_params['smpl_root_orient']
        smpl_trans = smpl_params['smpl_trans']

        # Compute canonical points using deformer or virtual bone
        if hasattr(self, "deformer"):
            if using_virtual_bone:
                if self.current_epoch < self.epoch_start_vb:
                    # Use standard SMPL deformer before virtual bones start
                    x_c, T, outlier_mask = self.deformer._deformer.forward(
                        x, smpl_tfs, 
                        inverse=True, smpl_verts=smpl_verts, 
                        return_T=True, body_verts=body_verts
                    )
                else:
                    # Use virtual bone for canonical points
                    x_c, T, outlier_mask = self.virtual_bone(
                        x, cond,
                        smpl_root_orient=smpl_root_orient,
                        smpl_trans=smpl_trans,
                        scale=scale, time_enc=time_enc
                    )
            else:
                # Use standard SMPL deformer
                x_c, T, outlier_mask = self.deformer._deformer.forward(
                    x, smpl_tfs, 
                    inverse=True, smpl_verts=smpl_verts,
                    return_T=True, body_verts=body_verts
                )
        else:
            raise RuntimeError("Deformer not initialized in the model.")

        # Query implicit network
        output = net(x_c.reshape(-1, 3), cond=cond)
        sdf = output[..., 0:1].reshape(*x_c.shape[:-1], 1)
        feature = output[..., 1:].reshape(*x_c.shape[:-1], -1)
        outlier_mask = outlier_mask.reshape(*x_c.shape[:-1])

        # Set large SDF for outlier points during evaluation
        if not self.training:
            sdf[outlier_mask] = 4.0

        return {
            "sdf": sdf,
            "x_c": x_c,
            "T": T,
            "feature": feature,
            "mask": ~outlier_mask,
        }

    def forward(self, input):
        """
        Main forward pass for ReLoo model: computes SMPL outputs, renders foreground and background,
        aggregates features, and optionally computes training losses.
        """

        # Parse model input
        intrinsics, pose, uv = input["intrinsics"], input["pose"], input["uv"]
        scale = input["smpl_params"][:, 0]
        smpl_pose, smpl_shape, smpl_trans = input["smpl_pose"], input["smpl_shape"], input["smpl_trans"]

        smpl_output = self.smpl_server(scale, smpl_trans, smpl_pose, smpl_shape)
        self.scale = scale
        smpl_tfs = smpl_output['smpl_tfs']
        smpl_root_orient = smpl_pose[:, :3]

        if 'cond' in input:
            cond = {'smpl': input['cond']}
        else:
            cond = {'smpl': smpl_pose[:, 3:] / np.pi}

        # To avoid cloth overfitting to posed space.
        if self.training:
            if input['current_epoch'] < 10 or input['current_epoch'] % 50 == 0:
                cond = {'smpl': smpl_pose[:, 3:] * 0.}

        # Ray directions and camera location
        ray_dirs, cam_loc = utils.get_camera_params(uv, pose, intrinsics)
        self.current_epoch = input['current_epoch']
        cam_loc = cam_loc.expand_as(ray_dirs).reshape(-1, 3)
        ray_dirs = ray_dirs.reshape(-1, 3)

        smpl_params = {
            'scale': scale,
            'smpl_root_orient': smpl_root_orient,
            'smpl_trans': smpl_trans
        }
        
        # Add body vertices for val/test
        smpl_output['body_verts'] = None if self.training else self.mesh_v_body

        # Foreground (Human) rendering
        fg_output = self.render_fg(cam_loc, ray_dirs, cond, smpl_params, smpl_output, input['time_enc'])
        
        # Background rendering
        z_vals_bg = self.inverse_sphere_sampler.get_z_vals(ray_dirs, cam_loc, self)
        z_vals_bg = z_vals_bg / self.sdf_bounding_sphere

        if 'image_id' in input.keys():
            frame_code = self.frame_latent_encoder(input['image_id'])
        else:
            frame_code = self.frame_latent_encoder(input['idx'])

        if input['idx'] is not None:
            bg_output = self.render_bg(z_vals_bg, cam_loc, ray_dirs, {'frame': frame_code})
            bg_rgb_values = bg_output["colors"]
        else:
            bg_rgb_values = torch.ones_like(fg_output["colors"])

        # Composite foreground and background
        acc_map = fg_output["acc_map"].unsqueeze(-1)  # [N, 1]
        rgb_values = fg_output["colors"] + bg_rgb_values * (1 - acc_map)
        feature_vectors = fg_output["feature_vectors"] + bg_output["feature_vectors"] * (1 - acc_map)

        if self.training:
            output = {}

            # in shape loss for SMPL
            params = self.smpl_server.param_canonical.clone()
            params[:, -10:] = smpl_shape

            # only regularze the torso part
            verts_c = self.smpl_server(*torch.split(params, [1, 3, 72, 10], dim=1))['smpl_verts'][:, self.smpl_server.smpl_torso_body_v_idx]
            indices = torch.randperm(verts_c.shape[1], device=verts_c.device)[:1024]
            verts_c = torch.index_select(verts_c, 1, indices)
            sample_smpl_verts_c = self.sampler.get_points(verts_c)
            smpl_surface_sdf = self.body_network(sample_smpl_verts_c, cond)[..., 0:1]
            output["smpl_surface_sdf"] = smpl_surface_sdf
            output["smpl_surface_sdf_ps_gt"] = self.deformer.signed_distance(sample_smpl_verts_c[0])

            # Eikonal points for body and clothing
            grad_list = []

            # Body
            verts_body = self.mesh_v_body[None]
            sample_body = self.sampler.get_points(verts_body, global_ratio=0.)
            sample_body.requires_grad_()
            pred_body = self.body_network(sample_body, cond)[..., 0:1]
            grad_list.append(utils.compute_gradient(sample_body, pred_body))

            # Clothing
            cloth_meshes = []
            cloth_nets = []

            # Collect all available clothing meshes and networks dynamically
            for i in range(1, self.num_clothes + 1):
                mesh_attr = f"mesh_v_cloth{i}" if self.num_clothes > 1 else "mesh_v_cloth"
                net_attr = f"clothing_network{i}" if self.num_clothes > 1 else "clothing_network"
                if hasattr(self, mesh_attr) and hasattr(self, net_attr):
                    cloth_meshes.append(getattr(self, mesh_attr))
                    cloth_nets.append(getattr(self, net_attr))

            # Compute gradients for each clothing
            for mesh, net in zip(cloth_meshes, cloth_nets):
                sample_cloth = self.sampler.get_points(mesh[None], global_ratio=0.)
                sample_cloth.requires_grad_()
                pred_cloth = net(sample_cloth, cond)[..., 0:1]
                grad_list.append(utils.compute_gradient(sample_cloth, pred_cloth))

            # If no cloth templates available yet
            if not cloth_meshes and self.num_clothes > 0:
                # use body sample for warm-up
                body_pred_min = pred_body.clone()
                for i in range(1, self.num_clothes + 1):
                    net_attr = f"clothing_network{i}" if self.num_clothes > 1 else "clothing_network"
                    if hasattr(self, net_attr):
                        pred_cloth = getattr(self, net_attr)(sample_body, cond)[..., 0:1]
                        body_pred_min = torch.min(body_pred_min, pred_cloth)
                grad_list = [utils.compute_gradient(sample_body, body_pred_min)]

            # Concatenate all gradients
            grad_theta = torch.cat(grad_list, dim=1)

            # Virtual bone deformation warm-up
            if hasattr(self, "mesh_v_cloth") or hasattr(self, "mesh_v_cloth1"):
                cond_vb = {'smpl': input['cond']} if 'cond' in input else {'smpl': smpl_pose[:, 3:] / np.pi}

                v_cloth_d_smpl = self.deformer._deformer.forward_skinning(
                    self.virtual_bone.virtual_bones_pos.unsqueeze(0),
                    smpl_tfs
                )[0].detach()
                v_cloth_d_vb = self.virtual_bone.forward(
                    x=None, cond=cond_vb,
                    smpl_root_orient=smpl_root_orient,
                    smpl_trans=smpl_trans,
                    scale=scale,
                    time_enc=input['time_enc'],
                    return_nodes_only=True
                )
            else:
                v_cloth_d_smpl, v_cloth_d_vb = None, None

            # Pack outputs
            output.update({
                'rgb_values': rgb_values,
                'normal_values': fg_output["normals"],
                'index_inside': input.get('index_inside'),
                'index_outside': input.get('index_outside'),
                'grad_theta': grad_theta,
                'acc_map': fg_output["acc_map"],
                'epoch': input['current_epoch'],
                'feature_vector': feature_vectors,
                'v_cloth_d_smpl': v_cloth_d_smpl,
                'v_cloth_d_vb': v_cloth_d_vb,
            })

        else:
            fg_rgb_values = fg_output["colors"] + (1 - fg_output["acc_map"].unsqueeze(-1)) * torch.ones_like(fg_output["colors"], device=fg_output["colors"].device)
            output = {
                'acc_map': fg_output["acc_map"],
                'rgb_values': rgb_values,
                'fg_rgb_values': fg_rgb_values,
                'normal_values': fg_output["normals"],
            }

        # Per-item accumulation
        if fg_output.get("acc_by_item") is not None:
            if self.num_clothes == 2:
                output.update({
                    "acc_map_body": fg_output["acc_by_item"][:, 0],
                    "acc_map_cloth1": fg_output["acc_by_item"][:, 1],
                    "acc_map_cloth2": fg_output["acc_by_item"][:, 2],
                })
            else:
                output.update({
                    "acc_map_body": fg_output["acc_by_item"][:, 0],
                    "acc_map_cloth": fg_output["acc_by_item"][:, 1],
                })

        return output

    def render_fg(self, rays_o, rays_d, cond, smpl_params, smpl_outputs, time_enc=None):
        """
        Render the foreground (FG) by performing SDF-based volumetric rendering.

        Args:
            rays_o (torch.Tensor): Ray origins [N, 3].
            rays_d (torch.Tensor): Ray directions [N, 3].
            cond (dict): Conditioning dictionary, e.g., SMPL parameters.
            smpl_params (dict): SMPL parameter set.
            smpl_outputs (dict): SMPL model outputs.
            time_enc (torch.Tensor, optional): Time encoding for temporal conditioning.

        Returns:
            dict: {
                "colors": RGB color map,
                "normals": Aggregated surface normals,
                "acc_map": Accumulated opacity map,
                "pts_c": Canonical-space points,
                "feature_vectors": Aggregated learned features,
                "acc_by_item": Optional per-item accumulation,
                "eikonal_points": Optional eikonal sampling points
            }
        """
        # Compute sphere intersection bounds
        sphere_inters = utils.get_sphere_intersections(rays_o, rays_d, r=3.0)
        near = sphere_inters[..., 0:1].clamp_min(0.0)
        far = sphere_inters[..., 1:2]

        # Prepare SDF functions based on clothing setup

        sdf_fn = [self.model_fg.get_sdf_fn(
            self, cond, smpl_params, smpl_outputs, time_enc, self.body_network
        )]

        if self.num_clothes == 1:
            sdf_fn.append(
                self.model_fg.get_sdf_fn(
                    self, cond, smpl_params, smpl_outputs, time_enc,
                    self.clothing_network, using_virtual_bone=True
                )
            )

        elif self.num_clothes == 2:
            sdf_fn.extend([
                self.model_fg.get_sdf_fn(
                    self, cond, smpl_params, smpl_outputs, time_enc,
                    self.clothing_network1
                ),
                self.model_fg.get_sdf_fn(
                    self, cond, smpl_params, smpl_outputs, time_enc,
                    self.clothing_network2, using_virtual_bone=True
                ),
            ])

        # Perform forward rendering pass
        fg_output = self.model_fg(rays_d, rays_o, near, far, sdf_fn)
        ray_shape = fg_output["ray_samples"].shape

        pts_c = fg_output["points_cano"]
        feature_vectors = fg_output["feature"]
        gradients = fg_output.get("gradients", None)

        # Normal computation
        normals = nn.functional.normalize(gradients, dim=-1, eps=1e-6)

        fg_rgb = self.rendering_network(
            pts_c, normals, None, cond['smpl'], feature_vectors
        )

        # Weighted aggregation
        weights = fg_output["weights"]
        bg_transmittance = fg_output["bg_transmittance"]

        colors = torch.sum(weights * fg_rgb.reshape(*ray_shape, 3), dim=-2)
        normals = torch.sum(weights * normals.reshape(*ray_shape, 3), dim=-2)
        features = torch.sum(
            weights * feature_vectors[..., -256:].reshape(*ray_shape, -1), dim=-2
        )

        # Assemble output dictionary
        return {
            "colors": colors,
            "normals": normals,
            "acc_map": 1 - bg_transmittance.squeeze(-1),
            "pts_c": pts_c,
            "feature_vectors": features,
            "acc_by_item": fg_output.get("acc_by_item"),
            "eikonal_points": fg_output.get("eikonal_points"),
        }

    def render_bg(self, z_vals_bg, cam_loc, ray_dirs, cond):
        """
        Render the background (BG) using the background implicit network.

        Args:
            z_vals_bg (torch.Tensor): Depth samples along rays [N_rays, N_samples].
            cam_loc (torch.Tensor): Camera locations [N_rays, 3].
            ray_dirs (torch.Tensor): Ray directions [N_rays, 3].
            cond (dict): Conditioning dictionary (e.g., frame info).

        Returns:
            dict: {
                "colors": Aggregated background RGB map,
                "feature_vectors": Aggregated learned BG features,
            }
        """
        # Prepare ray samples in 3D
        shape = z_vals_bg.shape  # [N_rays, N_samples]

        bg_dirs = ray_dirs.unsqueeze(-2)  # [N_rays, 1, 3]
        bg_locs = cam_loc.unsqueeze(-2)   # [N_rays, 1, 3]

        bg_points = self.depth2pts_outside(bg_locs, bg_dirs, torch.flip(z_vals_bg, dims=[-1]))
        bg_dirs = bg_dirs.expand_as(bg_points[..., :3])

        # Query the background implicit network
        bg_output = self.bg_implicit_network(bg_points.reshape(-1, 4), cond).squeeze(0)
        bg_sdf = bg_output[:, :1]
        bg_features = bg_output[:, 1:]

        # Compute background colors
        bg_rgb = self.bg_rendering_network(
            None, None, bg_dirs.reshape(-1, 3), None, bg_features, cond["frame"]
        )

        if bg_rgb.shape[-1] == 4:
            bg_rgb = (1 - bg_rgb[..., -1]) * bg_rgb[..., :3]
        bg_rgb = bg_rgb.reshape(*shape, 3)

        # Compute volume rendering weights
        dists = z_vals_bg[:, 1:] - z_vals_bg[:, :-1]
        dists = torch.cat([dists, torch.ones_like(z_vals_bg[:, :1]) * 1e10], -1)

        bg_sdf = bg_sdf.reshape(shape)
        bg_features = bg_features.reshape(*shape, -1)

        weights, _ = self.volume_rendering(dists, self.bg_density(bg_sdf))

        # Weighted aggregation
        colors = torch.sum(bg_rgb * weights[..., None], dim=-2)
        features = torch.sum(bg_features[..., -256:] * weights[..., None], dim=-2)

        # Return results
        return {
            "colors": colors,
            "feature_vectors": features,
        }


    def volume_rendering(self, dists, density):
        """
        Compute volume rendering weights from distances and density values.

        Args:
            dists (torch.Tensor): Distance between consecutive samples along rays [N_rays, N_samples].
            density (torch.Tensor): Density values at each sample [N_rays, N_samples].

        Returns:
            weights (torch.Tensor): Volume rendering weights [N_rays, N_samples].
            bg_transmittance (torch.Tensor): Transmittance to background [N_rays].
        """

        # Compute free energy for each segment
        free_energy = dists * density # [N_rays, N_samples]

        # Shifted free energy for cumulative product
        shifted_free_energy = torch.cat([torch.zeros_like(free_energy[..., :1]), free_energy], dim=-1)

        # Alpha: probability of termination in each segment
        alpha = 1 - torch.exp(-free_energy) # [N_rays, N_samples]

        # Transmittance: cumulative product of survival probabilities
        transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))

        # Compute final weights
        weights = alpha * transmittance[..., :-1] # weight for each sample
        return weights, transmittance[..., -1]
    
    def depth2pts_outside(self, ray_o, ray_d, depth):
        """
        Convert depth samples outside a bounding sphere to 3D points along rays.

        Args:
            ray_o (torch.Tensor): Ray origins [..., 3].
            ray_d (torch.Tensor): Ray directions [..., 3].
            depth (torch.Tensor): Depth values [...], normalized inside [0, 1].

        Returns:
            torch.Tensor: Points with shape [..., 4], where last dimension is depth.
        """

        # Compute intersection with bounding sphere
        o_dot_d = torch.sum(ray_d * ray_o, dim=-1)
        under_sqrt = o_dot_d ** 2 - ((ray_o ** 2).sum(-1) - self.sdf_bounding_sphere ** 2)

        d_sphere = torch.sqrt(under_sqrt) - o_dot_d
        p_sphere = ray_o + d_sphere.unsqueeze(-1) * ray_d
        
        # Compute midpoint vector for rotation
        p_mid = ray_o - o_dot_d.unsqueeze(-1) * ray_d
        p_mid_norm = torch.norm(p_mid, dim=-1)

        # Compute rotation axis and angles
        rot_axis = torch.cross(ray_o, p_sphere, dim=-1)
        rot_axis = rot_axis / torch.norm(rot_axis, dim=-1, keepdim=True)
        
        phi = torch.asin(p_mid_norm / self.sdf_bounding_sphere)
        theta = torch.asin(p_mid_norm * depth)  # depth is inside [0, 1]
        rot_angle = (phi - theta).unsqueeze(-1)  # [..., 1]

        # Now rotate p_sphere
        p_sphere_new = p_sphere * torch.cos(rot_angle) + \
                       torch.cross(rot_axis, p_sphere, dim=-1) * torch.sin(rot_angle) + \
                       rot_axis * torch.sum(rot_axis * p_sphere, dim=-1, keepdim=True) * (1. - torch.cos(rot_angle))
        p_sphere_new = p_sphere_new / torch.norm(p_sphere_new, dim=-1, keepdim=True)
        pts = torch.cat((p_sphere_new, depth.unsqueeze(-1)), dim=-1)
        return pts