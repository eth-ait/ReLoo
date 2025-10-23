import pytorch_lightning as pl
import torch.optim as optim
from lib.model.reloo import ReLoo
from lib.model.body_model_params import BodyModelParams
import cv2
import torch
from lib.model.loss import Loss
import hydra
import os
import numpy as np
from lib.utils.meshing import generate_mesh
from tqdm import tqdm
import trimesh
from lib.utils import utils
from collections import defaultdict
from lib.utils.meshing import simplify_mesh
import glob

class ReLooModel(pl.LightningModule):
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt

        # Dataset and training info
        num_training_frames = opt.dataset.metainfo.end_frame - opt.dataset.metainfo.start_frame
        num_clothes = opt.dataset.metainfo.num_clothes
        self.num_clothes = num_clothes
        self.gender = opt.dataset.metainfo.gender
        self.betas_path = os.path.join(hydra.utils.to_absolute_path('..'), 'data', opt.dataset.metainfo.data_dir, 'mean_shape.npy')
        self.start_frame = opt.dataset.metainfo.start_frame
        self.end_frame = opt.dataset.metainfo.end_frame
        self.training_indices = list(range(self.start_frame, self.end_frame))

        # Main ReLoo model
        self.model = ReLoo(opt.model, self.betas_path, self.gender, num_training_frames, num_clothes)

        # Trainable SMPL params
        self.body_model_params = BodyModelParams(num_training_frames, model_type='smpl')
        self.load_body_model_params()
        optim_params = self.body_model_params.param_names
        for param_name in optim_params:
            self.body_model_params.set_requires_grad(param_name, requires_grad=True)
        
        self.training_modules = ['model', 'body_model_params']
        self.loss = Loss(opt.model.loss)
        
    def load_body_model_params(self):
        """Load and initialize SMPL body model parameters from dataset."""
        data_root = os.path.join('../data', self.opt.dataset.metainfo.data_dir)
        data_root = hydra.utils.to_absolute_path(data_root)

        # Initialize container for parameters
        body_model_params = {param_name: [] for param_name in self.body_model_params.param_names}

        # Load camera info
        camera_dict = np.load(os.path.join(data_root, 'cameras_normalize.npz'))
        scale = camera_dict['scale_mat_0'][0, 0]
        self.human_scale = torch.tensor(1 / scale, dtype=torch.float32, device=self.device)[None]
        
        # Load shapes and poses
        betas_np = np.load(os.path.join(data_root, 'mean_shape.npy'))[None]
        poses_np = np.load(os.path.join(data_root, 'poses.npy'))[self.training_indices]
        trans_np = np.load(os.path.join(data_root, 'normalize_trans.npy'))[self.training_indices]

        # Convert to tensors
        body_model_params['betas'] = torch.tensor(betas_np, dtype=torch.float32, device=self.device)
        body_model_params['global_orient'] = torch.tensor(poses_np[:, :3], dtype=torch.float32, device=self.device)
        body_model_params['body_pose'] = torch.tensor(poses_np[:, 3:], dtype=torch.float32, device=self.device)
        body_model_params['transl'] = torch.tensor(trans_np, dtype=torch.float32, device=self.device)

        # Initialize parameters in body model
        for param_name, value in body_model_params.items():
            self.body_model_params.init_parameters(param_name, value, requires_grad=False)

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        base_lr = self.opt.model.learning_rate
        params = []

        # Helper: add params safely
        def add_module_params(module_name, lr_scale=1.0):
            module = getattr(self.model, module_name, None)
            if module is not None:
                params.append({'params': module.parameters(), 'lr': base_lr * lr_scale})

        skip_prefixes = ['deformer', 'virtual_bone']
        if self.num_clothes == 2:
            skip_prefixes += ['clothing_network1', 'clothing_network2']
        elif self.num_clothes == 1:
            skip_prefixes += ['clothing_network']
        
        # Human model with base learning rate
        for name, param in self.model.named_parameters():
            if not any(name.startswith(pfx) for pfx in skip_prefixes):
                params.append({'params': param, 'lr': base_lr})

        # Clothing models with doubled learning rates
        if self.num_clothes == 2:
            add_module_params('clothing_network1', lr_scale=2.0)
            add_module_params('clothing_network2', lr_scale=2.0)
        elif self.num_clothes == 1:
            add_module_params('clothing_network', lr_scale=2.0)

        add_module_params('deformer', lr_scale=0.1)
        add_module_params('virtual_bone', lr_scale=1.0)


        # SMPL model parameters with reduced learning rate
        params.append({
            'params': self.body_model_params.parameters(),
            'lr': base_lr * 0.1
        })

        # Optimizer and scheduler
        self.optimizer = optim.Adam(params, lr=base_lr, eps=1e-8)
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, 
            milestones=self.opt.model.sched_milestones, 
            gamma=self.opt.model.sched_factor
        )

        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "epoch",
                "frequency": 1,
            },

        }

    def training_step(self, batch):
        """Single training step for ReLooModel."""
        inputs, targets = batch
        batch_idx = inputs["idx"]

        # Prepare SMPL/body model parameters
        body_model_params = self.body_model_params(batch_idx)
        inputs.update({
            "smpl_pose": torch.cat(
                (body_model_params["global_orient"], body_model_params["body_pose"]), dim=1
            ),
            "smpl_shape": body_model_params["betas"],
            "smpl_trans": body_model_params["transl"],
            "current_epoch": self.current_epoch,
        })

        # Forward pass
        model_outputs = self.model(inputs)

        # Compute loss
        loss_output = self.loss(model_outputs, targets)
        total_loss = loss_output["loss"]

        # Logging
        for name, value in loss_output.items():
            log_args = dict(prog_bar=(name != "loss"), on_step=True)
            self.log(name, value.detach(), **log_args)
        
        # Log model metrics
        metrics = self.model.model_fg.get_metrics_dict()
        for name, value in metrics.items():
            self.log(name, value.detach(), prog_bar=True, on_step=True)

        return total_loss

    def on_train_epoch_end(self):
        """
        Update canonical meshes and virtual bones at the end of certain epochs.
        Triggered every 10 epochs (9, 19, 29, ...).
        """
        # Only run on specific epochs
        if self.current_epoch == 0 or self.current_epoch % 10 != 9:
            return

        # Conditioning vector for canonical query
        cond = {'smpl': torch.zeros(1, 69, device='cuda')}
        verts = self.model.smpl_server.verts_c[0]  # canonical SMPL vertices

        # Generate body mesh
        mesh_body = self._generate_and_simplify_mesh(
            lambda x: self.query_oc(x, cond, "body"), verts
        )

        # Generate clothing meshes
        cloth_meshes = []
        for i, _ in enumerate(self.model.clothing_network_list):
            mode = f"cloth{i+1}"
            mesh = self._generate_and_simplify_mesh(lambda x: self.query_oc(x, cond, mode), verts)
            cloth_meshes.append(mesh)

        # Export simplified meshes
        os.makedirs("simplified", exist_ok=True)
        mesh_body.export(f"simplified/body.ply")
        for i, mesh in enumerate(cloth_meshes):
            mesh.export(f"simplified/cloth{i+1}.ply")

        # Prepare SMPL vertices for deformer update
        params = self.model.smpl_server.param_canonical.clone()
        params[:, -10:] = self.body_model_params.betas.weight.detach()
        smpl_verts = self.model.smpl_server.forward(
            *torch.split(params, [1, 3, 72, 10], dim=1)
        )['smpl_verts']

        # Generate virtual bones mesh
        os.makedirs("virtual_bones", exist_ok=True)
        base_mesh = cloth_meshes[-1]
        virtual_bones_mesh = simplify_mesh(base_mesh, self.model.num_vb)
        virtual_bones_mesh.export(f"virtual_bones/virtual_bones.ply")
        np.save(f"virtual_bones/virtual_bones.npy", virtual_bones_mesh.vertices)

        # Update the model template
        self.model.update_template(mesh_body, *cloth_meshes)

        # Update deformer with SMPL vertices
        self.model.deformer.update(verts=smpl_verts)

    def _generate_and_simplify_mesh(self, query_fn, verts, target_faces: int = 2048):
        """
        Generate a mesh using a query function and simplify it.

        Args:
            query_fn (callable): Function mapping points to occupancy/SDF values
            verts (torch.Tensor): SMPL template vertices
            target_faces (int): Number of faces after simplification

        Returns:
            Simplified mesh object
        """
        mesh = generate_mesh(query_fn, verts, point_batch=10000, res_up=2)
        simplified_mesh = simplify_mesh(mesh, target_faces)
        return simplified_mesh

    def query_oc(self, x, cond, mode: str = "cano", return_features: bool = False):
        """
        Query occupancy/SDF predictions from the implicit and clothing networks.

        Args:
            x (torch.Tensor): Input 3D points of shape (..., 3)
            cond (dict): Conditioning input (e.g., SMPL pose, shape, etc.)
            mode (str): One of ["cano", "body", "cloth1", "cloth2", ...]
            return_features (bool): Whether to also return intermediate features

        Returns:
            dict with keys:
                'sdf': Tensor (N, 1)
                'features': Tensor (N, F) if return_features=True
        """
        x = x.reshape(-1, 3)

        # Helper function to forward a network and extract sdf + optional features
        def forward_network(net):
            out = net(x, cond)
            sdf = out[..., 0:1].reshape(-1, 1)
            features = out[..., 1:] if return_features else None
            return sdf, features

        mnfld_pred, result_features = None, None

        # Canonical mode: combine body and all clothing networks
        if mode == "cano":
            mnfld_pred, _ = forward_network(self.model.body_network)
            for net in self.model.clothing_network_list:
                cloth_pred, _ = forward_network(net)
                mnfld_pred = torch.min(mnfld_pred, cloth_pred)

        # Body-only mode: use the implicit network
        elif mode == "body":
            mnfld_pred, result_features = forward_network(self.model.body_network)

        # Single clothing network mode: e.g., "cloth1", "cloth2"
        elif mode.startswith("cloth"):
            idx = int(mode.replace("cloth", "")) - 1
            if idx < 0 or idx >= len(self.model.clothing_network_list):
                raise ValueError(f"Invalid mode '{mode}': model has {len(self.model.clothing_network_list)} clothing networks.")
            mnfld_pred, result_features = forward_network(self.model.clothing_network_list[idx])

        # Error for unsupported modes
        else:
            raise ValueError(f"Unsupported mode '{mode}'")

        # Prepare output dictionary
        result = {"sdf": mnfld_pred}
        if return_features and result_features is not None:
            result["features"] = result_features

        return result

    def query_od(self, x, cond, smpl_params, smpl_outputs, time_enc, mode="cano"):
        """Query deformed SDFs for body and clothing networks."""
        x = x.reshape(-1, 3)

        # Body SDF (no virtual bone)
        body_sdf = self.model.sdf_func_with_deformer(
            x, cond, smpl_params, smpl_outputs, time_enc,
            self.model.body_network, using_virtual_bone=False
        )['sdf'][:, 0:1]

        # Clothing SDFs
        clothing_sdfs = []
        for i, clothing_net in enumerate(self.model.clothing_network_list):
            use_vb = (i == len(self.model.clothing_network_list) - 1) # assume last clothing is loose and uses virtual bones
            sdf = self.model.sdf_func_with_deformer(
                x, cond, smpl_params, smpl_outputs, time_enc,
                clothing_net, using_virtual_bone=use_vb
            )['sdf'][:, 0:1]
            clothing_sdfs.append(sdf)

        # Combine SDFs
        if clothing_sdfs:
            total_sdf = torch.min(torch.stack([body_sdf] + clothing_sdfs), dim=0)[0]
        else:
            total_sdf = body_sdf

        # Mode-specific return
        if mode == "cano":
            return {'sdf': total_sdf}
        elif mode == "body":
            return {'sdf': body_sdf}
        elif mode.startswith("cloth"):
            idx = int(mode[-1]) - 1 if len(mode) > 5 and mode[-1].isdigit() else 0
            return {'sdf': clothing_sdfs[idx]}

    def validation_step(self, batch, *args, **kwargs):
        inputs, targets = batch
        inputs['current_epoch'] = self.current_epoch

        # Prepare SMPL parameters
        body_model_params = self.body_model_params(inputs['image_id'])
        inputs['smpl_pose'] = torch.cat((body_model_params['global_orient'], body_model_params['body_pose']), dim=1)
        inputs['smpl_shape'] = body_model_params['betas']
        inputs['smpl_trans'] = body_model_params['transl']
        cond = {'smpl': inputs["smpl_pose"][:, 3:] * 0.}

        # Extract canonical meshes
        os.makedirs("meshes", exist_ok=True)
        smpl_verts_c = self.model.smpl_server.verts_c[0]
        mode_list = ["cano", "body", "cloth1", "cloth2"] if self.num_clothes == 2 else ["cano", "body", "cloth"]

        for mode in mode_list:
            if mode not in ["cano"] and not hasattr(self.model, 'clothing_network') and not hasattr(self.model, 'clothing_network1'):
                continue
            try:
                mesh = generate_mesh(lambda x: self.query_oc(x, cond, mode), smpl_verts_c, point_batch=10000, res_up=3)
                mesh.export(f"meshes/{self.current_epoch:05d}_{mode}.ply")
            except Exception as e:
                print(f"Fail to generate {mode} mesh: {e}")

        # Posed rendering
        img_size = targets["img_size"]
        n_pixels = min(targets['pixel_per_batch'], img_size[0] * img_size[1])
        split = utils.split_input(inputs, targets["total_pixels"][0], n_pixels=n_pixels)

        res = defaultdict(list)
        for s in tqdm(split, leave=False):
            out = self.model(s)
            for k, v in out.items():
                if isinstance(v, torch.Tensor):
                    out[k] = v.detach().cpu()

            res["rgb_values"].append(out["rgb_values"])
            res["normal_values"].append(out["normal_values"])
            res["fg_rgb_values"].append(out["fg_rgb_values"])
            if self.num_clothes == 2:
                for key in ["acc_map_cloth1", "acc_map_cloth2"]:
                    if key in out: res[key].append(out[key])
            elif self.num_clothes == 1:
                if "acc_map_cloth" in out: res["acc_map_cloth"].append(out["acc_map_cloth"])
            if "acc_map_body" in out: res["acc_map_body"].append(out["acc_map_body"])

        for k, v in res.items():
            res[k] = torch.cat(v, dim=0).reshape(*img_size, -1).squeeze(-1)

        # Save foreground rendering
        os.makedirs("fg_rendering", exist_ok=True)
        fg_rgb_pred = (res["fg_rgb_values"].numpy() * 255).astype(np.uint8)
        cv2.imwrite(f"fg_rendering/{self.current_epoch:05d}.png", fg_rgb_pred[:, :, ::-1])

        # Save full rendering
        os.makedirs("rendering", exist_ok=True)
        rgb_gt = targets["rgb"].reshape(*img_size, -1).cpu()
        rgb_pred = res["rgb_values"].cpu()
        rgb = (torch.cat([rgb_gt, rgb_pred], dim=0).numpy() * 255).astype(np.uint8)
        cv2.imwrite(f"rendering/{self.current_epoch:05d}.png", rgb[:, :, ::-1])

        # Save normal maps and virtual bones
        os.makedirs("normal", exist_ok=True)
        normal = ((res["normal_values"].cpu().numpy() + 1) * 127.5).astype(np.uint8)[:, :, ::-1]
        cv2.imwrite(f"normal/{self.current_epoch:05d}.png", normal)

        # Save masks
        os.makedirs("mask", exist_ok=True)
        mask = np.zeros_like(normal)
        if self.num_clothes == 2:
            if "acc_map_cloth1" in res and "acc_map_cloth2" in res:
                mask[:, :, 0] = (res["acc_map_body"].cpu().numpy() * 255).astype(np.uint8)
                mask[:, :, 1] = (res["acc_map_cloth1"].cpu().numpy() * 255).astype(np.uint8)
                mask[:, :, 2] = (res["acc_map_cloth2"].cpu().numpy() * 255).astype(np.uint8)
        elif self.num_clothes == 1:
            if "acc_map_cloth" in res:
                mask[:, :, 0] = (res["acc_map_body"].cpu().numpy() * 255).astype(np.uint8)
                mask[:, :, 1] = (res["acc_map_cloth"].cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(f"mask/{self.current_epoch:05d}.png", mask)

        # Force garbage collection
        torch.cuda.empty_cache()
    
    def test_step(self, batch, *args, **kwargs):
        inputs, targets, pixel_per_batch, total_pixels, idx = batch
        self.model.current_epoch = self.current_epoch

        num_splits = (total_pixels + pixel_per_batch - 1) // pixel_per_batch
        results = []

        # Extract SMPL parameters
        body_model_params = self.body_model_params(inputs['idx'])
        smpl_shape = body_model_params['betas']
        if smpl_shape.dim() == 1:
            smpl_shape = smpl_shape.unsqueeze(0)
        smpl_trans = body_model_params['transl']
        smpl_pose = torch.cat((body_model_params['global_orient'], body_model_params['body_pose']), dim=1)
        smpl_root_orient = smpl_pose[:, :3]

        scale = inputs["smpl_params"][:, 0:1]

        smpl_outputs = self.model.smpl_server(scale, smpl_trans, smpl_pose, smpl_shape)
        if 'body_verts' not in smpl_outputs:
            smpl_outputs['body_verts'] = self.model.mesh_v_body
        smpl_tfs = smpl_outputs['smpl_tfs']
        smpl_verts = smpl_outputs['smpl_verts']

        smpl_params_dict = {
            'scale': scale,
            'smpl_root_orient': smpl_root_orient,
            'smpl_trans': smpl_trans
        }

        cond = {'smpl': smpl_pose[:, 3:] / np.pi}

        # Prepare directories
        os.makedirs("test_mesh", exist_ok=True)
        os.makedirs("test_mask", exist_ok=True)
        os.makedirs("test_rendering", exist_ok=True)
        os.makedirs("test_fg_rendering", exist_ok=True)
        os.makedirs("test_normal", exist_ok=True)

        smpl_verts_c = self.model.smpl_server.verts_c[0]

        # Generate meshes
        modes = ["body", "cano"] if self.num_clothes == 1 else ["body", "cloth1", "cloth2", "cano"]

        for mode in modes:
            cond = {'smpl': smpl_pose[:, 3:] / np.pi}

            if mode == "cano":
                virtual_bones = trimesh.load(sorted(glob.glob("simplified/*.ply"))[-1]).vertices
                self.model.virtual_bone.update_virtual_bones(virtual_bones)
                self.model.virtual_bone.update_K(25)
                self.model.deformer._deformer.update_K(3)
                mesh_deformed = generate_mesh(
                    lambda x: self.query_od(x, cond, smpl_params_dict, smpl_outputs, inputs['time_enc'], mode=mode),
                    smpl_verts[0], point_batch=10000, res_up=4
                )
            elif mode == "body":
                mesh_canonical = generate_mesh(lambda x: self.query_oc(x, cond, mode), smpl_verts_c,
                                            point_batch=10000, res_up=4)
                mesh_canonical.export(f"test_mesh/{int(idx.cpu().numpy()):05d}_{mode}_canonical.ply")
                self.model.deformer._deformer.update_K(7)
                mesh_v_deformed = self.model.deformer._deformer.forward_skinning(
                    torch.tensor(mesh_canonical.vertices).cuda().float().unsqueeze(0),
                    smpl_tfs
                ).squeeze(0)
                mesh_deformed = trimesh.Trimesh(vertices=mesh_v_deformed.cpu().numpy(),
                                                faces=mesh_canonical.faces, process=False)
            else:  # cloth / cloth1 / cloth2
                virtual_bones = trimesh.load(sorted(glob.glob("simplified/*.ply"))[-1]).vertices
                self.model.virtual_bone.update_virtual_bones(virtual_bones)
                self.model.virtual_bone.update_K(25)
                mesh_deformed = generate_mesh(
                    lambda x: self.query_od(x, cond, smpl_params_dict, smpl_outputs, inputs['time_enc'], mode=mode),
                    smpl_verts[0], point_batch=10000, res_up=4
                )

            mesh_deformed.export(f"test_mesh/{int(idx.cpu().numpy()):05d}_{mode}_deformed.ply")

        # Process image batches
        for i in range(num_splits):
            indices = list(range(i * pixel_per_batch, min((i + 1) * pixel_per_batch, total_pixels)))
            batch_inputs = {
                "uv": inputs["uv"][:, indices],
                "intrinsics": inputs['intrinsics'],
                "pose": inputs['pose'],
                "smpl_params": inputs["smpl_params"],
                "smpl_pose": torch.cat((body_model_params['global_orient'], body_model_params['body_pose']), dim=1),
                "smpl_shape": body_model_params['betas'],
                "smpl_trans": body_model_params['transl'],
                "idx": inputs.get("idx", None),
                "time_enc": inputs.get("time_enc", None),
                "current_epoch": self.current_epoch
            }

            batch_targets = {
                "rgb": targets.get("rgb")[:, indices].detach().clone() if 'rgb' in targets else None,
                "img_size": targets["img_size"]
            }

            with torch.no_grad():
                model_outputs = self.model(batch_inputs)

            output_dict = {
                "rgb_values": model_outputs["rgb_values"].detach().clone(),
                "fg_rgb_values": model_outputs["fg_rgb_values"].detach().clone(),
                "normal_values": model_outputs["normal_values"].detach().clone(),
                "acc_map_body": model_outputs["acc_map_body"].detach().clone(),
                **batch_targets
            }

            if self.num_clothes == 2:
                output_dict.update({
                    "acc_map_cloth1": model_outputs["acc_map_cloth1"].detach().clone(),
                    "acc_map_cloth2": model_outputs["acc_map_cloth2"].detach().clone()
                })
            elif self.num_clothes == 1:
                output_dict.update({
                    "acc_map_cloth": model_outputs["acc_map_cloth"].detach().clone()
                })

            results.append(output_dict)

        # Aggregate results
        img_size = results[0]["img_size"]
        rgb_pred = torch.cat([r["rgb_values"] for r in results], dim=0).reshape(*img_size, -1)
        fg_rgb_pred = torch.cat([r["fg_rgb_values"] for r in results], dim=0).reshape(*img_size, -1)
        normal_pred = torch.cat([r["normal_values"] for r in results], dim=0).reshape(*img_size, -1)
        normal_pred = (normal_pred + 1) / 2
        pred_mask_body = torch.cat([r["acc_map_body"] for r in results], dim=0).reshape(*img_size, -1)

        pred_mask = np.zeros_like(fg_rgb_pred.cpu().numpy().squeeze())
        if self.num_clothes == 2:
            pred_mask[:, :, 1] = torch.cat([r["acc_map_cloth1"] for r in results], dim=0).reshape(*img_size, -1).squeeze(-1).cpu().numpy() * 255
            pred_mask[:, :, 2] = torch.cat([r["acc_map_cloth2"] for r in results], dim=0).reshape(*img_size, -1).squeeze(-1).cpu().numpy() * 255
        elif self.num_clothes == 1:
            pred_mask[:, :, 1] = torch.cat([r["acc_map_cloth"] for r in results], dim=0).reshape(*img_size, -1).squeeze(-1).cpu().numpy() * 255
        pred_mask[:, :, 0] = (pred_mask_body.cpu().numpy().squeeze() * 255).astype(np.uint8)

        # Compose final images
        if results[0]['rgb'] is not None:
            rgb_gt = torch.cat([r["rgb"] for r in results], dim=1).reshape(*img_size, -1) 
            rgb = (torch.cat([rgb_gt, rgb_pred], dim=0).cpu().numpy() * 255).astype(np.uint8)
        else:
            rgb = (rgb_pred.cpu().numpy() * 255).astype(np.uint8)

        fg_rgb = (fg_rgb_pred.cpu().numpy() * 255).astype(np.uint8)
        normal = (normal_pred.cpu().numpy() * 255).astype(np.uint8)

        # Save outputs
        cv2.imwrite(f"test_mask/{int(idx.cpu().numpy()):04d}.png", pred_mask)
        cv2.imwrite(f"test_rendering/{int(idx.cpu().numpy()):04d}.png", rgb[:, :, ::-1])
        cv2.imwrite(f"test_normal/{int(idx.cpu().numpy()):04d}.png", normal[:, :, ::-1])
        cv2.imwrite(f"test_fg_rendering/{int(idx.cpu().numpy()):04d}.png", fg_rgb[:, :, ::-1])