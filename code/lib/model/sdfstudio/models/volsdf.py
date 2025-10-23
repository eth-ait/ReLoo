import torch
import torch.nn as nn
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict
from ..samplers.volsdf import ErrorBoundedSampler
from ..utils.rays import RayBundle
from ..fields.sdf_field import LaplaceDensity


@dataclass
class VolSDFModelConfig:
    """VolSDF Model Config"""
    num_samples: int = 64
    """Number of samples after error bounded sampling"""
    num_samples_eval: int = 128
    """Number of samples per iteration used in error bounded sampling"""
    num_samples_extra: int = 32
    """Number of uniformly sampled points for training"""

class VolSDFModel(nn.Module):
    config = VolSDFModelConfig()

    def __init__(self) -> None:
        super().__init__()
        self.sampler = ErrorBoundedSampler()
        self.laplace_density = LaplaceDensity(init_val=0.1, beta_min=0.0001)

    def get_sdf_fn(self, model, cond, smpl_params, smpl_outputs, time_enc, net, using_virtual_bone=False):
        def sdf_fn(samples, sdf_only=True, requires_grad=False):
            scale = smpl_outputs["scale"]
            points = samples.frustums.get_start_positions()
            if requires_grad:
                points.requires_grad_(True)
                with torch.enable_grad():
                    outputs = model.sdf_func_with_deformer(
                        points.view(-1, 3), cond, smpl_params, smpl_outputs, time_enc, net, using_virtual_bone
                    )
                    gradients = torch.autograd.grad(
                        outputs=outputs["sdf"],
                        inputs=outputs["x_c"],
                        grad_outputs=torch.ones_like(outputs["sdf"]),
                        create_graph=self.training,
                        retain_graph=self.training,
                        only_inputs=True,
                    )[0]
                    gradients = (outputs["T"][0, :, :3, :3] @ gradients[..., None]).squeeze(-1)
                    outputs["gradients"] = gradients / scale
            else:
                outputs = model.sdf_func_with_deformer(
                    points.view(-1, 3), cond, smpl_params, smpl_outputs, time_enc, net, using_virtual_bone
                )

            shape = points.shape[:-1]
            ret = {
                "sdf": outputs["sdf"].reshape(*shape, 1),
                "x_c": outputs["x_c"].reshape(*shape, 3),
                "feature": outputs["feature"].reshape(*shape, -1),
                "mask": outputs["mask"].reshape(*shape),
                "gradients": outputs["gradients"].reshape(*shape, 3) if requires_grad else None,
            }
            if sdf_only:
                return ret["sdf"]
            else:
                return ret
        return sdf_fn

    def forward(self, ray_d, ray_o, near, far, sdf_fn):
        ray_bundle = RayBundle(
            origins=ray_o,
            directions=ray_d,
            nears=near,
            fars=far,
            pixel_area=None,
            camera_indices=None,
            metadata=None,
            times=None,
        )

        # Check if sdf_fn is a list (indicating multiple SDFs)
        if isinstance(sdf_fn, list):
            # Ensure the number of SDFs is supported (1, 2, or 3)
            assert len(sdf_fn)>=1 and len(sdf_fn) <= 3

            # Initialize containers for results, ray samples, and SDF values
            results = defaultdict(list)
            ray_samples_list, sdf_list = [], []
            for item_id, sdf_fn_i in enumerate(sdf_fn):
                ray_samples_i, _ = self.sampler(
                    ray_bundle, density_fn=self.laplace_density, sdf_fn=sdf_fn_i
                )
                outputs_i = sdf_fn_i(ray_samples_i, sdf_only=False, requires_grad=True)
                for k, v in outputs_i.items():
                    results[k].append(v)
                ray_samples_list.append(ray_samples_i)
                sdf_list.append(outputs_i["sdf"])
                results["items"].append(torch.ones_like(outputs_i["sdf"]) * item_id)

            ray_samples, sorted_index = self.sampler.merge_ray_samples(ray_bundle, *ray_samples_list)

            d1 = torch.arange(len(sorted_index), device=sorted_index.device).unsqueeze(-1).expand_as(sorted_index)
            for k, v in results.items():
                results[k] = torch.cat(v, dim=1)[d1, sorted_index]
            density = self.laplace_density(results["sdf"])
            weights, transmittance = ray_samples.get_weights_and_transmittance(density)
            acc = []
            for i in range(len(sdf_fn)):
                weight_idx = (results["items"] == i)
                weight_i = weights[weight_idx].reshape(sdf_list[i].shape[:-1])
                acc.append(torch.sum(weight_i, dim=-1))
            acc = torch.stack(acc, dim=-1)

            bg_transmittance = transmittance[:, -1, :]
            return {
                "ray_samples": ray_samples,
                "points_cano": results["x_c"],
                "points_mask": results["mask"],
                "weights": weights,
                "feature": results["feature"],
                "bg_transmittance": bg_transmittance,
                "acc_by_item": acc,
                "gradients": results["gradients"],
            }
        else:
            ray_samples, _ = self.sampler(
                ray_bundle, density_fn=self.laplace_density, sdf_fn=sdf_fn
            )
            results = sdf_fn(ray_samples, sdf_only=False, requires_grad=True)
            density = self.laplace_density(results["sdf"])
            weights, transmittance = ray_samples.get_weights_and_transmittance(density)
            bg_transmittance = transmittance[:, -1, :]
            return {
                "ray_samples": ray_samples,
                "points_cano": results["x_c"],
                "points_mask": results["mask"],
                "feature": results["feature"],
                "weights": weights,
                "bg_transmittance": bg_transmittance,
                "gradients": results["gradients"],
            }

    def get_metrics_dict(self) -> Dict:
        metrics_dict = {}
        if self.training: # training statics
            metrics_dict["beta"] = self.laplace_density.get_beta()
            metrics_dict["alpha"] = 1.0 / self.laplace_density.get_beta()
        return metrics_dict
