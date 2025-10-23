from .ray_samplers import Sampler, UniformSampler, PDFSampler
from typing import Optional, Callable, Tuple, Union
import torch
from ..utils.rays import RayBundle, RaySamples


class ErrorBoundedSampler(Sampler):
    """VolSDF's error bounded sampler that uses a sdf network to generate samples."""

    def __init__(
        self,
        num_samples: int = 64,
        num_samples_eval: int = 128,
        num_samples_extra: int = 32,
        eps: float = 0.1,
        beta_iters: int = 10,
        max_total_iters: int = 5,
        add_tiny: float = 1e-6,
        single_jitter: bool = False,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.num_samples_eval = num_samples_eval
        self.num_samples_extra = num_samples_extra
        self.eps = eps
        self.beta_iters = beta_iters
        self.max_total_iters = max_total_iters
        self.add_tiny = add_tiny
        self.single_jitter = single_jitter

        # samplers
        self.uniform_sampler = UniformSampler(single_jitter=single_jitter)
        self.pdf_sampler = PDFSampler(
            include_original=False,
            single_jitter=single_jitter,
            histogram_padding=1e-5,
        )

    def generate_ray_samples(
        self,
        ray_bundle: Optional[RayBundle] = None,
        density_fn: Optional[Callable] = None,
        sdf_fn: Optional[Callable] = None,
        return_eikonal_points: bool = True,
    ) -> Union[Tuple[RaySamples, torch.Tensor], RaySamples]:
        assert ray_bundle is not None
        assert density_fn is not None
        assert sdf_fn is not None

        beta0 = density_fn.get_beta().detach()

        # Start with uniform sampling
        ray_samples = self.uniform_sampler(ray_bundle, num_samples=self.num_samples_eval)

        # Get maximum beta from the upper bound (Lemma 2)
        deltas = ray_samples.deltas.squeeze(-1)

        bound = (1.0 / (4.0 * torch.log(torch.tensor(self.eps + 1.0)))) * (deltas**2.0).sum(-1)
        beta = torch.sqrt(bound)

        total_iters, not_converge = 0, True
        sorted_index = None
        new_samples = ray_samples

        # Algorithm 1
        while not_converge and total_iters < self.max_total_iters:

            with torch.no_grad():
                new_sdf = sdf_fn(new_samples)

            # merge sdf predictions
            if sorted_index is not None:
                sdf_merge = torch.cat([sdf.squeeze(-1), new_sdf.squeeze(-1)], -1)
                sdf = torch.gather(sdf_merge, 1, sorted_index).unsqueeze(-1)
            else:
                sdf = new_sdf

            # Calculating the bound d* (Theorem 1)
            d_star = self.get_dstar(sdf, ray_samples)

            # Updating beta using line search
            beta = self.get_updated_beta(beta0, beta, density_fn, sdf, d_star, ray_samples)

            # Upsample more points
            density = density_fn(sdf.reshape(ray_samples.shape), beta=beta.unsqueeze(-1))

            weights, transmittance = ray_samples.get_weights_and_transmittance(density.unsqueeze(-1))

            #  Check if we are done and this is the last sampling
            total_iters += 1
            not_converge = beta.max() > beta0

            if not_converge and total_iters < self.max_total_iters:
                # Sample more points proportional to the current error bound
                deltas = ray_samples.deltas.squeeze(-1)

                error_per_section = (
                    torch.exp(-d_star / beta.unsqueeze(-1)) * (deltas**2.0) / (4 * beta.unsqueeze(-1) ** 2)
                )

                error_integral = torch.cumsum(error_per_section, dim=-1)
                weights = (torch.clamp(torch.exp(error_integral), max=1.0e6) - 1.0) * transmittance[..., 0]

                new_samples = self.pdf_sampler(
                    ray_bundle, ray_samples, weights.unsqueeze(-1), num_samples=self.num_samples_eval
                )

                ray_samples, sorted_index = self.merge_ray_samples(ray_bundle, ray_samples, new_samples)

            else:
                # Sample the final sample set to be used in the volume rendering integral
                ray_samples = self.pdf_sampler(ray_bundle, ray_samples, weights, num_samples=self.num_samples)

        if return_eikonal_points:
            # sample some of the near surface points for eikonal loss
            sampled_points = ray_samples.frustums.get_positions().view(-1, 3)
            idx = torch.randint(sampled_points.shape[0], (ray_samples.shape[0] * 10,)).to(sampled_points.device)
            points = sampled_points[idx]

        # Add extra samples uniformly
        if self.num_samples_extra > 0:
            ray_samples_uniform = self.uniform_sampler(ray_bundle, num_samples=self.num_samples_extra)
            ray_samples, _ = self.merge_ray_samples(ray_bundle, ray_samples, ray_samples_uniform)

        if return_eikonal_points:
            return ray_samples, points

        return ray_samples

    def get_dstar(self, sdf, ray_samples: RaySamples):
        """Calculating the bound d* (Theorem 1) from VolSDF"""
        d = sdf.reshape(ray_samples.shape)
        dists = ray_samples.deltas.squeeze(-1)
        a, b, c = dists[:, :-1], d[:, :-1].abs(), d[:, 1:].abs()
        first_cond = a.pow(2) + b.pow(2) <= c.pow(2)
        second_cond = a.pow(2) + c.pow(2) <= b.pow(2)
        d_star = torch.zeros(ray_samples.shape[0], ray_samples.shape[1] - 1).to(d.device)
        d_star[first_cond] = b[first_cond]
        d_star[second_cond] = c[second_cond]
        s = (a + b + c) / 2.0
        area_before_sqrt = s * (s - a) * (s - b) * (s - c)
        mask = ~first_cond & ~second_cond & (b + c - a > 0)
        d_star[mask] = (2.0 * torch.sqrt(area_before_sqrt[mask])) / (a[mask])
        d_star = (d[:, 1:].sign() * d[:, :-1].sign() == 1) * d_star  # Fixing the sign

        # padding to make the same shape as ray_samples
        # d_star_left = torch.cat((d_star[:, :1], d_star), dim=-1)
        # d_star_right = torch.cat((d_star, d_star[:, -1:]), dim=-1)
        # d_star = torch.minimum(d_star_left, d_star_right)

        d_star = torch.cat((d_star, d_star[:, -1:]), dim=-1)
        return d_star

    def get_updated_beta(self, beta0, beta, density_fn, sdf, d_star, ray_samples: RaySamples):
        curr_error = self.get_error_bound(beta0, density_fn, sdf, d_star, ray_samples)
        beta[curr_error <= self.eps] = beta0
        beta_min, beta_max = beta0.repeat(ray_samples.shape[0]), beta
        for j in range(self.beta_iters):
            beta_mid = (beta_min + beta_max) / 2.0
            curr_error = self.get_error_bound(beta_mid.unsqueeze(-1), density_fn, sdf, d_star, ray_samples)
            beta_max[curr_error <= self.eps] = beta_mid[curr_error <= self.eps]
            beta_min[curr_error > self.eps] = beta_mid[curr_error > self.eps]
        beta = beta_max
        return beta

    def get_error_bound(self, beta, density_fn, sdf, d_star, ray_samples):
        """Get error bound from VolSDF"""
        densities = density_fn(sdf.reshape(ray_samples.shape), beta=beta)

        deltas = ray_samples.deltas.squeeze(-1)
        delta_density = deltas * densities

        integral_estimation = torch.cumsum(delta_density[..., :-1], dim=-1)
        integral_estimation = torch.cat(
            [torch.zeros((*integral_estimation.shape[:1], 1), device=densities.device), integral_estimation], dim=-1
        )

        error_per_section = torch.exp(-d_star / beta) * (deltas**2.0) / (4 * beta**2)
        error_integral = torch.cumsum(error_per_section, dim=-1)
        bound_opacity = (torch.clamp(torch.exp(error_integral), max=1.0e6) - 1.0) * torch.exp(-integral_estimation)

        return bound_opacity.max(-1)[0]
    
    def merge_ray_samples(
        self,
        ray_bundle: RayBundle,
        *ray_samples_list: RaySamples,
    ):
        """Merge multiple sets of ray samples and return sorted index usable for merging SDF values.

        Args:
            ray_bundle: RayBundle containing ray information.
            *ray_samples_list: Variable number of RaySamples objects to merge.
        """
        if len(ray_samples_list) == 0:
            raise ValueError("At least one RaySamples object must be provided.")

        # Extract all start positions
        starts_list = [rs.spacing_starts[..., 0] for rs in ray_samples_list]
        # Concatenate all starts
        all_starts = torch.cat(starts_list, dim=-1)

        # Compute the maximum 'ends' among all ray samples
        all_ends = torch.stack(
            [rs.spacing_ends[..., -1:, 0] for rs in ray_samples_list],
            dim=0
        )
        ends = torch.max(all_ends, dim=0).values  # shape (..., 1)

        # Sort bins
        bins, sorted_index = torch.sort(all_starts, dim=-1)
        bins = torch.cat([bins, ends], dim=-1)

        # Stop gradients
        bins = bins.detach()

        # Use spacing_to_euclidean_fn from the first ray sample
        spacing_to_euclidean_fn = ray_samples_list[0].spacing_to_euclidean_fn
        euclidean_bins = spacing_to_euclidean_fn(bins)

        # Create new merged RaySamples
        ray_samples = ray_bundle.get_ray_samples(
            bin_starts=euclidean_bins[..., :-1, None],
            bin_ends=euclidean_bins[..., 1:, None],
            spacing_starts=bins[..., :-1, None],
            spacing_ends=bins[..., 1:, None],
            spacing_to_euclidean_fn=spacing_to_euclidean_fn,
        )

        return ray_samples, sorted_index