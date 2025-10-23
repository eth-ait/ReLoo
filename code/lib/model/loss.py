import torch
from torch import nn
from torch.nn import functional as F
from collections import defaultdict
def gmof(x, sigma=100):
    """
    Geman-McClure error function
    """
    x_squared = x ** 2
    sigma_squared = sigma ** 2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)

class Loss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.eikonal_weight = opt.eikonal_weight
        self.bce_weight = opt.bce_weight
        self.mask_weight = opt.mask_weight
        self.seg_weight = opt.seg_weight

        self.eps = opt.eps
        self.milestone = opt.milestone

        self.l1_loss = nn.L1Loss(reduction='mean')
        self.l2_loss = nn.MSELoss(reduction='mean')
    
    # L1 reconstruction loss for RGB
    def get_rgb_loss(self, rgb_values, rgb_gt):
        rgb_loss = self.l1_loss(rgb_values, rgb_gt)
        return rgb_loss
    
    # Eikonal loss
    def get_eikonal_loss(self, grad_theta):
        eikonal_loss = ((grad_theta.norm(2, dim=-1) - 1)**2).mean()
        return eikonal_loss

    # BCE loss for clear boundary
    def get_bce_loss(self, acc_map):
        binary_loss = -1 * (acc_map * (acc_map + self.eps).log() + (1-acc_map) * (1 - acc_map + self.eps).log()).mean() * 2
        return binary_loss

    # L1 loss for mask supervision
    def get_mask_loss(self, acc_map, index_inside, index_outside):
        mask_loss = self.l1_loss(acc_map[index_inside], torch.ones_like(acc_map[index_inside])) + \
                    self.l1_loss(acc_map[index_outside], torch.zeros_like(acc_map[index_outside]))
        return mask_loss

    # Geman-McClure loss for segmentation supervision
    def get_seg_loss(self, model_outputs, ground_truth):
        if 'mask_cloth' in ground_truth:
            cloth_mask_loss = gmof(model_outputs['acc_map_cloth'] - ground_truth['mask_cloth'][0]).mean()
        if 'mask_cloth1' in ground_truth and 'mask_cloth2' in ground_truth:
            cloth_mask_loss =  gmof(model_outputs['acc_map_cloth1'] - ground_truth['mask_cloth1'][0]).mean()
            cloth_mask_loss += gmof(model_outputs['acc_map_cloth2'] - ground_truth['mask_cloth2'][0]).mean()
        return cloth_mask_loss

    # L2 loss for virtual bone deformation field regularization
    def get_df_reg_loss(self, model_outputs):
        return self.l2_loss(model_outputs['v_cloth_d_smpl'], model_outputs['v_cloth_d_vb'])

    def forward(self, model_outputs, ground_truth):
        output = defaultdict(float)

        nan_filter = ~torch.any(model_outputs['rgb_values'].isnan(), dim=1)
        curr_epoch_for_loss = min(self.milestone, model_outputs['epoch']) # constant after the milestone
        curr_epoch_for_reg_loss = min(200, model_outputs['epoch'])

        rgb_gt = ground_truth['rgb'][0].cuda()
        rgb_loss = self.get_rgb_loss(model_outputs['rgb_values'][nan_filter], rgb_gt[nan_filter])
        eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])
        bce_loss = self.get_bce_loss(model_outputs['acc_map'])
        mask_loss = self.get_mask_loss(model_outputs['acc_map'], model_outputs['index_inside'], model_outputs['index_outside'])
        seg_loss = self.get_seg_loss(model_outputs, ground_truth)
        if model_outputs['v_cloth_d_smpl'] is not None:
            dg_reg_loss = self.get_df_reg_loss(model_outputs)
        else:
            dg_reg_loss = torch.tensor(0.).cuda()

        loss = rgb_loss + \
               self.eikonal_weight * eikonal_loss + \
               self.bce_weight * bce_loss + \
               self.mask_weight * mask_loss + \
               self.seg_weight * seg_loss + \
               (1 - curr_epoch_for_loss / (self.milestone+1)) * dg_reg_loss

        # SMPL body shape regularization
        body_reg_loss = self.l1_loss(model_outputs["smpl_surface_sdf"].squeeze(0).squeeze(-1), model_outputs["smpl_surface_sdf_ps_gt"]) # (sdf[mask] - 0.01).mean()
        output["body_reg_loss"] += body_reg_loss
        if model_outputs['epoch'] >= 100:
            loss += (10.1 - curr_epoch_for_reg_loss * 0.05) * body_reg_loss
        else:
            loss += 5 * body_reg_loss

        output.update({
            'rgb_loss': rgb_loss,
            'eikonal_loss': eikonal_loss,
            'bce_loss': bce_loss,
            'mask_loss': mask_loss,
            'seg_loss': seg_loss,
            'dg_reg_loss': dg_reg_loss,
            'loss': loss
        })
        return output