import os
import glob
import hydra
import cv2
import numpy as np
import torch
from lib.utils import utils
import torch.nn.functional as F

class Dataset(torch.utils.data.Dataset):
    def __init__(self, metainfo, split):
        root = os.path.join("../data", metainfo.data_dir)
        root = hydra.utils.to_absolute_path(root)

        self.num_clothes = metainfo.num_clothes
        self.start_frame = metainfo.start_frame
        self.end_frame = metainfo.end_frame
        self.skip_step = 1
        self.images, self.img_sizes = [], []
        self.training_indices = list(
            range(metainfo.start_frame, metainfo.end_frame, self.skip_step)
        )

        self.max_freq = 4

        # images
        img_dir = os.path.join(root, "image")
        self.img_paths = sorted(glob.glob(f"{img_dir}/*.png"))

        # only store the image paths to avoid OOM
        self.img_paths = [self.img_paths[i] for i in self.training_indices]
        self.img_size = cv2.imread(self.img_paths[0]).shape[:2]
        self.n_images = len(self.img_paths)

        # coarse projected SMPL masks, only for sampling
        mask_dir = os.path.join(root, "mask")
        self.mask_paths = sorted(glob.glob(f"{mask_dir}/*.png"))
        self.mask_paths = [self.mask_paths[i] for i in self.training_indices]

        if self.num_clothes == 2:
            mask_cloth1_dir = os.path.join(root, "mask_cloth1")
            self.mask_cloth1_paths = sorted(glob.glob(f"{mask_cloth1_dir}/*.png"))
            mask_cloth2_dir = os.path.join(root, "mask_cloth2")
            self.mask_cloth2_paths = sorted(glob.glob(f"{mask_cloth2_dir}/*.png"))
        elif self.num_clothes == 1:
            mask_cloth_dir = os.path.join(root, "mask_cloth1")
            self.mask_cloth_paths = sorted(glob.glob(f"{mask_cloth_dir}/*.png"))
            self.mask_cloth_paths = [self.mask_cloth_paths[i] for i in self.training_indices]

        self.shape = np.load(os.path.join(root, "mean_shape.npy"))
        self.poses = np.load(os.path.join(root, "poses.npy"))[self.training_indices]
        self.trans = np.load(os.path.join(root, "normalize_trans.npy"))[
            self.training_indices
        ]
        # cameras
        camera_dict = np.load(os.path.join(root, "cameras_normalize.npz"))
        scale_mats = [
            camera_dict["scale_mat_%d" % idx].astype(np.float32)
            for idx in self.training_indices
        ]
        world_mats = [
            camera_dict["world_mat_%d" % idx].astype(np.float32)
            for idx in self.training_indices
        ]

        self.scale = 1 / scale_mats[0][0, 0]

        self.intrinsics_all = []
        self.pose_all = []
        self.P = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            self.P.append(P)
            P = P[:3, :4]
            intrinsics, pose = utils.load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())
        assert len(self.intrinsics_all) == len(self.pose_all)

        # other properties
        self.num_sample = split.num_sample
        self.sampling_strategy = "weighted"

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        # normalize RGB
        img = cv2.imread(self.img_paths[idx])
        # preprocess: BGR -> RGB -> Normalize

        img = img[:, :, ::-1] / 255

        # preprocess: BGR -> Gray -> Mask
        mask = cv2.imread(self.mask_paths[idx])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) > 0
        if self.num_clothes == 2:
            mask_cloth1 = cv2.imread(self.mask_cloth1_paths[idx])
            mask_cloth1 = cv2.cvtColor(mask_cloth1, cv2.COLOR_BGR2GRAY) > 0
            mask_cloth2 = cv2.imread(self.mask_cloth2_paths[idx])
            mask_cloth2 = cv2.cvtColor(mask_cloth2, cv2.COLOR_BGR2GRAY) > 0
        elif self.num_clothes == 1:
            mask_cloth = cv2.imread(self.mask_cloth_paths[idx])
            mask_cloth = cv2.cvtColor(mask_cloth, cv2.COLOR_BGR2GRAY) > 0

        img_size = self.img_size

        uv = np.mgrid[: img_size[0], : img_size[1]].astype(np.int32)
        uv = np.flip(uv, axis=0).copy().transpose(1, 2, 0).astype(np.float32)

        smpl_params = torch.zeros([86]).float()
        smpl_params[0] = torch.from_numpy(np.asarray(self.scale)).float()

        smpl_params[1:4] = torch.from_numpy(self.trans[idx]).float()
        smpl_params[4:76] = torch.from_numpy(self.poses[idx]).float()
        smpl_params[76:] = torch.from_numpy(self.shape).float()

        t = idx / self.__len__()
        time_enc = utils.time_encoding(t, smpl_params[0].dtype, self.max_freq)

        if self.num_sample > 0:
            data = {
                "rgb": img,
                "uv": uv,
                "object_mask": mask,
            }
            if self.num_clothes == 2:
                data.update({"cloth_mask1": mask_cloth1, "cloth_mask2": mask_cloth2})
            elif self.num_clothes == 1:
                data.update({"cloth_mask": mask_cloth})

            samples, index_inside, index_outside = utils.edge_sampling(
                data, self.num_sample
            )

            inputs = {
                "uv": samples["uv"].astype(np.float32),
                "intrinsics": self.intrinsics_all[idx],
                "pose": self.pose_all[idx],
                "P": self.P[idx],
                "smpl_params": smpl_params,
                "index_outside": index_outside,
                "index_inside": index_inside,
                "time_enc": time_enc,
                "idx": idx,
            }
            images = {
                "rgb": samples["rgb"].astype(np.float32),
                "mask": samples["object_mask"].astype(np.float32),
            }
            if self.num_clothes == 2:
                images.update(
                    {
                        "mask_cloth1": samples["cloth_mask1"].astype(np.float32),
                        "mask_cloth2": samples["cloth_mask2"].astype(np.float32),
                    }
                )
            elif self.num_clothes == 1:
                images.update(
                    {"mask_cloth": samples["cloth_mask"].astype(np.float32)}
                )
            return inputs, images
        else:
            inputs = {
                "uv": uv.reshape(-1, 2).astype(np.float32),
                "intrinsics": self.intrinsics_all[idx],
                "pose": self.pose_all[idx],
                "P": self.P[idx],
                "smpl_params": smpl_params,
                "time_enc": time_enc,
                "idx": idx,
            }
            images = {
                "rgb": img.reshape(-1, 3).astype(np.float32),
                # "feat": feat.reshape(-1, 256).astype(np.float32),
                "img_size": self.img_size,
            }
            return inputs, images


class ValDataset(torch.utils.data.Dataset):
    def __init__(self, metainfo, split):
        self.dataset = Dataset(metainfo, split)
        self.img_size = self.dataset.img_size

        self.total_pixels = np.prod(self.img_size)
        self.pixel_per_batch = split.pixel_per_batch

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        image_id = int(np.random.choice(len(self.dataset), 1))
        self.data = self.dataset[image_id]
        inputs, images = self.data

        inputs = {
            "uv": inputs["uv"],
            "intrinsics": inputs["intrinsics"],
            "pose": inputs["pose"],
            "P": inputs["P"],
            "smpl_params": inputs["smpl_params"],
            "image_id": image_id,
            "time_enc": inputs["time_enc"],
            "idx": inputs["idx"],
        }
        images = {
            "rgb": images["rgb"],
            "img_size": images["img_size"],
            "pixel_per_batch": self.pixel_per_batch,
            "total_pixels": self.total_pixels,
        }
        return inputs, images

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, metainfo, split):
        self.dataset = Dataset(metainfo, split)

        self.img_size = self.dataset.img_size

        self.total_pixels = np.prod(self.img_size)
        self.pixel_per_batch = split.pixel_per_batch

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]

        inputs, images = data
        inputs = {
            "uv": inputs["uv"],
            "intrinsics": inputs["intrinsics"],
            "pose": inputs["pose"],
            "P": inputs["P"],
            "smpl_params": inputs["smpl_params"],
            "time_enc": inputs["time_enc"],
            "idx": inputs["idx"],
        }
        images = {"rgb": images["rgb"], "img_size": images["img_size"]}
        return inputs, images, self.pixel_per_batch, self.total_pixels, idx
