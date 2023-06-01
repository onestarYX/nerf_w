import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image, ImageDraw
from torchvision import transforms as T
from pathlib import Path

from .ray_utils import *

ORIGINAL_W = 2048
ORIGINAL_H = 1536

def add_perturbation(img, perturbation, seed):
    if 'color' in perturbation:
        np.random.seed(seed)
        img_np = np.array(img)/255.0
        s = np.random.uniform(0.8, 1.2, size=3)
        b = np.random.uniform(-0.2, 0.2, size=3)
        img_np[..., :3] = np.clip(s*img_np[..., :3]+b, 0, 1)
        img = Image.fromarray((255*img_np).astype(np.uint8))
    if 'occ' in perturbation:
        draw = ImageDraw.Draw(img)
        np.random.seed(seed)
        left = np.random.randint(200, 400)
        top = np.random.randint(200, 400)
        for i in range(10):
            np.random.seed(10*seed+i)
            random_color = tuple(np.random.choice(range(256), 3))
            draw.rectangle(((left+20*i, top), (left+20*(i+1), top+200)),
                            fill=random_color)
    return img


class BehaveDataset(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(2048, 1536),
                 perturbation=[]):
        self.root_dir = root_dir
        self.split = split
        self.img_wh = img_wh
        self.define_transforms()

        # self.perturbation = perturbation
        # if self.split == 'train':
        #     print(f'add {self.perturbation} perturbation!')
        self.read_meta()
        self.white_back = True

    def read_meta(self):
        # Process paths
        self.seq_dir = Path(self.root_dir)
        self.root_dir = self.root_dir.split('/')
        scene_name, seq_name = self.root_dir[-2:]
        self.root_dir = Path('/'.join(self.root_dir[:-2]))
        intrinsics_dir = self.root_dir / 'calibs' / 'intrinsics'
        extrinsics_dir = self.root_dir / 'calibs' / scene_name / 'config'
        self.cam_intrinsics_meta = []
        self.cam_extrinsics_meta = []
        for i in range(4):
            with open(intrinsics_dir / f"{i}" / 'calibration.json', 'r') as f:
                self.cam_intrinsics_meta.append(json.load(f))
            with open(extrinsics_dir / f"{i}" / 'config.json', 'r') as f:
                self.cam_extrinsics_meta.append(json.load(f))

        w, h = self.img_wh
        w_ratio = w / ORIGINAL_W
        h_ratio = h / ORIGINAL_H
        self.focal = []
        self.K = []
        for i in range(4):
            fx = self.cam_intrinsics_meta[i]['color']['fx'] * w_ratio
            fy = self.cam_intrinsics_meta[i]['color']['fy'] * h_ratio
            cx = self.cam_intrinsics_meta[i]['color']['cx'] * w_ratio
            cy = self.cam_intrinsics_meta[i]['color']['cy'] * h_ratio
            self.focal.append((fx, fy))
            K = np.eye(3)
            K[0, 0] = fx
            K[1, 1] = fy
            K[0, 2] = cx
            K[1, 2] = cy
            self.K.append(K)

        # bounds, common for all scenes
        self.near = 0.1
        self.far = 20
        self.bounds = np.array([self.near, self.far])
        
        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = []
        for i in range(4):
            self.directions.append(get_ray_directions(h, w, self.K[i])) # (h, w, 3)
            
        # create buffer of all rays and rgb data
        self.all_rays = []
        self.all_rgbs = []
        # Get list of all frame directories.
        def sort_frame(path):
            return int(path.name[1:5])
        frame_dir_list = []
        for frame_dir in self.seq_dir.iterdir():
            if frame_dir.name == 'info.json':
                continue
            frame_dir_list.append(frame_dir)
        # frame_dir_list.sort(key=sort_frame)

        # Do train/validation split
        num_frames_train = int(len(frame_dir_list) * 0.8)
        if self.split == 'train':
            t = 0
            frame_dir_list = frame_dir_list[:num_frames_train]
        elif self.split == 'val':
            t = num_frames_train * 4    # TODO: this should be modified if not all frame dir has 4 images.
            frame_dir_list = frame_dir_list[num_frames_train:]

        for frame_dir in frame_dir_list:
            for file in frame_dir.iterdir():
                if 'color.jpg' not in file.name:
                    continue
                image_path = str(file)
                cam_idx = int(file.name.split('.')[0][1])
                c2w = np.zeros((3, 4))
                rotation = np.array(self.cam_extrinsics_meta[cam_idx]['rotation']).reshape((3, 3))
                translation = np.array(self.cam_extrinsics_meta[cam_idx]['translation'])
                c2w[:3, :3] = rotation
                c2w[:3, 3] = translation
                c2w = torch.tensor(c2w, dtype=torch.float32)

                with Image.open(image_path) as img:
                    img = self.transform(img)  # (4, h, w)
                    img = img.view(3, -1).permute(1, 0)  # (h*w, 4) RGBA
                    # img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
                    self.all_rgbs += [img]

                    rays_o, rays_d = get_rays(self.directions[cam_idx], c2w)  # both (h*w, 3)
                    rays_t = t * torch.ones(len(rays_o), 1)

                    self.all_rays += [torch.cat([rays_o, rays_d,
                                                 self.near * torch.ones_like(rays_o[:, :1]),
                                                 self.far * torch.ones_like(rays_o[:, :1]),
                                                 rays_t],
                                                1)]  # (h*w, 8)
                t += 1

        if self.split == 'train':
            self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w, 3)

    def define_transforms(self):
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize(size=self.img_wh)
        ])


    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        elif self.split == 'val':
            return len(self.all_rays)

    def __getitem__(self, idx):
        if self.split == 'train':
            sample = {'rays': self.all_rays[idx, :8],
                      'ts': self.all_rays[idx, 8].long(),
                      'rgbs': self.all_rgbs[idx]}
        elif self.split == 'val':
            sample = {
                'rays': self.all_rays[idx][:, :8],
                'ts': self.all_rays[idx][:, 8].long(),
                'rgbs': self.all_rgbs[idx]
            }

        return sample