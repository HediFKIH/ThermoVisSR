import os
from imageio import imread
from PIL import Image,  ImageFilter
import numpy as np
import glob
import random
from skimage.color import rgb2lab, lab2rgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from skimage.filters import gaussian

import warnings
warnings.filterwarnings("ignore")


class RandomRotate(object):
    def __call__(self, sample):
        k1 = np.random.randint(0, 4)
        sample['LR_vis'] = np.rot90(sample['LR_vis'], k1).copy()
        sample['HR_vis'] = np.rot90(sample['HR_vis'], k1).copy()
        sample['LR_thr'] = np.rot90(sample['LR_thr'], k1).copy()
        sample['HR_thr'] = np.rot90(sample['HR_thr'], k1).copy()
        sample['LR_sr_vis'] = np.rot90(sample['LR_sr_vis'], k1).copy()
        sample['LR_sr_thr'] = np.rot90(sample['LR_sr_thr'], k1).copy()
        k2 = np.random.randint(0, 4)
        sample['Ref_vis'] = np.rot90(sample['Ref_vis'], k2).copy()
        sample['Ref_thr'] = np.rot90(sample['Ref_thr'], k2).copy()
        sample['Ref_sr_vis'] = np.rot90(sample['Ref_sr_vis'], k2).copy()
        sample['Ref_sr_thr'] = np.rot90(sample['Ref_sr_thr'], k2).copy()
        return sample


class RandomFlip(object):
    def __call__(self, sample):
        if (np.random.randint(0, 2) == 1):
            sample['LR_vis'] = np.fliplr(sample['LR_vis']).copy()
            sample['HR_vis'] = np.fliplr(sample['HR_vis']).copy()
            sample['LR_thr'] = np.fliplr(sample['LR_thr']).copy()
            sample['HR_thr'] = np.fliplr(sample['HR_thr']).copy()
            sample['LR_sr_vis'] = np.fliplr(sample['LR_sr_vis']).copy()
            sample['LR_sr_thr'] = np.fliplr(sample['LR_sr_thr']).copy()
        if (np.random.randint(0, 2) == 1):
            sample['Ref_vis'] = np.fliplr(sample['Ref_vis']).copy()
            sample['Ref_sr_vis'] = np.fliplr(sample['Ref_sr_vis']).copy()
            sample['Ref_thr'] = np.fliplr(sample['Ref_thr']).copy()
            sample['Ref_sr_thr'] = np.fliplr(sample['Ref_sr_thr']).copy()
        if (np.random.randint(0, 2) == 1):
            sample['LR_vis'] = np.flipud(sample['LR_vis']).copy()
            sample['HR_vis'] = np.flipud(sample['HR_vis']).copy()
            sample['LR_thr'] = np.flipud(sample['LR_thr']).copy()
            sample['HR_thr'] = np.flipud(sample['HR_thr']).copy()
            sample['LR_sr_vis'] = np.flipud(sample['LR_sr_vis']).copy()
            sample['LR_sr_thr'] = np.flipud(sample['LR_sr_thr']).copy()
        if (np.random.randint(0, 2) == 1):
            sample['Ref_vis'] = np.flipud(sample['Ref_vis']).copy()
            sample['Ref_sr_vis'] = np.flipud(sample['Ref_sr_vis']).copy()
            sample['Ref_thr'] = np.flipud(sample['Ref_thr']).copy()
            sample['Ref_sr_thr'] = np.flipud(sample['Ref_sr_thr']).copy()
        return sample


class ToTensor(object):
    def __call__(self, sample):
        LR_vis, LR_thr, LR_sr_vis, LR_sr_thr, HR_vis, HR_thr, Ref_vis, Ref_sr_vis, Ref_thr, Ref_sr_thr = sample['LR_vis'], sample['LR_thr'], sample['LR_sr_vis'], sample['LR_sr_thr'], sample['HR_vis'], sample['HR_thr'], sample['Ref_vis'], sample['Ref_sr_vis'], sample['Ref_thr'], sample['Ref_sr_thr']
        LR_vis = LR_vis.transpose((2,0,1))
        LR_thr = LR_thr.transpose((2,0,1))
        LR_sr_vis = LR_sr_vis.transpose((2,0,1))
        LR_sr_thr = LR_sr_thr.transpose((2,0,1))
        HR_vis = HR_vis.transpose((2,0,1))
        HR_thr = HR_thr.transpose((2,0,1))
        Ref_vis = Ref_vis.transpose((2,0,1))
        Ref_sr_vis = Ref_sr_vis.transpose((2,0,1))
        Ref_thr = Ref_thr.transpose((2,0,1))
        Ref_sr_thr = Ref_sr_thr.transpose((2,0,1))
        return {'LR_vis': torch.from_numpy(LR_vis).float(),
                'LR_thr': torch.from_numpy(LR_thr).float(),
                'LR_sr_vis': torch.from_numpy(LR_sr_vis).float(),
                'LR_sr_thr': torch.from_numpy(LR_sr_thr).float(),
                'HR_vis': torch.from_numpy(HR_vis).float(),
                'HR_thr': torch.from_numpy(HR_thr).float(),
                'Ref_vis': torch.from_numpy(Ref_vis).float(),
                'Ref_sr_vis': torch.from_numpy(Ref_sr_vis).float(),
                'Ref_thr': torch.from_numpy(Ref_thr).float(),
                'Ref_sr_thr': torch.from_numpy(Ref_sr_thr).float()}


class TrainSet(Dataset):
    def __init__(self, args, transform=transforms.Compose([RandomFlip(), RandomRotate(), ToTensor()]) ):
        self.input_vis_list = sorted([os.path.join(args.dataset_dir, 'train/input_vis', name) for name in 
            os.listdir( os.path.join(args.dataset_dir, 'train/input_vis') )])
        self.input_thr_list = sorted([os.path.join(args.dataset_dir, 'train/input_thr', name) for name in 
            os.listdir( os.path.join(args.dataset_dir, 'train/input_thr') )])
        self.ref_vis_list = sorted([os.path.join(args.dataset_dir, 'train/ref_vis', name) for name in 
            os.listdir( os.path.join(args.dataset_dir, 'train/ref_vis') )])        
        self.ref_thr_list = sorted([os.path.join(args.dataset_dir, 'train/ref_thr', name) for name in 
            os.listdir( os.path.join(args.dataset_dir, 'train/ref_thr') )])
        self.transform = transform

        print("="*60)
        print("DATASET LOADING VERIFICATION")
        print("="*60)
        print(f"Input Visible images: {len(self.input_vis_list)}")
        print(f"Input Thermal images: {len(self.input_thr_list)}")
        print(f"Ref Visible images: {len(self.ref_vis_list)}")
        print(f"Ref Thermal images: {len(self.ref_thr_list)}")
        print("="*60)
        
        # Vérifier que toutes les listes ont la même longueur
        lengths = [
            len(self.input_vis_list),
            len(self.input_thr_list),
            len(self.ref_vis_list),
            len(self.ref_thr_list)
        ]
        
        if len(set(lengths)) != 1:
            raise ValueError(
                f"   ERROR: Mismatch in dataset sizes!\n"
                f"   Input Visible: {len(self.input_vis_list)}\n"
                f"   Input Thermal: {len(self.input_thr_list)}\n"
                f"   Ref Visible: {len(self.ref_vis_list)}\n"
                f"   Ref Thermal: {len(self.ref_thr_list)}\n"
                f"   Please check your dataset folder structure!"
            )
        
        print(f"All lists have the same length: {len(self.input_vis_list)}")
        print("="*60)

    def __len__(self):
        return min(
            len(self.input_vis_list),
            len(self.input_thr_list),
            len(self.ref_vis_list),
            len(self.ref_thr_list)
        )
    
    def __getitem__(self, idx):
        if idx >= len(self.ref_thr_list):
            raise IndexError(
                f"Index {idx} out of range for ref_thr_list "
                f"(size: {len(self.ref_thr_list)}, requested idx: {idx})"
            )
        v_cut = 0.1 # c’est la valeur de coupure du signal   
        sigma = 4/np.pi*np.sqrt(np.log(v_cut**(-2)))
        ### HR
        HR_vis = imread(self.input_vis_list[idx])
        HR_thr = imread(self.input_thr_list[idx])
        HR_thr = np.repeat(HR_thr[..., np.newaxis], 3, -1)
        h,w = HR_vis.shape[:2]
        #HR = np.concatenate([HR_vis, HR_thr], axis=2)
        #HR = HR[:h//4*4, :w//4*4, :]

        ### LR and LR_sr
        #LR_vis = gaussian(HR_vis, sigma, output=None, mode='reflect',multichannel=True, preserve_range=True)
        #LR_vis = LR_vis.astype('uint8')
        LR_vis = np.array(Image.fromarray(HR_vis).resize((w//4, h//4), Image.NEAREST))
        #LR_thr = HR_thr.filter(ImageFilter.GaussianBlur)
        #LR_thr = gaussian(HR_thr, sigma, output=None, mode='reflect',multichannel=True, preserve_range=True)
        #LR_thr = LR_thr.astype('uint8')
        LR_thr = np.array(Image.fromarray(HR_thr).resize((w//4, h//4), Image.NEAREST))
        #LR = np.concatenate([LR_vis, LR_thr], axis=2)
        LR_sr_vis = np.array(Image.fromarray(LR_vis).resize((w, h), Image.NEAREST))
        LR_sr_thr = np.array(Image.fromarray(LR_thr).resize((w, h), Image.NEAREST))

        ### Ref and Ref_sr
        Ref_sub_vis = imread(self.ref_vis_list[idx])
        Ref_sub_thr = imread(self.ref_thr_list[idx])
        Ref_sub_thr = np.repeat(Ref_sub_thr[..., np.newaxis], 3, -1)
        h2, w2 = Ref_sub_vis.shape[:2]
        #Ref_sub = np.concatenate([Ref_sub_vis, Ref_sub_thr], axis=2)
        Ref_sr_sub_vis = np.array(Image.fromarray(Ref_sub_vis).resize((w2//4, h2//4), Image.NEAREST))
        Ref_sr_sub_thr = np.array(Image.fromarray(Ref_sub_thr).resize((w2//4, h2//4), Image.NEAREST))
        Ref_sr_sub_vis = np.array(Image.fromarray(Ref_sr_sub_vis).resize((w2, h2), Image.NEAREST))
        Ref_sr_sub_thr = np.array(Image.fromarray(Ref_sr_sub_thr).resize((w2, h2), Image.NEAREST))
    
        ### complete ref and ref_sr to the same size, to use batch_size > 1
        Ref_vis = np.zeros((360, 360, 3))
        Ref_thr = np.zeros((360, 360, 3))
        Ref_sr_vis = np.zeros((360, 360, 3))
        Ref_sr_thr = np.zeros((360, 360, 3))
        Ref_vis[:h2, :w2, :] = Ref_sub_vis
        Ref_thr[:h2, :w2, :] = Ref_sub_thr
        Ref_sr_vis[:h2, :w2, :] = Ref_sr_sub_vis
        Ref_sr_thr[:h2, :w2, :] = Ref_sr_sub_thr

        ### change type
        LR_vis = LR_vis.astype(np.float32)
        LR_thr = LR_thr.astype(np.float32)
        LR_sr_vis = LR_sr_vis.astype(np.float32)
        LR_sr_thr = LR_sr_thr.astype(np.float32)
        HR_vis = HR_vis.astype(np.float32)
        HR_thr = HR_thr.astype(np.float32)
        Ref_vis = Ref_vis.astype(np.float32)
        Ref_sr_vis = Ref_sr_vis.astype(np.float32)
        Ref_thr = Ref_thr.astype(np.float32)
        Ref_sr_thr = Ref_sr_thr.astype(np.float32)

        ### rgb range to [-1, 1]
        LR_vis = LR_vis / 127.5 - 1.
        LR_thr = LR_thr / 127.5 - 1.
        LR_sr_vis = LR_sr_vis / 127.5 - 1.
        LR_sr_thr = LR_sr_thr / 127.5 - 1.
        HR_vis = HR_vis / 127.5 - 1.
        HR_thr = HR_thr / 127.5 - 1.
        Ref_vis = Ref_vis / 127.5 - 1.
        Ref_sr_vis = Ref_sr_vis / 127.5 - 1.
        Ref_thr = Ref_thr / 127.5 - 1.
        Ref_sr_thr = Ref_sr_thr / 127.5 - 1.

        sample = {'LR_vis': LR_vis,
                  'LR_thr': LR_thr,
                  'LR_sr_vis': LR_sr_vis,
                  'LR_sr_thr': LR_sr_thr,
                  'HR_vis': HR_vis,
                  'HR_thr': HR_thr,
                  'Ref_vis': Ref_vis, 
                  'Ref_sr_vis': Ref_sr_vis,
                  'Ref_thr': Ref_thr, 
                  'Ref_sr_thr': Ref_sr_thr}

        if self.transform:
            sample = self.transform(sample)
        return sample


class TestSet(Dataset):
    def __init__(self, args, ref_level='1', transform=transforms.Compose([ToTensor()])):
        self.input_vis_list = sorted(glob.glob(os.path.join(args.dataset_dir, 'test/input_vis', '*_0.jpg')))
        self.ref_vis_list = sorted(glob.glob(os.path.join(args.dataset_dir, 'test/input_vis', 
            '*_' + ref_level + '.jpg')))
        self.input_thr_list = sorted(glob.glob(os.path.join(args.dataset_dir, 'test/input_thr', '*_0.jpg')))
        self.ref_thr_list = sorted(glob.glob(os.path.join(args.dataset_dir, 'test/input_thr', 
            '*_' + ref_level + '.jpg')))
        self.transform = transform

    def __len__(self):
        return len(self.input_vis_list)
    
    def __getitem__(self, idx):
        v_cut = 0.1 # c’est la valeur de coupure du signal   
        sigma = 4/np.pi*np.sqrt(np.log(v_cut**(-2)))
        ### HR
        HR_vis = imread(self.input_vis_list[idx])
        HR_thr = imread(self.input_thr_list[idx])
        HR_thr = np.repeat(HR_thr[..., np.newaxis], 3, -1)
        #HR = np.concatenate([HR_vis, HR_thr], axis=2)
        h, w = HR_vis.shape[:2]
        h, w = h//4*4, w//4*4
        HR_vis = HR_vis[:h, :w, :] ### crop to the multiple of 4
        HR_thr = HR_thr[:h, :w, :]

        ### LR and LR_sr
        #LR_vis = HR_vis.filter(ImageFilter.GaussianBlur)
        #LR_vis = gaussian(HR_vis, sigma, output=None, mode='reflect',multichannel=True, preserve_range=True)
        #LR_vis = LR_vis.astype('uint8')
        LR_vis = np.array(Image.fromarray(HR_vis).resize((w//4, h//4), Image.NEAREST))
        #LR_thr = HR_thr.filter(ImageFilter.GaussianBlur)
        #LR_thr = gaussian(HR_thr, sigma, output=None, mode='reflect',multichannel=True, preserve_range=True)
        #LR_thr = LR_thr.astype('uint8')
        LR_thr = np.array(Image.fromarray(HR_thr).resize((w//4, h//4), Image.NEAREST))
        #LR = np.concatenate([LR_vis, LR_thr], axis=2)
        LR_sr_vis = np.array(Image.fromarray(LR_vis).resize((w, h), Image.NEAREST))
        LR_sr_thr = np.array(Image.fromarray(LR_thr).resize((w, h), Image.NEAREST))

        ### Ref and Ref_sr
        Ref_vis = imread(self.ref_vis_list[idx])
        Ref_thr = imread(self.ref_thr_list[idx])
        Ref_thr = np.repeat(Ref_thr[..., np.newaxis], 3, -1)
        h2, w2 = Ref_vis.shape[:2]
        h2, w2 = h2//4*4, w2//4*4
        Ref_vis = Ref_vis[:h2, :w2, :]
        Ref_thr = Ref_thr[:h2, :w2, :]
        Ref_sr_vis = np.array(Image.fromarray(Ref_vis).resize((w2//4, h2//4), Image.NEAREST))
        Ref_sr_vis = np.array(Image.fromarray(Ref_sr_vis).resize((w2, h2), Image.NEAREST))
        Ref_sr_thr = np.array(Image.fromarray(Ref_thr).resize((w2//4, h2//4), Image.NEAREST))
        Ref_sr_thr = np.array(Image.fromarray(Ref_sr_thr).resize((w2, h2), Image.NEAREST))

        ### change type
        LR_vis = LR_vis.astype(np.float32)
        LR_thr = LR_thr.astype(np.float32)
        LR_sr_vis = LR_sr_vis.astype(np.float32)
        LR_sr_thr = LR_sr_thr.astype(np.float32)
        HR_vis = HR_vis.astype(np.float32)
        HR_thr = HR_thr.astype(np.float32)
        Ref_vis = Ref_vis.astype(np.float32)
        Ref_sr_vis = Ref_sr_vis.astype(np.float32)
        Ref_thr = Ref_thr.astype(np.float32)
        Ref_sr_thr = Ref_sr_thr.astype(np.float32)

        ### rgb range to [-1, 1]
        LR_vis = LR_vis / 127.5 - 1.
        LR_thr = LR_thr / 127.5 - 1.
        LR_sr_vis = LR_sr_vis / 127.5 - 1.
        LR_sr_thr = LR_sr_thr / 127.5 - 1.
        HR_vis = HR_vis / 127.5 - 1.
        HR_thr = HR_thr / 127.5 - 1.
        Ref_vis = Ref_vis / 127.5 - 1.
        Ref_sr_vis = Ref_sr_vis / 127.5 - 1.
        Ref_thr = Ref_thr / 127.5 - 1.
        Ref_sr_thr = Ref_sr_thr / 127.5 - 1.

        sample = {'LR_vis': LR_vis,
                  'LR_thr': LR_thr,
                  'LR_sr_vis': LR_sr_vis,
                  'LR_sr_thr': LR_sr_thr,
                  'HR_vis': HR_vis,
                  'HR_thr': HR_thr,
                  'Ref_vis': Ref_vis, 
                  'Ref_sr_vis': Ref_sr_vis,
                  'Ref_thr': Ref_thr, 
                  'Ref_sr_thr': Ref_sr_thr}

        if self.transform:
            sample = self.transform(sample)
        return sample

