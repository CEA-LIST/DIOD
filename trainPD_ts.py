from __future__ import annotations

from collections import OrderedDict
import os
import argparse
from tkinter import NONE
from tqdm import tqdm
import time
import datetime
import torch.optim as optim
import torch.nn as nn
import random
import matplotlib.pyplot as plt
import scipy.optimize
import torch.nn.functional as F
import numpy as np 
import torch

from datasetPD import PDDataset
from datasetPDEval import PDDataset as PDDatasetEval

from  models.model_bg import SlotAttentionAutoEncoder
from transforms import get_normalize_transform, get_ssl_train_online_transform, get_crop_transform
from utils import adjusted_rand_index as ARI 
from utils import temporal_loss as t_loss 

from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import InterpolationMode

import math
from scipy.ndimage import label

# Set the random seed for consistant comparisons/ablation studies and reproducibility  
# No tuning was performed over multiple seeds
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# Set CUDA deterministic flags
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument
# basic configurations
parser.add_argument('--model_dir', default='/home/data/skara/tmp/', type=str, help='where to save models' )
parser.add_argument('--sample_dir', default = './samples/', type = str, help = 'where to save the plots')
# parser.add_argument('--exp_name', default='check_release_train_DIODPD_500_se42', type=str, help='experiment name, used for model saving/plotting ect' )
parser.add_argument('--exp_name', default='check_release_train_DIODPD_500_DIODENV', type=str, help='experiment name, used for model saving/plotting ect' )

parser.add_argument('--num_workers', default=12, type=int, help='number of workers for loading data')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--data_path', default = '/home/data/skara/TRI_PD/TRI_PD_train', type = str, help = 'path of PD dataset')
parser.add_argument('--supervision',  default = 'est', choices=['moving', 'all','est'], help = 'type of supervision, currently available: moving and all')
# model parameters
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--num_slots', default=45, type=int, help='Number of slots in Slot Attention.')
parser.add_argument('--hid_dim', default=64, type=int, help='hidden dimension size')
parser.add_argument('--learning_rate', default=0.0005, type=float)
parser.add_argument('--warmup_steps', default=3000, type=int, help='Number of warmup steps for the learning rate.')
parser.add_argument('--decay_rate', default=0.5, type=float, help='Rate for the learning rate decay.')
parser.add_argument('--decay_steps', default=50000, type=int, help='Number of steps for the learning rate decay.')
parser.add_argument('--num_epochs', default=500, type=int, help='number of workers for loading data')
parser.add_argument('--weight_mask', default = 1.0, type = float, help = 'weight for the mask loss')
parser.add_argument('--weight_NLL', default = 1.0, type = float, help = 'weight for the NLL loss')
parser.add_argument('--weight_reg', default = 0.3, type = float, help = 'weight for the regularization term')
parser.add_argument('--confid_th', default = 0.9, type = float, help = 'confidence threshold used to filter pseudo-labels during distillation')
# Data augmentations
parser.add_argument('--crop-size', default=[], type = int, nargs="*", help = "crop's size")
parser.add_argument('--crop-scale', default=[0.75, 1.0], type = int, nargs="+", help = "crop's size")
parser.add_argument('--crop-ratio', default=[2.0, 2.0], type = int, nargs="+", help = "crop's ratio")
parser.add_argument('--crop-interpolation', default="bilinear", type = str, help = "crop's interpolation")
parser.add_argument('--crop-interpolation-mask', default="nearest", type = str, help = "crop's interpolation mask")
parser.add_argument('--p-crop', default=0.4, type = float, help = "crop's probability")
parser.add_argument('--p-flip', default=0.4, type = float, help = "flip's probability")
parser.add_argument('--brightness', default=0.4, type = float, help = "color jitter's brightness")
parser.add_argument('--contrast', default=0.4, type = float, help = "color jitter's contrast")
parser.add_argument('--saturation', default=0.4, type = float, help = "color jitter's saturation")
parser.add_argument('--hue', default=0.1, type = float, help = "color jitter's hue")
parser.add_argument('--p-contrastive', default=0.8, type = float, help = "color jitter's probability")
parser.add_argument('--p-grayscale', default=0.2, type = float, help = "grayscale's probability")
parser.add_argument('--kernel-size-blur', default=11, type = int, help = "blur's kernel size")
parser.add_argument('--sigma-blur', default=[0.1, 2.0], type = float, nargs="+", help = "blur's sigma")
parser.add_argument('--p-blur', default=0.0, type = float, help = "blur's probability")
parser.add_argument('--threshold-solarize', default=0.5, type = float, help = "solarize's threshold")
parser.add_argument('--p-solarize', default=0.2, type = float, help = "solarize's probability")
parser.add_argument('--mean', default=[0.5, 0.5, 0.5], type = float, nargs="+", help = "normalize's mean")
parser.add_argument('--std', default=[0.5, 0.5, 0.5], type = float, nargs="+", help = "normalize's std")
# Burn_in
parser.add_argument('--burn_in_ckpt', default='DIODPD_burn_in_400.ckpt', type=str, help='ckpt from burn_in to load when starting distillation')
parser.add_argument('--burn_in_exp', default='checkpoints', type=str, help='burn_in experiment name' )

# burn_in or distillation?
parser.add_argument('--start_teacher', default=0, type = int, help = "teacher starting epoch")

def computeWBCE(opt, masks, mask_gt, scores):
    scores = scores[:,:,:-1,:] # to discard last slot (reserved for Bg)
    masks = masks * 0.999 + 1e-8
    loss_mask = 0
    for i in range(opt.batch_size):
        for j in range(5):
            matches = scipy.optimize.linear_sum_assignment(-scores[i,j])
            id_slot, id_gt = matches 
            tmp_BCE, nb_objects = 0, 0
            for idx, s in enumerate(id_slot):
                mask_2_weight = mask_gt[i,j,id_gt[idx],:,:]
                if not mask_2_weight.max(): continue # corresponding object non-available
                fg_portion = torch.where(mask_gt[i,j,id_gt[idx],:,:]!=0)[0].size()[0]/(120*242)
                
                tmp_BCE += ( -(2-fg_portion)*torch.log(masks[i,j,s,:,:]) * mask_gt[i,j,id_gt[idx],:,:] - (1- mask_gt[i,j,id_gt[idx],:,:]) * torch.log(1-masks[i,j,s,:,:])).mean()
                
                nb_objects += 1
            if nb_objects == 0:
                continue
            tmp_BCE /= nb_objects
            loss_mask += tmp_BCE

    loss_mask /= opt.batch_size
    return loss_mask


def computeWBCE_TS(opt, masks, mask_gt, scores, mask_fg_t=None, device = None, teacher = 0 ):
    scores = scores[:,:,:-1,:] # to discard last slot (reserved for Bg)
    masks = masks * 0.999 + 1e-8
    loss_mask = 0
    fg_gt_after_filtering = torch.zeros(mask_fg_t[:,:,0,:,:].shape).to(device) 
    for i in range(opt.batch_size):
        for j in range(5):
            matches = scipy.optimize.linear_sum_assignment(-scores[i,j])
            id_slot, id_gt = matches 
            tmp_BCE, nb_objects = 0, 0
            for idx, s in enumerate(id_slot):
                mask_2_weight = mask_gt[i,j,id_gt[idx],:,:]
                if not mask_2_weight.max(): 
                    continue # corresponding object non-available

                # using fg_teacher as a confidence map on pseudo_labels
                weight = mask_fg_t[i,j,0,:,:][mask_gt[i,j,id_gt[idx],:,:] == 1].mean()
                if not teacher:  
                    if weight>=opt.confid_th:
                        nb_objects += 1
                        fg_gt_after_filtering[i,j,:,:] += mask_gt[i,j,id_gt[idx],:,:]
                        tmp_BCE += ( -(1+weight)*torch.log(masks[i,j,s,:,:]) * mask_gt[i,j,id_gt[idx],:,:] - (1- mask_gt[i,j,id_gt[idx],:,:]) * torch.log(1-masks[i,j,s,:,:])).mean()
                else:
                    if weight>=opt.confid_th:
                        nb_objects += 1
                        tmp_BCE += ( -(1+weight)*torch.log(masks[i,j,s,:,:]) * mask_gt[i,j,id_gt[idx],:,:] - (1- mask_gt[i,j,id_gt[idx],:,:]) * torch.log(1-masks[i,j,s,:,:])).mean()

            if nb_objects == 0:
                continue
            tmp_BCE /= nb_objects
            loss_mask += tmp_BCE

    loss_mask /= opt.batch_size

    if not teacher:
        return loss_mask, fg_gt_after_filtering
    return loss_mask


@torch.no_grad()
def _update_teacher_model(model, teacher_model, keep_rate=0.996):
    
    student_model_dict = model.state_dict()

    new_teacher_dict = OrderedDict()
    for key, value in teacher_model.state_dict().items():
        if key in student_model_dict.keys():
            new_teacher_dict[key] = (
                student_model_dict[key] *
                (1 - keep_rate) + value * keep_rate
            )
        else:
            raise Exception("{} is not found in student model".format(key))

    teacher_model.load_state_dict(new_teacher_dict)
    

def main():
    opt = parser.parse_args()

    resolution = (480, 968)

    
    if not os.path.exists(opt.model_dir):
        os.mkdir(opt.model_dir)
    if not os.path.exists(opt.sample_dir):
        os.mkdir(opt.sample_dir)
    if not os.path.exists(os.path.join(opt.model_dir, opt.exp_name)):
        os.mkdir(os.path.join(opt.model_dir, opt.exp_name))
    if not os.path.exists(os.path.join(opt.sample_dir, opt.exp_name)):
        os.mkdir(os.path.join(opt.sample_dir, opt.exp_name))

    data_path = opt.data_path
    train_transform = get_crop_transform(
        opt.crop_size if opt.crop_size != [] else None, 
        opt.crop_scale, 
        opt.crop_ratio, 
        InterpolationMode(opt.crop_interpolation), 
        InterpolationMode(opt.crop_interpolation_mask), 
        opt.p_crop, 
        opt.p_flip
    )
    print("train transform", train_transform)
    train_set = PDDataset(split = 'train', root = data_path, supervision = opt.supervision, transform=train_transform, apply_img_transform=True)
    test_set = PDDatasetEval(split = 'eval', root = data_path)


    model = SlotAttentionAutoEncoder(resolution, opt.num_slots, opt.hid_dim, output_channel = 3).to(device)
    model = nn.DataParallel(model)
    student_train_transform = get_ssl_train_online_transform(
        opt.brightness,
        opt.contrast,
        opt.saturation,
        opt.hue,
        opt.p_contrastive,
        opt.p_grayscale,
        opt.kernel_size_blur,
        opt.sigma_blur,
        opt.p_blur,
        opt.threshold_solarize,
        opt.p_solarize,
        opt.mean,
        opt.std
    ).to(device)
    print("student train transform", student_train_transform)
    teacher_transform = get_normalize_transform(
        opt.mean,
        opt.std
    ).to(device)
    print("teacher train transform", teacher_transform)
    val_transform = get_normalize_transform(
        opt.mean,
        opt.std
    ).to(device)
    print("val transform", val_transform)

    teacher = SlotAttentionAutoEncoder(resolution, opt.num_slots, opt.hid_dim, output_channel = 3).to(device)
    teacher = nn.DataParallel(teacher)
    

    criterion = nn.MSELoss()

    params = [{'params': model.parameters()}]

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size,
                            shuffle=True, num_workers=opt.num_workers, drop_last=True)

    optimizer = optim.Adam(params, lr=opt.learning_rate)

    start = time.time()
    step = 0
    
   
    print('Model build finished!')
    for epoch in range(opt.num_epochs):

        if epoch == opt.start_teacher:
            print('____________________burn_in_ckpt', opt.burn_in_ckpt)
            print('____________________burn_in_exp', opt.burn_in_exp)
            print('____________________supervision', opt.supervision)
            print('____________________regularization', opt.weight_reg)

            model_path = os.path.join(opt.model_dir, opt.burn_in_exp, opt.burn_in_ckpt)
            model.load_state_dict(torch.load(model_path)['model_state_dict'])
            teacher.load_state_dict(torch.load(model_path)['model_state_dict'])

        model.train()

        total_loss = 0
        M_loss = 0
        N_loss = 0
        LR_loss = 0

        M_loss_t = 0
        N_loss_t = 0
        for sample in tqdm(train_dataloader):
            step += 1
        
            if step < opt.warmup_steps:
                learning_rate = opt.learning_rate * (step / opt.warmup_steps)
            else:
                learning_rate = opt.learning_rate

            learning_rate = learning_rate * (opt.decay_rate ** (
                step / opt.decay_steps))

            optimizer.param_groups[0]['lr'] = learning_rate
            
            image = sample['image'].to(device, non_blocking=True)
            mask_gt = sample['mask'].to(device, non_blocking=True)
            student_image = student_train_transform(image)
            teacher_image = teacher_transform(image)

            mask_gt = F.interpolate(mask_gt.float(), (120, 242)).long().to(device)

            if epoch >= opt.start_teacher: 
                with torch.no_grad():
                    _, masks_t, mask_fg_t, _ = teacher(teacher_image)
                    masks_t = masks_t * 0.999 + 1e-8
                    masks_t_quant = masks_t.argmax(2).cpu().numpy()
                    masks_t_quant = np.where(masks_t_quant==44, 0, masks_t_quant)

                    
                    # get connex regions
                    for i in range(opt.batch_size):
                        for j in range(5):
                            activated_slots = list(np.unique(masks_t_quant[i,j,:,:]))
                            
                            
                            empty_slots = [x for x in range(45) if x not in activated_slots]
                            for s in activated_slots:
                                tmp_bin = np.where(masks_t_quant[i,j,:,:] == s, 1, 0)
                                # Label each connected component
                                labeled_mask, num_features = label(tmp_bin)
                                label_counts = np.bincount(labeled_mask.ravel())
                                if num_features > 1:
                                    # starts from 2 to discard background and 1st connected region
                                    for ft in range(2, num_features + 1):
                                        if label_counts[ft] > 10 and mask_fg_t[i,j,0,:,:][labeled_mask == ft].mean() > opt.confid_th:
                                            
                                            try: # if no empty slot available, stop adding objects
                                                masks_t_quant[i,j,:,:] = np.where(labeled_mask == ft, empty_slots[0], masks_t_quant[i,j,:,:])
                                                empty_slots.remove(empty_slots[0])
                                            except:
                                                print('more teacher predictions than student slots')
                                                continue
                                        else:
                                            masks_t_quant[i,j,:,:] = np.where(labeled_mask == ft, 0, masks_t_quant[i,j,:,:])
                    


                    _h, _w = masks_t_quant[0,0,:,:].shape

                    masks_t_bin_np = np.zeros((opt.batch_size, 5,_h, _w))
                    
                    # binarize the teacher predictions
                    for i in range(opt.batch_size):
                        for j in range(5):
                            values, indices, counts = np.unique(masks_t_quant[i,j,:,:], return_inverse=True, return_counts=True)
                            to_eliminate = counts <= 10
                            mapping = np.arange(len(values))
                            mapping[to_eliminate] = 0
                            masks_t_bin_np[i,j,:,:] = mapping[indices].reshape((_h, _w))
                    masks_t_bin = torch.Tensor(masks_t_bin_np).long()
                    n_objects_t = masks_t_bin.max()
                    masks_t_bin = F.one_hot(masks_t_bin, n_objects_t+1)
                    masks_t_bin = masks_t_bin[:,:,:,:,1:]
                    masks_t_bin = masks_t_bin.permute(0,1,4,2,3).float()
                    masks_t_bin_gpu = masks_t_bin.to(device)
                    masks_t_bin_np = masks_t_bin.flatten(3,4)
                    masks_t_bin_np = masks_t_bin_np.detach().cpu().numpy() 

                      
            recon_combined, masks, mask_fg, slots = model(student_image)
            recon_combined = recon_combined.view(opt.batch_size,5,3,resolution[0],resolution[1])

            # reconstruction loss
            loss = criterion(recon_combined, student_image) 
            
            # mask loss
            loss_mask = 0.0
            loss_mask_t = 0.0

            NLL_loss = 0.0
            L1_reg = 0.0
            
            mask_detach = masks.detach().flatten(3,4)
            mask_detach = mask_detach * 0.999 + 1e-8
            mask_detach = mask_detach.cpu().numpy()
            n_objects = mask_gt.max()
            mask_gt = F.one_hot(mask_gt, n_objects+1)
            mask_gt = mask_gt[:,:,:,:,1:]
            mask_gt = mask_gt.permute(0,1,4,2,3).float()
            mask_gt_np = mask_gt.flatten(3,4)
            mask_gt_np = mask_gt_np.detach().cpu().numpy()

            scores = np.zeros((opt.batch_size, 5, opt.num_slots, n_objects)) 
            for i in range(opt.batch_size):
                for j in range(5):
                    cross_entropy_cur = np.matmul(np.log( mask_detach[i,j]), mask_gt_np[i,j].T) + np.matmul(np.log(1 - mask_detach[i,j]), (1 - mask_gt_np[i,j]).T)
                    scores[i,j] += cross_entropy_cur
            

            # weighted BCE
            if epoch >= opt.start_teacher: 
                loss_mask, filtered_fg_gt = computeWBCE_TS(opt, masks, mask_gt, scores, mask_fg_t=mask_fg_t, device=device)
            else:
                loss_mask = computeWBCE(opt, masks, mask_gt, scores)



             # NLL (reg on whole image)
            mask_fg = mask_fg * 0.999 + 1e-8
            fg_gt = mask_gt.sum(axis = 2)

            if epoch >= opt.start_teacher: 
                NLL_loss =  (- torch.log(mask_fg[:,:,0,:,:]) * filtered_fg_gt).mean()
            else:
                NLL_loss =  (- torch.log(mask_fg[:,:,0,:,:]) * fg_gt).mean()
            L1_reg = mask_fg[:,:,0,:,:].mean()


            # teacherBCE
            if epoch >= opt.start_teacher:
                scores = np.zeros((opt.batch_size, 5, opt.num_slots, n_objects_t)) 
                for i in range(opt.batch_size):
                    for j in range(5):
                        cross_entropy_cur = np.matmul(np.log( mask_detach[i,j]), masks_t_bin_np[i,j].T) + np.matmul(np.log(1 - mask_detach[i,j]), (1 - masks_t_bin_np[i,j]).T)
                        scores[i,j] += cross_entropy_cur
                loss_mask_t = computeWBCE_TS(opt, masks, masks_t_bin_gpu, scores, mask_fg_t, device=device, teacher = True)


            if epoch >= opt.start_teacher:
                whole_loss = loss + opt.weight_mask*loss_mask + opt.weight_NLL*NLL_loss + opt.weight_reg*L1_reg + loss_mask_t 
            else:
                whole_loss = loss + opt.weight_mask*loss_mask + opt.weight_NLL*NLL_loss + opt.weight_reg*L1_reg 

            optimizer.zero_grad()
            whole_loss.backward()
            clip_grad_norm_(model.parameters(),1.0)

            optimizer.step()

            # update teacher
            if epoch >= opt.start_teacher:
                _update_teacher_model(model, teacher, keep_rate=0.996)

            total_loss += loss.item()
            try:
                M_loss += loss_mask.item()*opt.weight_mask
            except:
                M_loss += loss_mask*opt.weight_mask
            N_loss += NLL_loss.item()*opt.weight_NLL
            LR_loss += L1_reg.item()*opt.weight_reg

            # teacher
            if epoch >= opt.start_teacher:
                try:
                    M_loss_t += loss_mask_t.item()
                except:
                    M_loss_t += loss_mask_t

            del recon_combined, masks, mask_fg, image, student_image, teacher_image, loss, whole_loss, mask_gt, loss_mask, NLL_loss, L1_reg, loss_mask_t
            # break


        total_loss /= len(train_dataloader)
        M_loss /= len(train_dataloader)
        N_loss /= len(train_dataloader)
        LR_loss /= len(train_dataloader)

        M_loss_t /= len(train_dataloader)
        N_loss_t /= len(train_dataloader)

        print ("Epoch: {}, Loss: {}, Loss_mask: {}, NLL_loss: {}, L1_reg: {}, Loss_mask_t: {}, Time: {}".format(epoch, total_loss,M_loss,
            N_loss, LR_loss, M_loss_t, datetime.timedelta(seconds=time.time() - start)))
        
        if not epoch % 10:
            rand_id = random.randint(0,190)
            sample: dict[str, Tensor] = test_set[0]
            image: Tensor = sample['image'][rand_id:rand_id+5,:,:,:].to(device, non_blocking=True)
            image = image.unsqueeze(0)

            recon_combined, masks, mask_fg, slots = model(image)

            index_mask = masks.argmax(dim = 2)
            index_mask = F.one_hot(index_mask,num_classes = opt.num_slots)
            index_mask = index_mask.permute(0,1,4,2,3)
            masks = masks * index_mask

            image = image[0]
            image = F.interpolate(image, (120,242))
            masks = masks[0]

            recon_combined = recon_combined.detach()
            masks = masks.detach()

            fig, ax = plt.subplots(math.ceil((opt.num_slots+2) / 10), 10, figsize=(45, 5 * math.ceil((opt.num_slots +2)/ 10)))
            for i in range(1):
                image_i = image[i]
                recon_combined_i = recon_combined[i]
                masks_i = masks[i].cpu().numpy()
                image_i = image_i.permute(1,2,0).cpu().numpy()
                image_i = image_i * 0.5 + 0.5
                recon_combined_i = recon_combined_i.permute(1,2,0)
                recon_combined_i = recon_combined_i.cpu().numpy()
                recon_combined_i = recon_combined_i * 0.5 + 0.5
                ax[i,0].imshow(image_i)
                ax[i,0].set_title('Image-f{}'.format(i))
                ax[i,1].imshow(recon_combined_i)
                ax[i,1].set_title('Recon.')
                for j in range(opt.num_slots):               
                    ax[(j+2)//10,(j + 2)%10].imshow(image_i)
                    ax[(j+2)//10,(j + 2)%10].imshow(masks_i[j], cmap = 'viridis', alpha = 0.6)
                    ax[(j+2)//10,(j + 2)%10].set_title('Slot %s' % str(j + 1))
                for j in range(math.ceil((opt.num_slots+2) / 10) * 10):
                    ax[(j)//10,(j)%10].grid(False)
                    ax[(j)//10,(j)%10].axis('off')
            eval_name = os.path.join(opt.sample_dir,opt.exp_name,'epoch_{}_slot.png'.format(epoch))
            fig.savefig(eval_name)
            plt.close(fig)
            plt.imsave(eval_name + '_fg.png', mask_fg.detach().cpu().numpy()[0,0,0,:,:])

            

            del masks, mask_fg, recon_combined, image, slots 
        
        if not epoch % 10:
            torch.save({
                'model_state_dict': model.state_dict(),
                }, os.path.join(opt.model_dir, opt.exp_name, 'epoch_{}.ckpt'.format(epoch))
                )
        
            test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                    shuffle=False, num_workers=4, drop_last=False)

            with torch.no_grad():
                ARIs, all_ARIs = [], []
                for idx, sample in enumerate(tqdm(test_dataloader)):
                    
                    image = sample['image'].to(device)
                    mask_gt = sample['mask']

                    mask_gts = F.interpolate(mask_gt.float(), (120, 242)).long()

                    for i in range(40):
                        _, masks, _, _= model(image[:,i*5:i*5+5,:,:,:])
                        mask_gt = mask_gts[:,i*5:i*5+5,:,:]
                        masks = masks.detach().cpu()
                        
                        gt_msk = mask_gt[0]
                        pred_msk = masks[0]
                        gt_msk = gt_msk.view(5,-1)
                        pred_msk = pred_msk.view(5,opt.num_slots,-1).permute(1,0,2)
                        gt_msk = gt_msk.view(-1)
                        pred_msk = pred_msk.reshape(opt.num_slots,-1)


                        pred_msk = pred_msk.permute(1,0)
                        gt_msk = F.one_hot(gt_msk)
                        _,n_cat = gt_msk.shape 
                        if n_cat <= 2:
                            continue

                        all_ari = ARI(gt_msk.unsqueeze(0), pred_msk.unsqueeze(0))
                        gt_msk = gt_msk[:,1:]
                        ari = ARI(gt_msk.unsqueeze(0), pred_msk.unsqueeze(0))
                        
                        # if not ari == ari: # one image in the val_set is causing instability in computing ari. Discarded
                        if not ari == ari:
                            print('discarded')
                            continue
                        
                        all_ARIs.append(all_ari)
                        ARIs.append(ari)
                    del image, mask_gt, masks
                print('final ARI:',sum(ARIs) / len(ARIs))
                print('final allARI:', sum(all_ARIs) / len(all_ARIs))
        

if __name__ == '__main__':
    main()
