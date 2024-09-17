import os
import argparse
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from datasetKITTIEval_dinov2 import KITTIDataset
from utils import adjusted_rand_index as ARI 

from models.model_bg_dinov2 import *

parser = argparse.ArgumentParser()

parser.add_argument
parser.add_argument('--ckpt_path', default='/home/users/skara/check_release/checkpoints/DIODKITTI_100_dinov2.ckpt', type=str, help='pre-trained model path' )
parser.add_argument('--test_path', default = '/home/data/skara/KITTI_DOM/KITTI_DOM_test', type = str, help = 'path of KITTI test set')
parser.add_argument('--num_slots', default=45, type=int, help='Number of slots in Slot Attention.')
parser.add_argument('--hid_dim', default=64, type=int, help='hidden dimension size')

def main():
    opt = parser.parse_args()
    resolution = (378, 1260)
    model_path = opt.ckpt_path
    data_path = opt.test_path


    test_set = KITTIDataset(split = 'test', root = data_path)

    model = SlotAttentionAutoEncoder(resolution, opt.num_slots, opt.hid_dim, 3).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path)['model_state_dict'])

    print('model load finished!')

    for param in model.module.parameters():
        param.requires_grad = False


    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=8,
                                shuffle=True, num_workers=4, drop_last=False)

    ARIs, allARIs = [], []

    for sample in tqdm(test_dataloader):
        image = sample['image'].to(device)
        image = image.unsqueeze(1)
        mask_gt = sample['mask']
        mask_gt = mask_gt.unsqueeze(1)

        _, masks, _ , _= model(image)
        masks = masks.detach().cpu()

        for i in range(8):
            gt_msk = mask_gt[i]
            pred_msk = masks[i]
            gt_msk = gt_msk.view(1,-1)
            pred_msk = pred_msk.view(1,opt.num_slots,-1).permute(1,0,2)

            gt_msk = gt_msk.view(-1)
            pred_msk = pred_msk.reshape(opt.num_slots,-1)

            pred_msk = pred_msk.permute(1,0)
            gt_msk = F.one_hot(gt_msk)

            all_ari = ARI(gt_msk.unsqueeze(0), pred_msk.unsqueeze(0))

            _, n_cat = gt_msk.shape 
            if n_cat <= 2:
                continue
            
            gt_msk = gt_msk[:,1:]

            ari = ARI(gt_msk.unsqueeze(0), pred_msk.unsqueeze(0))
            
            allARIs.append(all_ari)
            ARIs.append(ari)

        del image, masks, mask_gt

    print('final ARI:',sum(ARIs) / len(ARIs))
    print('final all-ARI:',sum(allARIs) / len(allARIs))

if __name__ == '__main__':
    main()
