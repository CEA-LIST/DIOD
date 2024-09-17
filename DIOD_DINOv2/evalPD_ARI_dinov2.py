import os
import argparse
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from datasetPDEval_dinov2 import PDDataset
from utils import adjusted_rand_index as ARI 
import matplotlib.pyplot as plt


from models.model_bg_dinov2 import *

parser = argparse.ArgumentParser()

parser.add_argument
parser.add_argument('--ckpt_path', default='/home/users/skara/check_release/checkpoints/DIODPD_500_dinov2.ckpt', type=str, help='pre-trained model path' )
parser.add_argument('--test_path', default = '/home/data/skara/test_video', type = str, help = 'path of PD test set')
parser.add_argument('--num_slots', default=45, type=int, help='Number of slots in Slot Attention.')
parser.add_argument('--hid_dim', default=64, type=int, help='hidden dimension size')


resolution = (490, 980)

def main():
    opt = parser.parse_args()
    model_path = opt.ckpt_path
    data_path = opt.test_path
    test_set = PDDataset(split = 'test', root = data_path)

    model = SlotAttentionAutoEncoder(resolution, opt.num_slots, opt.hid_dim, 3).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path)['model_state_dict'])

    print('model load finished!')

    for param in model.module.parameters():
        param.requires_grad = False


    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                shuffle=False, num_workers=4, drop_last=False)

    ARIs, all_ARIs = [], []

    for idx, sample in enumerate(tqdm(test_dataloader)):
        image = sample['image'].to(device)
        mask_gt = sample['mask']

        """
        for i in range (5):
            plt.imsave('./infer/epoch200_moving/eval_pd_mask{}.png'.format(i), mask_gt[0, i,:,:].unsqueeze(2).detach().cpu().numpy())
            plt.imsave('./infer/epoch200_moving/eval_pd_image{}.png'.format(i), image[0, i,:,:,:].permute(1,2,0).detach().cpu().numpy()*0.5+0.5)
        """

        mask_gts = F.interpolate(mask_gt.float(), (123, 245)).long()

        for i in range(40):
            _, masks, _, _ = model(image[:,i*5:i*5+5,:,:,:])
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
            print('ari', ari)
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