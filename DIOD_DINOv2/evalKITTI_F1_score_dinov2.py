import numpy as np
import torch
import os
import argparse
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from datasetKITTIEval_dinov2 import KITTIDataset
from utils import adjusted_rand_index as ARI 
import matplotlib.pyplot as plt
from models.model_bg_dinov2 import *

EPS = 1e-10


parser = argparse.ArgumentParser()

parser.add_argument('--ckpt_path', default='/home/users/skara/check_release/checkpoints/DIODKITTI_100_dinov2.ckpt', type=str, help='pre-trained model path' )
parser.add_argument('--test_path', default = '/home/data/skara/KITTI_DOM/KITTI_DOM_test', type = str, help = 'path of KITTI test set')
parser.add_argument('--num_slots', default=45, type=int, help='Number of slots in Slot Attention.')
parser.add_argument('--hid_dim', default=64, type=int, help='hidden dimension size')
parser.add_argument('--is_bg_handled', default=True, type=bool, help='is_bg_handled')


resolution = (378, 1260)


def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap



def calculate_iou(mask1, set_masks):
    # Initialize a list to store the IoU values
    iou_scores = []

    # Convert masks to NumPy arrays
    mask1 = np.array(mask1, dtype=bool)

    for mask2 in set_masks:
        mask2 = np.array(mask2, dtype=bool)

        # Calculate intersection and union
        intersection = np.logical_and(mask1, mask2)
        union = np.logical_or(mask1, mask2)

        # Compute IoU
        iou = np.sum(intersection) / np.sum(union)
        iou_scores.append(iou)

    return iou_scores

def voc_eval(gt_masks,
               pred_scores, pred_masks,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file


    
    nb_images = len(pred_masks) #added
    image_ids = []
    class_recs = {}
    nb_gt = 0

    for im in range(nb_images):
        image_ids += [im]*len(pred_masks[im])
        class_recs[im] = [False] * len(gt_masks[im]) 
        nb_gt += len(gt_masks[im])
    
    
    
    # flatten preds and scores
    pred_scores_flat = np.array([item for sublist in pred_scores for item in sublist])
    

    pred_masks_flat = np.stack([item for sublist in pred_masks for item in sublist])
    
    

    # sort by confidence
    sorted_ind = np.argsort(-pred_scores_flat)
    pred_masks_flat = pred_masks_flat[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        mask = pred_masks_flat[d]
        ovmax = -np.inf
        MASKSGT = gt_masks[image_ids[d]]

        
        # compute overlaps
        overlaps = calculate_iou(mask, MASKSGT)

        # case of no gt in one frame
        if not len(overlaps):
            ovmax = 0
        else:
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        
        if ovmax > ovthresh:
                if not R[jmax]:
                    tp[d] = 1.
                    R[jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)

    # # check F1 score
    # print('check_FP = ', fp[-1])
    # print('check_TP = ', tp[-1])
    # print('check_FN = ', nb_gt - tp[-1])
    # print('check_F1_score', 2*tp[-1]/(2*tp[-1] + fp[-1] + nb_gt - tp[-1]))



    rec = tp / float(nb_gt)
    
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap

def compute_f1(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def main():
    opt = parser.parse_args()
    model_path = opt.ckpt_path
    data_path = opt.test_path
    test_set = KITTIDataset(split = 'test', root = data_path)
   

    model = SlotAttentionAutoEncoder(resolution, opt.num_slots, opt.hid_dim).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    
    print('model load finished!')

    for param in model.module.parameters():
        param.requires_grad = False


    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=8,
                                shuffle=False, num_workers=4, drop_last=False)

    
    active_slots, scores, all_gt = [], [], []
    

    nb_samples = 0
    for  sample in tqdm(test_dataloader):
        
        # evaluate our method with background modeling (within last slot)
        if opt.is_bg_handled: 
            nb_samples += 1
            image = sample['image'].to(device)
            image = image.unsqueeze(1)
            mask_gt = sample['mask']
            mask_gt = mask_gt.unsqueeze(1)

            _, masks, _ ,_= model(image)
            masks = masks.detach().cpu()

            for i in range(8):

                gt_msk = F.one_hot(mask_gt[i,0,:,:])

                pred_msk = F.one_hot(masks[i,0,:,:,:].argmax(dim=0, keepdim=False))
                                            
                active_slots.append([pred_msk[:,:,s] for s in range(opt.num_slots-1) if pred_msk[:,:,s].max().item()>0])

                scores.append([masks[i,0,s,:,:][pred_msk[:,:,s] == 1].mean() for s in range(opt.num_slots-1) if pred_msk[:,:,s].max().item()>0])

                all_gt.append([gt_msk[:,:,s] for s in range(1, gt_msk.shape[2]) if gt_msk[:,:,s].max().item()>0])

        # evaluate other baselines w/o background modeling           
        else:
            nb_samples += 1
            image = sample['image'].to(device)
            image = image.unsqueeze(1)
            mask_gt = sample['mask']
            mask_gt = mask_gt.unsqueeze(1)

            _, masks, _ = model(image)
            masks = masks.detach().cpu()

            for i in range(8):

                gt_msk = F.one_hot(mask_gt[i,0,:,:])

                pred_msk = F.one_hot(masks[i,0,:,:,:].argmax(dim=0, keepdim=False), num_classes=opt.num_slots)
                
                active_slots.append([pred_msk[:,:,s] for s in range(opt.num_slots) if pred_msk[:,:,s].max().item()>0])

                scores.append([masks[i,0,s,:,:][pred_msk[:,:,s] == 1].mean() for s in range(opt.num_slots) if pred_msk[:,:,s].max().item()>0])

                all_gt.append([gt_msk[:,:,s] for s in range(1, gt_msk.shape[2]) if gt_msk[:,:,s].max().item()>0])
         
        del image, mask_gt, masks
        
    
    rec, prec, _ = voc_eval(all_gt, scores, active_slots, 0.5)
    f1_score = compute_f1(prec[-1], rec[-1])
    print(f"F1 Score: {f1_score}")


if __name__ == '__main__':
    main()