# Self Distillation Meets Object Discovery

[Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Kara_DIOD_Self-Distillation_Meets_Object_Discovery_CVPR_2024_paper.html) | [Video](https://www.youtube.com/watch?v=2uC4jSAt6-U)

This repository is the official implementation of [**DIOD: Self-Distillation Meets Object Discovery**](https://openaccess.thecvf.com/content/CVPR2024/html/Kara_DIOD_Self-Distillation_Meets_Object_Discovery_CVPR_2024_paper.html), published at CVPR 2024.

![Method Scheme](DIOD.png) 

## Set Up

Use the following command to create the environment from the DIOD_env.yml file:

```bash
conda env create -f DIOD_env.yml
```

## Datasets

To train/evaluate DIOD, please download the required datasets along with the pseudo-labels shared by [DOM](https://github.com/zpbao/Discovery_Obj_Move):

- [TRI-PD](https://openaccess.thecvf.com/content/CVPR2022/papers/Bao_Discovering_Objects_That_Can_Move_CVPR_2022_paper.pdf)
- [KITTI-train](https://www.cvlibs.net/datasets/kitti/raw_data.php), and [KITTI-test](https://www.cvlibs.net/datasets/kitti/eval_instance_seg.php?benchmark=instanceSeg2015)

## Training

**On TRI-PD:**

- For burn-in phase, run:
  
  ```bash
  # set start_teacher > num_epochs not to run distillation, example:
  python trainPD_ts.py --num_epochs 500\
  --start_teacher 501 
  ```

- For teacher-student training, run:

  ```bash
  python trainPD_ts.py --start_teacher 0\
  --burn_in_exp 'your_burn_in_experiment_directory'\
  --burn_in_ckpt 'ckpt_name'
  ```

- Our used configuration is set as default values.
- We provide [here](https://drive.google.com/drive/u/0/folders/1OJMQeH4gJu9D1aF9lQ9azs7tvkXW_SZJ) our model checkpoint at the end of burn in. It can be used to directly run distillation. Example:
  ```bash
  python trainPD_ts.py --start_teacher 0\
  --burn_in_exp 'checkpoints'\
  --burn_in_ckpt 'DIODPD_burn_in_400.ckpt'
  ```

**On KITTI:**

Models trained on KITTI are initialized from TRI-PD experiment, so for KITTI directly run:

  ```bash
  python trainKITTI_ts.py --start_teacher 0\
 --burn_in_exp 'TRI-PD_experiment_ckpt_directory'\
 --burn_in_ckpt 'ckpt_name'
  ```
- The used configuration is set as default values.
- Example using our checkpoint **DIODPD_500.ckpt** provided [here](https://drive.google.com/drive/u/0/folders/1OJMQeH4gJu9D1aF9lQ9azs7tvkXW_SZJ):
```bash
python trainKITTI_ts.py --start_teacher 0\
--burn_in_exp 'checkpoints'\
--burn_in_ckpt 'DIODPD_500.ckpt'
```

## Evaluation
- We provide [here](https://drive.google.com/drive/u/0/folders/1OJMQeH4gJu9D1aF9lQ9azs7tvkXW_SZJ) the checkpoints for our trained models.

For fg-ARI and all-ARI, run:

```bash
python evalPD_ARI.py       # for TRI-PD
python evalKITTI_ARI.py    # for KITTI
```

For F1 score, run:
```bash
python evalPD_F1_score.py       # for TRI-PD
python evalKITTI_F1_score.py    # for KITTI
```


**DIOD_DINOv2 is the version of DIOD that uses a ViT-S14 backbone pre-trained with DINOv2. Inside this directory, you can use the same commands provided above to run this version.**

## Acknowledgements

This code is built upon the [DOM](https://github.com/zpbao/Discovery_Obj_Move) codebase, and uses pre-trained models from [DINOv2](https://github.com/facebookresearch/dinov2). We thank the authors for their great work and for sharing their code/pre-trained models.

## Citation

```bibtex
@inproceedings{kara2024diod,
  title={DIOD: Self-Distillation Meets Object Discovery},
  author={Kara, Sandra and Ammar, Hejer and Denize, Julien and Chabot, Florian and Pham, Quoc-Cuong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3975--3985},
  year={2024}
}
```

## License

This project is under the CeCILL license 2.1.


