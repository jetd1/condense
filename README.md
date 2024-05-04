# condense
ConDense backbone, weights, and evaluation code. This repo aims to replicate most of the main experiments covered in the ConDense paper.

## Prepare Environment
```bash
./env.sh
```

## Prepare Data and Weights
```bash
# Download data, replace {DATASET_NAME} with voc2012, ade20k, imagenet, or places205
./scripts/download_{DATASET_NAME}.sh

# Download weights, including ConDense, DINOv2, and corresponding head weights
./scripts/download_ckpts.sh
```

The Places205 dataset was taken down by the original authors due to multiple concerns, and not supported in this repo.

## Evaluation Results

| Model        | Task           | Dataset     | Eval mIoU/Acc (This Repo) | Reported mIoU/Acc |
|--------------|----------------|-------------|---------------------------|-------------------|
| ConDense-g14 | Segmentation   | VOC2012     | 85.388                    | 85.1              |
| DinoV2-g14   | Segmentation   | VOC2012     | 83.181                    | 83.0              |
| ConDense-g14 | Segmentation   | ADE20k      | 53.450                    | 53.6              |
| DinoV2-g14   | Segmentation   | ADE20k      | 48.989                    | 49.0              |
| ConDense-g14 | Classification | ImageNet-1k | 90.130                    | 89.6              |
| DinoV2-g14   | Classification | ImageNet-1k | 86.618                    | 86.5              |
| ConDense-g14 | Classification | Places205   | -                         | 70.2              |
| DinoV2-g14   | Classification | Places205   | -                         | 67.5              |


## Evaluate with Pretrained Weights
```bash
# Segmentation
PYTHONPATH=. python ./scripts/eval_seg.py -c ./config/seg_voc2012_dinov2_standard.yaml
PYTHONPATH=. python ./scripts/eval_seg.py -c ./config/seg_ade20k_dinov2_standard.yaml

# Classification
PYTHONPATH=. python ./scripts/eval_cls.py -c ./config/cls_imagenet_dinov2_standard.yaml
PYTHONPATH=. python ./scripts/eval_cls.py -c ./config/cls_places205_dinov2_standard.yaml
```
You can change the first several lines in `yaml` configs to switch between different backbones and weights.


## TODOs
- [ ] Graceful way to add support for Places205 dataset
- [ ] Depth Evaluations
- [ ] lin. 4 Evaluations
- [ ] 2D Retrieval Evaluations
- [ ] 3D Backbone Impl and Weights
- [ ] 3D Backbone Evaluations
