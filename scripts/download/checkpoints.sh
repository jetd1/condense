mkdir -p ./ckpt

# Original DINOv2 weights
wget -O ./ckpt/dinov2_vitg14_pretrain.pth https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth
wget -O ./ckpt/dinov2_vitg14_ade20k_linear_head.pth https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_ade20k_linear_head.pth
wget -O ./ckpt/dinov2_vitg14_voc2012_linear_head.pth https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_voc2012_linear_head.pth
wget -O ./ckpt/dinov2_vitg14_imagenet_linear_head.pth https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_linear_head.pth
wget -O ./ckpt/dinov2_vitg14_places205_linear_head.pth https://s3-haosu.nrp-nautilus.io/condense-release/dinov2_vitg14_places205_linear_head.pth

# ConDense weights
wget -O ./ckpt/condense_vitg14.pth https://s3-haosu.nrp-nautilus.io/condense-release/condense_vitg14.pth
wget -O ./ckpt/condense_vitg14_ade20k_linear_head.pth https://s3-haosu.nrp-nautilus.io/condense-release/condense_vitg14_ade20k_linear_head.pth
wget -O ./ckpt/condense_vitg14_voc2012_linear_head.pth https://s3-haosu.nrp-nautilus.io/condense-release/condense_vitg14_voc2012_linear_head.pth
wget -O ./ckpt/condense_vitg14_imagenet_linear_head.pth https://s3-haosu.nrp-nautilus.io/condense-release/condense_vitg14_imagnet_linear_head.pth
wget -O ./ckpt/condense_vitg14_places205_linear_head.pth https://s3-haosu.nrp-nautilus.io/condense-release/condense_vitg14_places205_linear_head.pth

# ConDense 3D backbone weights
wget -O ./ckpt/condense_vitg14.pth https://s3-haosu.nrp-nautilus.io/condense-release/condense3d_minkunet34.pth
wget -O ./ckpt/condense_vitg14.pth https://s3-haosu.nrp-nautilus.io/condense-release/condense3d_minkunet14.pth
wget -O ./ckpt/condense_vitg14.pth https://s3-haosu.nrp-nautilus.io/condense-release/condense3d_minkresnet101.pth

# ConDense 3D head weights
