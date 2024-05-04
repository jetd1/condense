mkdir -p ./data/ILSVRC2012_img_val && \
wget -O ./data/ILSVRC2012_img_val.tar https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar && \
tar -xvf ./data/ILSVRC2012_img_val.tar -C ./data/ILSVRC2012_img_val && \
rm ./data/ILSVRC2012_img_val.tar && \
wget -O ./data/ILSVRC2012_img_val/ILSVRC2012_val_labels.txt https://s3-haosu.nrp-nautilus.io/condense-release/misc/ILSVRC2012_val_labels.txt && \
echo "ImageNet downloaded."
