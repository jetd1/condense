mkdir -p ./data  && \
wget -O ./data/VOCtrainval_11-May-2012.tar http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar  && \
tar -xvf ./data/VOCtrainval_11-May-2012.tar -C ./data  && \
rm ./data/VOCtrainval_11-May-2012.tar  && \
echo "VOC2012 downloaded."
