mkdir -p ./data  && \
wget -O ./data/ADEChallengeData2016.zip http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip  && \
unzip ./data/ADEChallengeData2016.zip -d ./data  && \
wget -O ./data/ADEChallengeData2016/training.odgt https://raw.githubusercontent.com/CSAILVision/semantic-segmentation-pytorch/master/data/training.odgt  && \
wget -O ./data/ADEChallengeData2016/validation.odgt https://raw.githubusercontent.com/CSAILVision/semantic-segmentation-pytorch/master/data/validation.odgt  && \
rm ./data/ADEChallengeData2016.zip  && \
echo "ADE20k downloaded."
