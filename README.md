# Pneumonia X-ray Classification with ResNet18

This project is a deep learning-based medical imaging classifier that detects pneumonia from chest X-rays using a fine-tuned ResNet18 architecture. 
It was able to achieve 98% test accuracy on a set of 1812 test X-ray images.
It demonstrates the potential of convolutional neural networks in healthcare diagnostics.

**Please keep in mind, this is my first ever ML project. I self thought myself all content with a big shoutout to Mr. Daniel Bourke for his free PyTorch course on YouTube with around 25 hours of great learning material (https://www.youtube.com/watch?v=V_xro1bcAuA&t=2598s). Amazing time that we live in!**

## Project overview

**Goal**: Classify chest X-ray images as either 'NORMAL' or 'PNEUMONIA'.

**Model**: Pretrained ResNet18 modified for grayscale input and binary classification.

**Dataset**: Balanced version of the Chest X-ray dataset from Kaggle https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia. Original dataset accredited to https://data.mendeley.com/datasets/rscbjbr9sj/2.
NOTE: I manually balanced out the dataset so the model wouldn't have a huge bias towards 'PNEUMONIA' cases; original dataset from Kaggle was skewed. 
- Train set: PNEUMONIA images ~74.3%, NORMAL make up only ~25.7%.
- Test: PNEUMONIA images ~62.5%, NORMAL make up ~37.5%
- Validation: 9/9 (balanced)
Also the data splits between train, test, and validation is not correctly distributed so this had to be adjusted also. 
- Kaggle dataset split: *Train: 89.04%; Test: 10.65%; Validation: 0.31%* (PNEUMONIA: ~73%; NORMAL: ~27%)
- Edited dataset split: *Train: 46.4%; Test 26.65; Validation: 26.6%* (PNEUMONIA: ~52%; NORMAL: ~48%)
**The split should really be at least 70/15/15; That's my bad, plenty of Data Science for me to master.**

**Accuracy**: ~98% on validation set. ~98% on test set. 

**Framework**: PyTorch, Torchvision

## Project structure
xray_pneumonia_project/
│
├── denver_xray_model.ipynb         # Jupyter notebook (I used PyCharm IDE)
├── README.md                       # Project overview, setup, usage, and results
├── requirements.txt                # required libraries
├── resnet18-f37072fd.pth           # Pretrained ResNet weights 
├── xray_model_weights.pth          # model’s state_dict

## Model Architecture

- Modified ResNet18
  - first conv layer edited to accept 1-channel (grayscale) images.
  - Final FC layer changed to 2 output units (binary classification).
- Data augmentation applied on training set:
  - Random horizontal flip
  - Random rotation
  - Small Translations

## How to Run
1. Clone the repository:
bash:
git clone https://github.com/elfudo3/xray_pneumonia_project.git
cd xray_pneumonia_project

2. (Optional) Create virtual enviroment
bash: 
python3 -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

3. Install dependencies
bash:
pip install -r requirements.txt

4. (These are included in the project folder, but if theres an issue): Download pretrained ResNet18 weights from: https://download.pytorch.org/models/resnet18-f37072fd.pth
Save it as:
bash:
wget https://download.pytorch.org/models/resnet18-f37072fd.pth

5. Run the notebook
   Make sure you have Jupyter installed. Then run:
bash:
jupyter notebook 

6. Open xray_project.ipynb in the browser and execute cells step by step

## Credits
- Dataset: Paul Mooney on Kaggle
- Pretrained ResNet18: PyTorch Model Zoo
- Course inspiration: Daniel Bourke - https://www.youtube.com/watch?v=V_xro1bcAuA
- Kaggle notebook used as guideline: [Denver Magtibay] https://www.kaggle.com/code/denvermagtibay/pneumonia-detection-with-resnet-pytorch

## Future improvements
- Better data split (70/15/15)
- Building custom CNN from scratch
- Lot's of other improvements that I am not even aware of yet! -> but will be as I continue my venture into Machine Learning. :)


