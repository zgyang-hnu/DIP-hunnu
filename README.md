# DIP 
This repo. provides the code used for "Domain-invariant prototypes for semantic segmentation"

We have tested our code on Ubuntu 18.04, Pytorch 1.7

The DIP provided an alternative solution to the commonly used self-training and adversarial training for domain adaptation in consideration of their training inefficiencty problem.

The DIP (see the Figure below) proposes to learn domain-invariant prototypes with a prototype-based few-shot learning framework. During training, we adopt few-shot annotated target images as the support set and treat all source images as the query set. In each training step, our model first embeds the support image and query image into semantic features using a Siamese Network. Then, a masked average pooling (MAP) operation is applied to the feature maps of support image to obtain class prototypes. Finally, predictions over the pixels of query image are obtained by finding the nearest class prototype to each pixel. Since the commonly used binary segmentation-based few-shot learning is not suited for high-resolution images containing an arbitrary number of classes, we propose a support image adaptive training strategy in which the classes to be segmented as well as their number of support samples are fully determined by the current support image. While the class labels that appeared in the support images are different and even nonconsecutive in different training
steps, we propose to replace the appeared class labels with their entry indices to ensure the class labels are consecutive so that the cross entropy loss can be applied. For the segmentation of a test image, we need only perform prototype-based segmentation with all the class prototypes that are pre-computed from these few-shot annotated target images

![image](https://github.com/zgyang-hnu/DIP-hunnu/blob/main/Framework.jpg)

# Train
## 1 Prepare the data  
You need first to download the dataset (e.g., GTA5, SYNTHIA, Cityscapes, BDD100K) used in this project by yourself.  After that, you may need to change the dataset path to your path in the trainDIP.py. 

## 2 training setups
The training setups such as learning rate, batch size, the path where the model saved and loaded are all located in .yaml file in the the config/pascal path.

## 3 training
Before running the trainDIP.py, activate the corresponding dataloader for different adaptation tasks.
For example, for the GTA5-to-Cityscapes adaptation task, you should activate the "dataset.CityGTA5NEW" dataloader for training.

# Test
## Get the class prototypes
Run the getsupp_pro.py, which reads the support images and extracted different class prototypes (class number * feature vectors) to a local pt file.

## Test
Run the testDIP.py, which reads the class prototypes from the local pt file, and then perform prototype-based semantic segmentation

# Pre-trained VGG ResNet
The pre-trained VGG ResNet can be downloaded at 
https://pan.baidu.com/s/1FwpcZQ3MPBrm8OvQT7N93Q?pwd=neey 

