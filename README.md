# DIP
This repo. provides the code used for "Domain-invariant prototypes for semantic segmentation"
We have tested our code on Ubuntu 18.04, Pytorch 1.7

The DIP provided an alternative solution to the commonly used self-training and adversarial training for domain adaptation in consideration of their training inefficiencty problem.

The DIP proposes to learn domain-invariant prototypes with a prototype-based few-shot learning framework. During training, we adopt few-shot annotated target images as the support set and treat all source images as the query set. In each training step, our model first embeds the support image and query image into semantic features using a Siamese Network. Then, a masked average pooling (MAP) operation is applied to the feature maps of support image to obtain class prototypes. Finally, predictions over the pixels of query image are obtained by finding the nearest class prototype to each pixel. Since the commonly used binary segmentation-based few-shot learning is not suited for high-resolution images containing an arbitrary number of classes, we propose a support image adaptive training strategy in which the classes to be segmented as well as their number of support samples are fully determined by the current support image. While the class labels that appeared in the support images are different and even nonconsecutive in different training
steps, we propose to replace the appeared class labels with their entry indices to ensure the class labels are consecutive so that the cross entropy loss can be applied. For the segmentation of a test image, we need only perform prototype-based segmentation with all the class prototypes that are pre-computed from these few-shot annotated target images

![image](https://github.com/zgyang-hnu/DIP-hunnu/blob/master/Framework.jpg)

# Train
Before running the trainDIP.py, activate the corresponding dataloader for different adaptation task

# Test
For domain adaptation, you should run the getsupp_pro.py to obtain the offline class prototypes and then run testDIP.py on the corresponding test dataset

# Pre-trained VGG ResNet
The pre-trained VGG ResNet can be downloaded at 
https://pan.baidu.com/s/1FwpcZQ3MPBrm8OvQT7N93Q?pwd=neey 

