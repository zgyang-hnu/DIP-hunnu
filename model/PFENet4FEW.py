import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import random
import time
import cv2

import model.resnet as models
import model.vgg as vgg_models


def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask#*element-wise mul 仅保留 MASK标记为1处的特征 其他统一置0
    #print(np.unique(mask.cpu().numpy())) [0,1]
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    #统计MASK中为1的数量
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
     #计算保留特征的平均值 即可获取MASK 的平均特征向量
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area  
   
    return supp_feat
  
def get_vgg16_layer(model,bn=False):
    if bn:
      layer0_idx = range(0,7)
      layer1_idx = range(7,14)
      layer2_idx = range(14,24)
      layer3_idx = range(24,34)
      layer4_idx = range(34,43)
    else:
      layer0_idx = range(0,5)
      layer1_idx = range(5,10)
      layer2_idx = range(10,17)
      layer3_idx = range(17,23)
      layer4_idx = range(23,29)

    layers_0 = []
    layers_1 = []
    layers_2 = []
    layers_3 = []
    layers_4 = []
    for idx in layer0_idx:
        layers_0 += [model.features[idx]]
    for idx in layer1_idx:
        layers_1 += [model.features[idx]]
    for idx in layer2_idx:
        layers_2 += [model.features[idx]]
    for idx in layer3_idx:
        layers_3 += [model.features[idx]]
    for idx in layer4_idx:
        layers_4 += [model.features[idx]]  
    layer0 = nn.Sequential(*layers_0) 
    layer1 = nn.Sequential(*layers_1) 
    layer2 = nn.Sequential(*layers_2) 
    layer3 = nn.Sequential(*layers_3) 
    layer4 = nn.Sequential(*layers_4)
    return layer0,layer1,layer2,layer3,layer4


class ASPP(nn.Module):
    """
    Atrous spatial pyramid pooling (ASPP)
    """

    def __init__(self, in_ch, out_ch, rates):
        super(ASPP, self).__init__()
        for i, rate in enumerate(rates):
            self.add_module(
                "c{}".format(i),
                nn.Conv2d(in_ch, out_ch, 3, 1, padding=rate, dilation=rate, bias=True),
            )

        for m in self.children():
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return sum([stage(x) for stage in self.children()])

class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                #nn.GroupNorm(8,reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)

class FEWDomain(nn.Module):
    def __init__(self, layers=101, classes=2, zoom_factor=8, \
        criterion=nn.CrossEntropyLoss(ignore_index=255), BatchNorm=nn.BatchNorm2d, \
        pretrained=True, sync_bn=True, shot=1, ppm_scales=[60, 30, 15, 8], vgg=False):
        super(FEWDomain, self).__init__()
        assert layers in [50, 101, 152]
        print(ppm_scales)
        assert classes > 1
        from torch.nn import BatchNorm2d as BatchNorm        
        self.zoom_factor = zoom_factor
        self.criterion = criterion
        self.shot = shot
        #self.ppm_scales = ppm_scales
        self.vgg = vgg

        models.BatchNorm = BatchNorm
        
        if self.vgg:
            print('INFO: Using VGG_16 bn')
            vgg_models.BatchNorm = BatchNorm
            vgg16 = vgg_models.vgg16(pretrained=pretrained)
            print(vgg16)
            self.layer0, self.layer1, self.layer2, \
                self.layer3, self.layer4 = get_vgg16_layer(vgg16)

        else:
            print('INFO: Using ResNet {}'.format(layers))
            if layers == 50:
                resnet = models.resnet50(pretrained=pretrained)
            elif layers == 101:
                resnet = models.resnet101(pretrained=pretrained)
            else:
                resnet = models.resnet152(pretrained=pretrained)
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2, resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
            self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            #self.aspp = ASPP(2048,19, [6, 12, 18, 24]) 
        feat_dim =512
        reduce_dim = 256*2
        if self.vgg:
            fea_dim = 512 + 256
        else:
            fea_dim = 2048 #1024 + 512       
        reduce_dim = 256*4
        self.ppm = PPM(fea_dim,int(fea_dim/4),[1,2,3,6])


        

        self.projector = nn.Sequential(
              nn.Conv2d(reduce_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
              nn.BatchNorm2d(reduce_dim),
              nn.ReLU(inplace=True),
              nn.Conv2d(reduce_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
              nn.BatchNorm2d(reduce_dim),
              nn.ReLU(inplace=True), #uncomment when using few-shot for fully supervised training
            # nn.Dropout2d(p=0.5) ,
            #nn.Conv2d(reduce_dim, 19, kernel_size=1, padding=0) 
              nn.Conv2d(reduce_dim, reduce_dim, kernel_size=1, padding=0, bias=False)         
        )  

        self.down = nn.Sequential(
            nn.Conv2d(fea_dim*2, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(reduce_dim),
            nn.ReLU(inplace=True)                        
        )
                               


    def forward(self, x,  train_label_list=None, y=None):
        x_size = x.size()
        print(x.shape)
        #assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        #h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        #w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)
        h,w = x_size[2],x_size[3]
        n_classes = len(train_label_list)
        #print(x_size,s_x.shape,s_y.shape,y.shape)


        #   Support Feature     
        prototype_list = []
        prototype_num_eachclass = [j-j for j in range(n_classes)]#不同shot图像 可能为同一类型贡献多个Prototype
        final_supp_list = []
        mask_list = []
        #print(s_x.shape)
        
        #prototype_tensor = torch.stack(prototype_list,dim=0)
        #print(prototype_tensor.shape)
        #torch.save(prototype_tensor,'support_tensor.pt')

        prototype_tensor = torch.load("support_tensorcity1226.pt")
        print(prototype_tensor.shape) 
        for i in range(prototype_tensor.shape[0]):
          prototype_list.append(prototype_tensor[i,:])         
        
        #   Query Feature
        with torch.no_grad():
            query_feat_0 = self.layer0(x)
            query_feat_1 = self.layer1(query_feat_0)
            query_feat_2 = self.layer2(query_feat_1)
            query_feat_3 = self.layer3(query_feat_2)  
            query_feat_4 = self.layer4(query_feat_3)#高阶语义特征用于后面Prior mask generation
            #if self.vgg:
            #    query_feat_4 = F.interpolate(query_feat_4, size=(query_feat_2.size(2),query_feat_2.size(3)), mode='bilinear', align_corners=True)

        query_feat = query_feat_4#torch.cat([query_feat_4, query_feat_2], 1)#网络中间层特征用于预测和分类
        query_feat = self.ppm(query_feat)
        query_feat = self.down(query_feat)#通道降维
        #query_feat = F.normalize(query_feat,dim=1)
        # if True:#self.training:
        query_feat = self.projector(query_feat)
        #query_feat = self.deeplabv1head(query_feat_4)
        dist = [self.calDist(query_feat, prototype) for prototype in  prototype_list]
        #if dist is None:
        
        out = torch.stack(dist, dim=1)  # 
        #print('outoutoutoutoutouuuuuuuuuuuuuuuuuuuuuuuuuuuu', len( prototype_list), out.shape)
              

        #   Output Part
        if self.zoom_factor != 1:
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            main_loss = self.criterion(out, y.long())#
            
            return out.max(1)[1], main_loss
        else:
            return out

    def calDist(self, fts, prototype, scaler=20):
        """
        Calculate the distance between features and prototypes: taken from Github PANet

        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """
        prototype = prototype.unsqueeze(0)#torch.squeeze(prototype).unsqueeze(0)#transform the protopye from 1*c*1*1 to 1*c
        #print(prototype.shape)
        dist = F.cosine_similarity(fts, prototype[..., None, None], dim=1) * scaler#...省略前面所有的 ‘：‘索引操作,None的作用主要是在使用None的位置新增一个维度。
        return dist           

