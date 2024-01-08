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

class PFENet(nn.Module):
    def __init__(self, layers=50, classes=2, zoom_factor=8, \
        criterion=nn.CrossEntropyLoss(ignore_index=255), BatchNorm=nn.BatchNorm2d, \
        pretrained=True, sync_bn=True, shot=1, ppm_scales=[60, 30, 15, 8], vgg=False):
        super(PFENet, self).__init__()
        assert layers in [50, 101, 152]
        print(ppm_scales)
        assert classes > 1
        from torch.nn import BatchNorm2d as BatchNorm        
        self.zoom_factor = zoom_factor
        self.criterion = criterion
        self.shot = shot
        self.ppm_scales = ppm_scales
        self.vgg = vgg

        models.BatchNorm = BatchNorm
        
        if self.vgg:
            print('INFO: Using VGG_16 bn')
            vgg_models.BatchNorm = BatchNorm
            vgg16 = vgg_models.vgg16_bn(pretrained=pretrained)
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

        reduce_dim = 256
        if self.vgg:
            fea_dim = 512 + 256
        else:
            fea_dim = 1024 + 512       

        self.cls = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),                 
            nn.Conv2d(reduce_dim, classes, kernel_size=1)
        )                 

        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)                  
        )
        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)                   
        )  

        self.pyramid_bins = ppm_scales
        self.avgpool_list = []
        for bin in self.pyramid_bins:
            if bin > 1:
                self.avgpool_list.append(
                    nn.AdaptiveAvgPool2d(bin)
                )


        factor = 1
        mask_add_num = 1
        self.init_merge = []
        self.beta_conv = []
        self.inner_cls = []        
        for bin in self.pyramid_bins:
            self.init_merge.append(nn.Sequential(
                nn.Conv2d(reduce_dim*2 + mask_add_num, reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
            ))                      
            self.beta_conv.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True)
            ))            
            self.inner_cls.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.1),                 
                nn.Conv2d(reduce_dim, classes, kernel_size=1)
            ))            
        self.init_merge = nn.ModuleList(self.init_merge) 
        self.beta_conv = nn.ModuleList(self.beta_conv)
        self.inner_cls = nn.ModuleList(self.inner_cls)                             


        self.res1 = nn.Sequential(
            nn.Conv2d(reduce_dim*len(self.pyramid_bins), reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),                          
        )              
        self.res2 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),   
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),                             
        )                        
     
        self.GAP = nn.AdaptiveAvgPool2d(1)

        self.alpha_conv = []
        for idx in range(len(self.pyramid_bins)-1):
            self.alpha_conv.append(nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU()
            ))     
        self.alpha_conv = nn.ModuleList(self.alpha_conv)
     


    def forward(self, x, s_x=torch.FloatTensor(1,1,3,473,473).cuda(), s_y=torch.FloatTensor(1,1,473,473).cuda(), y=None):
        x_size = x.size()
        assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        #   Query Feature
        with torch.no_grad():
            query_feat_0 = self.layer0(x)
            query_feat_1 = self.layer1(query_feat_0)
            query_feat_2 = self.layer2(query_feat_1)
            query_feat_3 = self.layer3(query_feat_2)  
            query_feat_4 = self.layer4(query_feat_3)#高阶语义特征用于后面Prior mask generation
            if self.vgg:
                query_feat_2 = F.interpolate(query_feat_2, size=(query_feat_3.size(2),query_feat_3.size(3)), mode='bilinear', align_corners=True)

        query_feat = torch.cat([query_feat_3, query_feat_2], 1)#网络中间曾特征用于预测和分类
        query_feat = self.down_query(query_feat)#通道降维

        #   Support Feature     
        supp_feat_list = []
        final_supp_list = []
        mask_list = []
        for i in range(self.shot):
            mask = (s_y[:,i,:,:] == 1).float().unsqueeze(1)
            mask_list.append(mask)
            with torch.no_grad():
                supp_feat_0 = self.layer0(s_x[:,i,:,:,:])
                supp_feat_1 = self.layer1(supp_feat_0)
                supp_feat_2 = self.layer2(supp_feat_1)
                supp_feat_3 = self.layer3(supp_feat_2)
                mask = F.interpolate(mask, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear', align_corners=True)
                #print(np.unique(mask.cpu().numpy())) [0,1]
                supp_feat_4 = self.layer4(supp_feat_3*mask)
                final_supp_list.append(supp_feat_4)#高阶语义特征用于后面Prior mask generation
                if self.vgg:
                    supp_feat_2 = F.interpolate(supp_feat_2, size=(supp_feat_3.size(2),supp_feat_3.size(3)), mode='bilinear', align_corners=True)
            
            supp_feat = torch.cat([supp_feat_3, supp_feat_2], 1)
            supp_feat = self.down_supp(supp_feat)
            #print(supp_feat.shape) torch.Size([1, 256, 60, 60])
            print('zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz',mask.shape)
            supp_feat = Weighted_GAP(supp_feat, mask)
            #print(supp_feat.shape) torch.Size([1, 256, 1, 1])

            supp_feat_list.append(supp_feat)

        #prior mask generation
        corr_query_mask_list = []
        cosine_eps = 1e-7
        for i, tmp_supp_feat in enumerate(final_supp_list):
            resize_size = tmp_supp_feat.size(2)
            tmp_mask = F.interpolate(mask_list[i], size=(resize_size, resize_size), mode='bilinear', align_corners=True)

            tmp_supp_feat_4 = tmp_supp_feat * tmp_mask                    
            q = query_feat_4
            s = tmp_supp_feat_4
            bsize, ch_sz, sp_sz, _ = q.size()[:]#返回 query  特征的尺寸 训练包 通道数 空间尺寸

            tmp_query = q
            tmp_query = tmp_query.contiguous().view(bsize, ch_sz, -1)
            tmp_query_norm = torch.norm(tmp_query, 2, 1, True) 

            tmp_supp = s               
            tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1) 
            tmp_supp = tmp_supp.contiguous().permute(0, 2, 1) 
            tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True) 
            #计算向量之间的余弦相似性
            similarity = torch.bmm(tmp_supp, tmp_query)/(torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)   
            similarity = similarity.max(1)[0].view(bsize, sp_sz*sp_sz)   
            similarity = (similarity - similarity.min(1)[0].unsqueeze(1))/(similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
            corr_query = similarity.view(bsize, 1, sp_sz, sp_sz)
            corr_query = F.interpolate(corr_query, size=(query_feat_3.size()[2], query_feat_3.size()[3]), mode='bilinear', align_corners=True)
            corr_query_mask_list.append(corr_query)  #将多张Support 图片生成的 prior mask放到一个列表中 
        corr_query_mask = torch.cat(corr_query_mask_list, 1).mean(1).unsqueeze(1)  # 
        corr_query_mask = F.interpolate(corr_query_mask, size=(query_feat.size(2), query_feat.size(3)), mode='bilinear', align_corners=True)  

        if self.shot > 1:
            supp_feat = supp_feat_list[0]
            for i in range(1, len(supp_feat_list)):
                supp_feat += supp_feat_list[i]
            supp_feat /= len(supp_feat_list)#将多张Support 图片获取的支撑向量 求平均 得到最终支撑向量

        out_list = []
        pyramid_feat_list = []

        #论文中的inter-source enrichment inter-scale interaction
        for idx, tmp_bin in enumerate(self.pyramid_bins):
            if tmp_bin <= 1.0:
                bin = int(query_feat.shape[2] * tmp_bin)
                query_feat_bin = nn.AdaptiveAvgPool2d(bin)(query_feat)
            else:
                bin = tmp_bin
                query_feat_bin = self.avgpool_list[idx](query_feat)
            supp_feat_bin = supp_feat.expand(-1, -1, bin, bin)#将支撑向量扩展成相应空间尺寸（bin bin）二维空间向量
            corr_mask_bin = F.interpolate(corr_query_mask, size=(bin, bin), mode='bilinear', align_corners=True)
            merge_feat_bin = torch.cat([query_feat_bin, supp_feat_bin, corr_mask_bin], 1)#拼接query 支撑向量 Prior mask
            merge_feat_bin = self.init_merge[idx](merge_feat_bin)#拼接完后 利用一个11卷积 进行初始变换
            #inter-source enrichment 以上
            #以下  inter-scale interaction
            if idx >= 1:#意味着 需要取金字塔上一尺度信息 进行Refine 见论文图4 中M 模块的引线图
                pre_feat_bin = pyramid_feat_list[idx-1].clone()
                pre_feat_bin = F.interpolate(pre_feat_bin, size=(bin, bin), mode='bilinear', align_corners=True)
                rec_feat_bin = torch.cat([merge_feat_bin, pre_feat_bin], 1)
                merge_feat_bin = self.alpha_conv[idx-1](rec_feat_bin) + merge_feat_bin  

            merge_feat_bin = self.beta_conv[idx](merge_feat_bin) + merge_feat_bin   
            inner_out_bin = self.inner_cls[idx](merge_feat_bin)
            merge_feat_bin = F.interpolate(merge_feat_bin, size=(query_feat.size(2), query_feat.size(3)), mode='bilinear', align_corners=True)
            pyramid_feat_list.append(merge_feat_bin)#将金字塔每层特征 存到一个列表中，方便后续传到更高层次处理
            out_list.append(inner_out_bin)
                 
        query_feat = torch.cat(pyramid_feat_list, 1)
        query_feat = self.res1(query_feat)
        query_feat = self.res2(query_feat) + query_feat           
        out = self.cls(query_feat)
        

        #   Output Part
        if self.zoom_factor != 1:
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            main_loss = self.criterion(out, y.long())#根据特征变成一个二分类问题了
            aux_loss = torch.zeros_like(main_loss).cuda()    

            for idx_k in range(len(out_list)):    
                inner_out = out_list[idx_k]
                inner_out = F.interpolate(inner_out, size=(h, w), mode='bilinear', align_corners=True)
                aux_loss = aux_loss + self.criterion(inner_out, y.long())   
            aux_loss = aux_loss / len(out_list)
            return out.max(1)[1], main_loss, aux_loss
        else:
            return out


class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
               # nn.GroupNorm(8,reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)


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

####################################################################################################
class FEWDomain(nn.Module):
    def __init__(self, layers=101, classes=2, zoom_factor=8, \
        criterion=nn.CrossEntropyLoss(ignore_index=255), BatchNorm=nn.BatchNorm2d, \
        pretrained=True, sync_bn=True, shot=1, ppm_scales=[60, 30, 15, 8], vgg=True):
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
           # self.aspp = nn.Conv2d(2048, 19, kernel_size=1, padding=0)#ASPP(2048,19, [6, 12, 18, 24])

        #reduce_dim = 256*2
       # feat_dim =512


       ################yzg vgg -frm based 
        if self.vgg:
            fea_dim = 512 + 256
        else:
            fea_dim = 2048 #+ 512       
        self.ppm = PPM(fea_dim,int(fea_dim/4),[1,2,3,6]) 
        reduce_dim = 1024
          

        self.down= nn.Sequential(
            nn.Conv2d(fea_dim*2, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(reduce_dim),
            nn.ReLU(inplace=True)                              
        )
 
       
        self.projector = nn.Sequential(
             nn.Conv2d(reduce_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
             nn.BatchNorm2d(reduce_dim),
             nn.ReLU(inplace=True),
            
             nn.Conv2d(reduce_dim, reduce_dim, kernel_size=1, padding=0, bias=False), 
             nn.BatchNorm2d(reduce_dim),
             nn.ReLU(inplace=True), #uncomment when using few-shot for fully supervised training
             #nn.Conv2d(reduce_dim, 19, kernel_size=1, padding=0)   #souonly
             nn.Conv2d(reduce_dim, reduce_dim, kernel_size=1, padding=0, bias=False)
            
        )  
  
      
        

   ###########forward for sourceonly or targetonly
    # def forward(self, x,  y=None):
    #     x_size = x.size()
    #     #print(x.shape)
    #     #assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
    #     #h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
    #     #w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)
    #     h,w = x_size[2],x_size[3]
       

     
    #     #   Query Feature
    #     #with torch.no_grad():
    #     query_feat_0 = self.layer0(x)
    #     query_feat_1 = self.layer1(query_feat_0)
    #     query_feat_2 = self.layer2(query_feat_1)
    #     query_feat_3 = self.layer3(query_feat_2)  
    #     query_feat_4 = self.layer4(query_feat_3)#高阶语义特征用于后面Prior mask generation
    #    # if self.vgg:
    #    #         query_feat_4 = F.interpolate(query_feat_4, size=(query_feat_2.size(2),query_feat_2.size(3)), mode='bilinear', align_corners=True)
    #     #out = self.deeplabv1head(query_feat_4)



    #     #out = self.aspp(query_feat_4 )
    #     #yzg vgg-frm
    #     query_feat = query_feat_4 #torch.cat([query_feat_4, query_feat_2], 1)#网络中间层特征用于预测和分类
    #     query_feat = self.ppm(query_feat)
    #     query_feat = self.down(query_feat)#通道降维
        
    #     out = self.projector(query_feat)   
    #    #yzg vgg-frm

    #     #print(out.shape)   
    #     #   Output Part
    #     if self.zoom_factor != 1:
    #         out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)

    #     if self.training:
    #         #classififeature = torch.cat([supp_feat, query_feat],1)
    #         #binary_logits = self.cls(classififeature)
            
    #         main_loss = self.criterion(out, y.long())#self.contrastloss(supp_feat, query_feat,s_y,y) #
    #         #auloss = self.prototype_contrast(prototype_tensor,q_prototype_tensor,disappear_label)
            
    #         return out.max(1)[1], main_loss#+0.05*auloss#out.max(1)[1], main_loss
    #     else:
     
    #         return out   


    


###########support and query merged  for training
    def forward(self, x, s_x=torch.FloatTensor(1,1,3,1025,1025).cuda(), s_y=torch.FloatTensor(1,1,1025,1025).cuda(), train_label_list=None, y=None, epoch=None):
        x_size = x.size()
        s_x =s_x.squeeze(0)
        temp_x = torch.cat([x,s_x],dim=0)
        #print(temp_x.shape)
        #assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        #h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        #w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)
        h,w = x_size[2],x_size[3]
        n_classes = len(train_label_list)
        #print(x_size,s_x.shape,s_y.shape,y.shape)


        #   Support Feature     
        prototype_list = []
        prototype_num_eachclass = [j-j for j in range(n_classes)]#不同shot图像 可能为同一类型贡献多个Prototype
      
            #print('self.shot',self.shot)
            #with torch.no_grad():
        feat_0 = self.layer0(temp_x)
        feat_1 = self.layer1(feat_0)
        feat_2 = self.layer2(feat_1)
        feat_3 = self.layer3(feat_2)
                #print('supp_feat_3.shape',supp_feat_3.shape)
                #mask = F.interpolate(mask, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear', align_corners=True)
                #print(np.unique(mask.cpu().numpy())) [0,1]
        feat_4 = self.layer4(feat_3)
                #final_supp_list.append(supp_feat_4)#高阶语义特征用于后面Prior mask generation
            #if self.vgg:
                   # supp_feat_4 = F.interpolate(supp_feat_4, size=(supp_feat_2.size(2),supp_feat_2.size(3)), mode='bilinear', align_corners=True)
            #supp_feat = self.deeplabv1head(supp_feat_4)


        feat = feat_4#torch.cat([supp_feat_4, supp_feat_2], 1)
        feat = self.ppm(feat)
        feat = self.down(feat)
            # #print('supp_feat_3.shape',supp_feat_3.shape)
            # #supp_feat = F.normalize(supp_feat,dim=1)
         #if self.training:
        feat = self.projector(feat)

        supp_feat = feat[1,:,:,:].unsqueeze(0)
        #print(supp_feat.shape)  
        for k, c in enumerate(train_label_list):#range(n_classes):
              #print('ccccccccccccccccccc',c,train_label_list)
              #print(k,c)
              mask = (s_y[:,0,:,:] == c.cuda()).float().unsqueeze(1)
              #print(len(train_label_list), np.unique(mask.cpu().numpy()),np.unique(s_y[:,shot,:,:].cpu().numpy()))
              #print('mask.shape,supp_feat.shape',mask.shape,supp_feat.shape)
              #mask_list.append(mask)
              mask = F.interpolate(mask, size=(supp_feat.size(2), supp_feat.size(3)), mode='bilinear', align_corners=True)
              #print('mask.shape after interpolation',np.unique(mask.cpu().numpy()))# 降采样后 MASK中有些小目标可能干没了
              prototype = Weighted_GAP(supp_feat, mask)#get protype for each class in train_label_list
              #print(prototype.shape) #torch.Size([1, 256, 1, 1])
              prototype = torch.squeeze(prototype)#.unsqueeze(0)  #256
            
              prototype_list.append(prototype)#n_clasess*256

              if 1 in np.unique(mask.cpu().numpy()): #in general, an image may not contribute prototype to all the classes
                prototype_num_eachclass[k] = prototype_num_eachclass[k]+1


        
        
        #print(prototype_tensor.shape)
        

        #prototype_tensor = torch.load("support_tensor.pt")
                  
        #   Query Feature
        query_feat =  feat[0,:,:,:].unsqueeze(0) 
        dist_s2q = [self.calDist(query_feat, prototype) for prototype in  prototype_list]
        dist_s2s =  [self.calDist(supp_feat, prototype) for prototype in  prototype_list]
        #if dist is None:
        
        out_q= torch.stack(dist_s2q, dim=1)  # 
        out_s = torch.stack(dist_s2s, dim=1)
       
              

        #   Output Part
        if self.zoom_factor != 1:
            out_q = F.interpolate(out_q, size=(h, w), mode='bilinear', align_corners=True)
            out_s= F.interpolate(out_s, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            #classififeature = torch.cat([supp_feat, query_feat],1)
            #binary_logits = self.cls(classififeature)
            
            main_loss = self.criterion(out_q, y.long())#self.contrastloss(supp_feat, query_feat,s_y,y) #
            main_loss2 = self.criterion(out_s, s_y.long().squeeze(1))
            # if prototype_tensor.size(0) >5 and epoch > 10:
            #    auloss = self.prototype_contrast(prototype_tensor,None,  None)
            # else:
            #    auloss = 0
            #if epoch>5:
            loss = main_loss+0.2* main_loss2
           # else:
            #    loss = main_loss#+0.2* main_loss2
            #print('outoutoutoutoutouuuuuuuuuuuuuuuuuuuuuuuuuuuu', out_q.shape, y.long().shape,s_y.long().squeeze(1).shape)
            return out_q.max(1)[1], loss#out.max(1)[1], main_loss
        else:
     
            return out_q


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



    def prototype_contrast(self, sp, qp=None,  invalidlabel=None):
       if qp is None:
          qp = sp
       sp = F.normalize(sp,dim=1)
       #qp = sp#F.normalize(qp,dim=1)
       doc_simi = torch.div(torch.matmul(sp,sp.t()),0.07)
       gt = torch.eye(doc_simi.shape[0]).cuda()
       
       
       logits_at_eachsample = F.softmax(doc_simi,dim=1)
       logits_at_eachsample = torch.log (logits_at_eachsample)

       logits_at_positive_active=torch.mul(logits_at_eachsample, gt)
       
       logits_at_positive_active_sum = torch.sum(logits_at_positive_active,dim=1) 
       if  invalidlabel:
            for i in invalidlabel:
                    logits_at_positive_active_sum[i]=0
       #print(gt.shape,logits_at_positive_active.shape, logits_at_positive_active_sum.shape)
       loss = torch.mean(-logits_at_positive_active_sum,dim=0)
     
       return loss
        

    def contrastloss(self, supp_feat, query_feat, s_label=None, q_label=None,binary_logits=None):
    #print('supp_label_ shape:',s_label.shape)
    #print('query_label_ shape:',q_label.shape)

       s_label = F.interpolate(s_label.float(), size=(supp_feat.size(2), supp_feat.size(3)), mode='nearest')
       q_label = F.interpolate(q_label.float().unsqueeze(1), size=(query_feat.size(2), query_feat.size(3)), mode='nearest')

       s_label = s_label.view(-1).unsqueeze(0)+1#1*HW +1 is used to transform label 0 to 1, so that the following reciprocal operation do not output NAN
       q_label = q_label.view(-1).unsqueeze(1)+1#HW*1
       positive_sample_binary_mat = torch.matmul(q_label,s_label)

       pixelind_invalid = np.where((positive_sample_binary_mat.cpu().numpy()) == 256*256)
    #print(len(pixelind_invalid[0]))
       s_label = torch.reciprocal(s_label)

       positive_sample_binary_mat = torch.matmul(q_label,s_label)#HW*hw

       pixelind = np.where((positive_sample_binary_mat.cpu().numpy()) !=1)# 1 means pixel in query images and support images have the same class
    #print(len(pixelind[0]))
    #pixelind = np.where((positive_sample_binary_mat.cpu().numpy()) ==1)
    #print(len(pixelind[0]))
       positive_sample_binary_mat[pixelind[0],pixelind[1]] = 0
       positive_sample_binary_mat[pixelind_invalid[0],pixelind_invalid[1]] = 0

       positive_num = torch.sum(positive_sample_binary_mat,dim=1) + 0.00001 # obtain the numbers of positive pixels in support image of each query pixel 

    #pixelind = np.where((positive_sample_binary_mat.cpu().numpy()) ==0)
    #print(len(pixelind[0]))
       positive_sample_binary_mat.unsqueeze(0)# B*HW*HW
    #print(positive_sample_binary_mat.shape,positive_num.shape)
    #print('supp_label_ shape:',s_label.shape)
    #print('query_label_ shape:',q_label.shape)
       assert (supp_feat.shape == query_feat.shape)
    #print('supp_feat_ shape:',supp_feat.shape)


    #classification loss

       #auloss = self.criterion(binary_logits,positive_sample_binary_mat)



    # contrast loss computation

       supp_feat = supp_feat.view(supp_feat.shape[0],supp_feat.shape[1],-1)#B *c*HW

       query_feat = query_feat.view(supp_feat.shape[0],supp_feat.shape[1],-1).permute(0,2,1)#B *HW*c
    #print('query_feat_ shape:',query_feat.shape)


    #we computed logits of all sample pairs first no matter whether they have the same label or not
       dotproc_similarity = torch.div(torch.matmul(query_feat,supp_feat),0.1)# B*HW_q*HW_s  Zi*Zp/temprature=0.07
    #print(np.unique(dotproc_similarity.detach().cpu().numpy()),torch.max(dotproc_similarity))
    #logits_max,_ = torch.max(dotproc_similarity,dim=2,keepdim= True)
    #dotproc_similarity = dotproc_similarity - logits_max.detach()
       exp_dotproc_similarity = torch.exp(dotproc_similarity)# exp(Zi*Zp/temprature)

       logits_at_eachsample = dotproc_similarity - torch.log(exp_dotproc_similarity.sum(2,keepdim=True))
    #dotproc_similarity_sum = torch.sum(dotproc_similarity,dim=2)#sum_a^HW exp(Zi*Za/t)  (B*HW)
    #print(np.unique(dotproc_similarity_sum.detach().cpu().numpy()))
    #logits_at_eachsample =  torch.div(dotproc_similarity,dotproc_similarity_sum.t())#    exp(Zi*Zp/temprature)/  sum_a^HWHW exp(Zi*Za/t) before log
    #logits_at_eachsample = torch.log (logits_at_eachsample)

    #
       logits_at_positive_active=torch.mul(logits_at_eachsample, positive_sample_binary_mat)# only keep logits of positive samples
    
       logits_at_positive_active_sum = torch.sum(logits_at_positive_active,dim=2)
       logits_final = -torch.div(logits_at_positive_active_sum,positive_num)
    #print(len(np.where(logits_final.detach().cpu().numpy()==0)[0]))
    #print(logits_at_positive_active_sum.shape, logits_final.shape)

       loss = torch.sum(logits_final,dim=1)/(logits_final.shape[1]-len(np.where(logits_final.detach().cpu().numpy()==0)[0]))

       
    #print(loss)
    #print('similarity_ shape:',dotproc_similarity.shape,logits_at_positive_active.shape,logits_at_eachsample.shape)


    #time.sleep(30)
       return loss

       

       




###################################################################

class FEWDomainComPro(nn.Module):
    def __init__(self, layers=101, classes=2, zoom_factor=8, \
        criterion=nn.CrossEntropyLoss(ignore_index=255), BatchNorm=nn.BatchNorm2d, \
        pretrained=True, sync_bn=True, shot=5, ppm_scales=[60, 30, 15, 8], vgg=True):
        super(FEWDomainComPro, self).__init__()
        assert layers in [50, 101, 152]
        print(ppm_scales)
        assert classes > 1
        from torch.nn import BatchNorm2d as BatchNorm        
        self.zoom_factor = zoom_factor
        self.criterion = criterion
        self.shot = shot
        #self.ppm_scales = ppm_scales
        self.vgg = vgg
        self.validlabel= None#[7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33]
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
        self.ppm = PPM(fea_dim,int(fea_dim/4),[1,2,3,6]) 
        reduce_dim = 256*4



        self.projector = nn.Sequential(
              nn.Conv2d(reduce_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
              nn.BatchNorm2d(reduce_dim),
              nn.ReLU(inplace=True),
            
              nn.Conv2d(reduce_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
              nn.BatchNorm2d(reduce_dim),
              nn.ReLU(inplace=True), #uncomment when using few-shot for fully supervised training
           
            #nn.Conv2d(reduce_dim, 19, kernel_size=1, padding=0), 
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=1, padding=0, bias=False),         
        )      

        self.down= nn.Sequential(
            nn.Conv2d(fea_dim*2, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(reduce_dim),
            nn.ReLU(inplace=True)
                           
        )
                           
         


    def forward(self, s_x=None, s_y=None, shot=5,prototype_num_eachclass=torch.zeros(19).cuda(),train_label_list=None, image_name=None,valid_image_list=[]):
        
        #print(x.shape)
        #assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        #h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        #w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)
        
        n_classes = len(train_label_list)
        #print(x_size,s_x.shape,s_y.shape,y.shape)


        #   Support Feature     
        prototype_list = []
        #prototype_num_eachclass = [j-j for j in range(n_classes)]#不同shot图像 可能为同一类型贡献多个Prototype
        final_supp_list = []
        mask_list = []
        disappear_label = []
        #print(s_x.shape)
        
            #print('self.shot',self.shot)
        with torch.no_grad():
                supp_feat_0 = self.layer0(s_x[:,:,:,:])
                supp_feat_1 = self.layer1(supp_feat_0)
                supp_feat_2 = self.layer2(supp_feat_1)
                supp_feat_3 = self.layer3(supp_feat_2)
                #print('supp_feat_3.shape',supp_feat_3.shape)
                #mask = F.interpolate(mask, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear', align_corners=True)
                #print(np.unique(mask.cpu().numpy())) [0,1]
                supp_feat_4 = self.layer4(supp_feat_3)
                #final_supp_list.append(supp_feat_4)#高阶语义特征用于后面Prior mask generation
                # if self.vgg:
                #     supp_feat_4 = F.interpolate(supp_feat_4, size=(supp_feat_2.size(2),supp_feat_2.size(3)), mode='bilinear', align_corners=True)
        #supp_feat = self.deeplabv1head(supp_feat_4)  
        supp_feat = supp_feat_4 # torch.cat([supp_feat_4, supp_feat_2], 1)
        supp_feat = self.ppm(supp_feat)
        supp_feat = self.down(supp_feat)
        # if  True:#self.training:
        #    print('is training')
        supp_feat  = self.projector(supp_feat )
        #print('supp_feat_3.shape',supp_feat.shape)
            #supp_feat = F.normalize(supp_feat,dim=1)
        print('........................',prototype_num_eachclass)    
           
        for k, c in enumerate(train_label_list):#range(n_classes):
              #print('ccccccccccccccccccc',c,train_label_list)
              #print(k,c)
              if self.validlabel:
                 mask = (s_y[:,:,:] == self.validlabel[k]).float().unsqueeze(1)
              else:
                 mask = (s_y[:,:,:] == c.cuda()).float().unsqueeze(1)
              #print(len(train_label_list), np.unique(mask.cpu().numpy()),np.unique(s_y[:,shot,:,:].cpu().numpy()))
              #print('mask.shape,supp_feat.shape',mask.shape,supp_feat.shape)
              #mask_list.append(mask)
              mask = F.interpolate(mask, size=(supp_feat.size(2), supp_feat.size(3)), mode='bilinear', align_corners=True)
              #print('mask.shape after interpolation',np.unique(mask.cpu().numpy()))# 降采样后 MASK中有些小目标可能干没了
              prototype = Weighted_GAP(supp_feat, mask)#get protype for each class in train_label_list
              prototype = torch.squeeze(prototype)
              validpixel_k = np.where(mask.cpu().numpy()==1)
             
         
              if len(validpixel_k[0]>500) and prototype_num_eachclass[k]<shot:
                 if image_name[0] not in valid_image_list:
                     valid_image_list.append(image_name[0])
                 prototype_num_eachclass[k] = prototype_num_eachclass[k]+1
              else:    
             
                prototype = torch.zeros_like(prototype)
              prototype_list.append(prototype)#n_clasess*256
            #print(prototype_num_eachclass)

        #print(len(prototype_list))
        for k, c in enumerate(train_label_list):
            
            if  prototype_num_eachclass[k]==0: #由于下采样的存在，可能导致支撑图片某些目标的消失，从而导致支撑图片和查询图片某些共有标签的消失,需要将查询图片对应共有标签替换为无效标签
         #       pixelind_i = np.where((y.cpu().numpy()) == c.cpu().numpy())
        #        y[pixelind_i[0],pixelind_i[1],pixelind_i[2]] = 255
                #prototype_list.pop(k)
                #disappear_label.append(k)
                print('disapper', k)

        #print(len(prototype_list))
        #print(prototype_list[0])
        


        
        prototype_tensor = torch.stack(prototype_list,dim=0)
        #print(prototype_tensor.shape)
       # torch.save(prototype_tensor,'support_tensorcity0809.pt')

        #prototype_tensor = torch.load("support_tensor.pt")
                  

        print(valid_image_list)
        return prototype_num_eachclass, prototype_tensor,valid_image_list

  

