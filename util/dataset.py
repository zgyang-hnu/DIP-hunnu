import os
import os.path
import cv2
import numpy as np

import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import random
import time
import glob
from tqdm import tqdm
from PIL import Image

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']



def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(split=0, data_root=None, data_list=None, sub_list=None):    
    assert split in [0, 1, 2, 3, 10, 11, 999]
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))

    # Shaban uses these lines to remove small objects:
    # if util.change_coordinates(mask, 32.0, 0.0).sum() > 2:
    #    filtered_item.append(item)      
    # which means the mask will be downsampled to 1/32 of the original size and the valid area should be larger than 2, 
    # therefore the area in original size should be accordingly larger than 2 * 32 * 32    
    image_label_list = []  
    list_read = open(data_list).readlines()
    print("Processing data...".format(sub_list))
    sub_class_file_list = {}
    for sub_c in sub_list:
        sub_class_file_list[sub_c] = []

    for l_idx in tqdm(range(len(list_read))):
        line = list_read[l_idx]
        line = line.strip()
        line_split = line.split(' ')
        image_name = os.path.join(data_root, line_split[0])
        label_name = os.path.join(data_root, line_split[1])
        
        item = (image_name, label_name)
        label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)
        label_class = np.unique(label).tolist()

        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255)

        new_label_class = []       
        for c in label_class:
            if c in sub_list:
                tmp_label = np.zeros_like(label)
                target_pix = np.where(label == c)
                tmp_label[target_pix[0],target_pix[1]] = 1 
                if tmp_label.sum() >= 2 * 32 * 32:      
                    new_label_class.append(c)

        label_class = new_label_class    

        if len(label_class) > 0:
            image_label_list.append(item)
            for c in label_class:
                if c in sub_list:
                    sub_class_file_list[c].append(item)
                    
    print("Checking image&label pair {} list done! ".format(split))
    
    return image_label_list, sub_class_file_list
           #                 元素为列表的字典                 




class SemData(Dataset):
    def __init__(self, split=3, shot=1, data_root=None, data_list=None, transform=None, mode='train', use_coco=False, use_split_coco=False):
        assert mode in ['train', 'val', 'test']
        
        self.mode = mode
        self.split = split  
        self.shot = shot
        self.data_root = data_root   

        if not use_coco:#划分训练集和验证集
            self.class_list = list(range(1, 21)) #[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
            if self.split == 3: 
                self.sub_list = list(range(1, 16)) #[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
                self.sub_val_list = list(range(16, 21)) #[16,17,18,19,20]
            elif self.split == 2:
                self.sub_list = list(range(1, 11)) + list(range(16, 21)) #[1,2,3,4,5,6,7,8,9,10,16,17,18,19,20]
                self.sub_val_list = list(range(11, 16)) #[11,12,13,14,15]
            elif self.split == 1:
                self.sub_list = list(range(1, 6)) + list(range(11, 21)) #[1,2,3,4,5,11,12,13,14,15,16,17,18,19,20]
                self.sub_val_list = list(range(6, 11)) #[6,7,8,9,10]
            elif self.split == 0:
                self.sub_list = list(range(6, 21)) #[6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
                self.sub_val_list = list(range(1, 6)) #[1,2,3,4,5]

        else:
            if use_split_coco:
                print('INFO: using SPLIT COCO')
                self.class_list = list(range(1, 81))
                if self.split == 3:
                    self.sub_val_list = list(range(4, 81, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))                    
                elif self.split == 2:
                    self.sub_val_list = list(range(3, 80, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))    
                elif self.split == 1:
                    self.sub_val_list = list(range(2, 79, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))    
                elif self.split == 0:
                    self.sub_val_list = list(range(1, 78, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))    
            else:
                print('INFO: using COCO')
                self.class_list = list(range(1, 81))
                if self.split == 3:
                    self.sub_list = list(range(1, 61))
                    self.sub_val_list = list(range(61, 81))
                elif self.split == 2:
                    self.sub_list = list(range(1, 41)) + list(range(61, 81))
                    self.sub_val_list = list(range(41, 61))
                elif self.split == 1:
                    self.sub_list = list(range(1, 21)) + list(range(41, 81))
                    self.sub_val_list = list(range(21, 41))
                elif self.split == 0:
                    self.sub_list = list(range(21, 81)) 
                    self.sub_val_list = list(range(1, 21))    

        print('sub_list: ', self.sub_list)
        print('sub_val_list: ', self.sub_val_list)    

        if self.mode == 'train':
            self.data_list, self.sub_class_file_list = make_dataset(split, data_root, data_list, self.sub_list)
            assert len(self.sub_class_file_list.keys()) == len(self.sub_list)
        elif self.mode == 'val':
            self.data_list, self.sub_class_file_list = make_dataset(split, data_root, data_list, self.sub_val_list)
            assert len(self.sub_class_file_list.keys()) == len(self.sub_val_list) 
        self.transform = transform


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        label_class = []
        image_path, label_path = self.data_list[index]#get item 开始只获取 query  image
        #print(label_path)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        #plt.imshow(image)
        #plt.show()
       
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  

        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Query Image & label shape mismatch: " + image_path + " " + label_path + "\n"))          
        label_class = np.unique(label).tolist()
        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255) 
        new_label_class = []       
        for c in label_class:#对于query图片中的类别属性，如果类别在训练集中，看当前模式是不是训练，是则添加到
            if c in self.sub_val_list:
                if self.mode == 'val' or self.mode == 'test':
                    new_label_class.append(c)
            if c in self.sub_list:
                if self.mode == 'train':
                    new_label_class.append(c)
        label_class = new_label_class    
        assert len(label_class) > 0


        class_chosen = label_class[random.randint(1,len(label_class))-1]
        class_chosen = class_chosen
        target_pix = np.where(label == class_chosen)
        ignore_pix = np.where(label == 255)
        label[:,:] = 0#二值标签替换
        if target_pix[0].shape[0] > 0:
            label[target_pix[0],target_pix[1]] = 1 #二值标签替换
        label[ignore_pix[0],ignore_pix[1]] = 255           
        #plt.imshow(label)
        #plt.show()
        #time.sleep(10)

        file_class_chosen = self.sub_class_file_list[class_chosen]##根据Query image label定位到相应类别的图像列表
        num_file = len(file_class_chosen)#获取该类 的训练图像 数量

        support_image_path_list = []
        support_label_path_list = []
        support_idx_list = []
        for k in range(self.shot):
            support_idx = random.randint(1,num_file)-1#随机选取一张作支撑图片（循环shot次）
            support_image_path = image_path
            support_label_path = label_path
            while((support_image_path == image_path and support_label_path == label_path) or support_idx in support_idx_list):
                support_idx = random.randint(1,num_file)-1
                support_image_path, support_label_path = file_class_chosen[support_idx]      #选到和Query、已经选中的 不重复的为止          
            support_idx_list.append(support_idx)
            support_image_path_list.append(support_image_path)
            support_label_path_list.append(support_label_path)

        support_image_list = []
        support_label_list = []
        subcls_list = []
        for k in range(self.shot):  
            if self.mode == 'train':
                subcls_list.append(self.sub_list.index(class_chosen))
            else:
                subcls_list.append(self.sub_val_list.index(class_chosen))
            support_image_path = support_image_path_list[k]
            support_label_path = support_label_path_list[k] 
            support_image = cv2.imread(support_image_path, cv2.IMREAD_COLOR)      
            support_image = cv2.cvtColor(support_image, cv2.COLOR_BGR2RGB)
            support_image = np.float32(support_image)
            support_label = cv2.imread(support_label_path, cv2.IMREAD_GRAYSCALE)
            target_pix = np.where(support_label == class_chosen)
            ignore_pix = np.where(support_label == 255)
            support_label[:,:] = 0
            support_label[target_pix[0],target_pix[1]] = 1 
            support_label[ignore_pix[0],ignore_pix[1]] = 255
            if support_image.shape[0] != support_label.shape[0] or support_image.shape[1] != support_label.shape[1]:
                raise (RuntimeError("Support Image & label shape mismatch: " + support_image_path + " " + support_label_path + "\n"))            
            support_image_list.append(support_image)
            support_label_list.append(support_label)
        assert len(support_label_list) == self.shot and len(support_image_list) == self.shot                    
        
        raw_label = label.copy()
        if self.transform is not None:
            image, label = self.transform(image, label)#对query 进行变换
            #plt.imshow(np.uint8(image.permute(1,2,0).cpu().numpy()))
            #plt.show()
            #cv2.imwrite('translabel.png', label.cpu().numpy())
            #plt.show()
            #time.sleep(100)
            for k in range(self.shot):
                support_image_list[k], support_label_list[k] = self.transform(support_image_list[k], support_label_list[k])

        s_xs = support_image_list
        s_ys = support_label_list
        s_x = s_xs[0].unsqueeze(0)
        for i in range(1, self.shot):
            s_x = torch.cat([s_xs[i].unsqueeze(0), s_x], 0)
        s_y = s_ys[0].unsqueeze(0)
        for i in range(1, self.shot):
            s_y = torch.cat([s_ys[i].unsqueeze(0), s_y], 0)

        if self.mode == 'train':
            return image, label, s_x, s_y, subcls_list
        else:
            return image, label, s_x, s_y, subcls_list, raw_label



class GTA5(Dataset):
      
      def __init__(self, train_path,  train_size=None, transform=None, use_val=False):
          super(GTA5, self).__init__()
          self.train_path = train_path
          self.train_size = train_size
          self.transform = transform
          self.validlabel= [7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33]
          self.train_list = self.get_train_list(train_path)
          #self.transform = transforms.Compose(transforms_)
          self.dataset_size = len(self.train_list)

      def __len__(self):
         return self.dataset_size


      def __getitem__(self, index):

          q_label_list =[]
          index1 = index
          while len(list (set(q_label_list) & set(self.validlabel))) <= 1:#some images in GTA5 contains only one valid label, we ignore these images during training
            q_image_path = self.train_list[index1]['image_path']
            q_label_path = self.train_list[index1]['label_path']
            q_image_name = self.train_list[index1]['image_name']
            q_image = cv2.imread(q_image_path, cv2.IMREAD_COLOR) 
            q_image = cv2.cvtColor(q_image, cv2.COLOR_BGR2RGB)  

            q_image = np.float32(q_image)
            #q_label = cv2.imread(q_label_path, cv2.IMREAD_GRAYSCALE)  
            q_label = np.array(Image.open(q_label_path))
             
            q_label_list = np.unique(q_label).tolist()
            #print(q_label_list)
            index1 = index1 +random.randint(1,5)


          if q_image.shape[0] != q_label.shape[0] or q_image.shape[1] != q_label.shape[1]:
            raise (RuntimeError("Query Image & label shape mismatch: " + q_image_path + " " + q_label_path + "\n"))
          
          #perform data augmentation on query images
          if self.transform:
              q_image,q_label = self.transform(q_image,q_label)
          

          train_label_list= []
          while len(train_label_list) <= 1: #query image and support image must contain at least 2 same labels
            s_image_idx = random.randint(1,self.dataset_size)-1
            s_image_path = self.train_list[s_image_idx]['image_path']
            s_label_path = self.train_list[s_image_idx ]['label_path']

            while(s_image_path == q_image_path):#support image and query image must be different 
               s_image_idx = random.randint(1,self.dataset_size)-1
               s_image_path = self.train_list[s_image_idx ]['image_path']
               s_label_path = self.train_list[s_image_idx ]['label_path']
            #end
          #print(q_image_path)
          #print(s_image_path)

            s_image = cv2.imread(s_image_path, cv2.IMREAD_COLOR) 
            s_image = cv2.cvtColor(s_image, cv2.COLOR_BGR2RGB)  

            s_image = np.float32(s_image)
            s_label = np.array(Image.open(s_label_path))#cv2.imread(s_label_path, cv2.IMREAD_GRAYSCALE)
            if self.transform:
              s_image,s_label = self.transform(s_image,s_label)
          
            
            s_label_list = np.unique(s_label).tolist()
            qscommon_label = list(set(q_label_list) & set(s_label_list))
                      
            #if 0 in train_label:
            #  train_label.remove(0)#0 is the ignored label
            #print(train_label)

            train_label_list = list (set(qscommon_label) & set(self.validlabel))#we only consider label that contained in 19 labels of cityscape dataset
            #print(len(train_label_list))
            if (len(train_label_list)<=1):
                 print(q_image_path)
                 print(s_image_path)
         # end of first while


          # labels contained both in support image and query image will be used for few-shot learning, other labels are set to invalid 
          for i in s_label_list:
             if i not in train_label_list:
                pixelind_i = np.where(s_label == i)
                s_label[pixelind_i[0],pixelind_i[1]] = 255  

          
          for i in q_label_list:
             if i not in train_label_list:
                pixelind_i = np.where(q_label == i)
                q_label[pixelind_i[0],pixelind_i[1]] = 255
          #print(train_label_list)
          #According to the label numbers (n) in train_label_list, formulate the few shot learning into a n-class classification problem
          for i in train_label_list:
            #print(i,train_label_list.index(i))
            pixelind_i = np.where(s_label==i)
            s_label[pixelind_i[0],pixelind_i[1]] = train_label_list.index(i) 

            pixelind_i = np.where(q_label==i)
            q_label[pixelind_i[0],pixelind_i[1]] = train_label_list.index(i) 

          #print(q_label_path,q_label_list)
          #print(s_label_path,s_label_list)
         
          #im=Image.frombytes('L', (s_label.shape[1],s_label.shape[0]),s_label)
          #im.save('slabelsaved.png','png')
          train_label_list = [i for i in range(len(train_label_list))]
          
          return q_image,q_label,s_image.unsqueeze(0),s_label.unsqueeze(0),train_label_list

     


      def get_train_list(self, train_path):

         image_dir = train_path[0]
         label_dir = train_path[1]



         if not os.path.exists(label_dir):
              raise IOError("No such training direcotry exist!")
         match_str=os.path.join(label_dir,  "*.png")
         #print(match_str)
         image_label_path_list=[]
         image_record_list=[]
         image_label_path_list.extend(glob.glob(match_str))
         #print(len(image_label_path_list))
    
         for f in image_label_path_list:                   
             image=os.path.splitext(f.split('/')[-1])[0]
        
             image_name=image
            
             image_path=os.path.join(image_dir, image_name+'.png')
        
             if not os.path.exists(image_path):
                print("Image %s is not exist %s" % (image_name,image_dir))
                raise IOError("Please Check")

             else:
                image_record={'label_path':f,'image_path':image_path,'image_name':image_name}
                                                                                                
                image_record_list.append(image_record)


         #print(image_record_list)
         print (len(image_record_list))
         
         return image_record_list

#['tubingen_000114_000019', 'tubingen_000122_000019', 'tubingen_000055_000019', 'tubingen_000021_000019', 'tubingen_000091_000019', 'tubingen_000076_000019', 'hanover_000000_039021']
#fewshot_name_list = ['tubingen_000114_000019', 'tubingen_000122_000019', 'tubingen_000055_000019', 'tubingen_000021_000019', 'tubingen_000091_000019', 'tubingen_000076_000019', 'hanover_000000_039021']
#fewshot_name_list =['tubingen_000114_000019', 'tubingen_000122_000019', 'tubingen_000055_000019', 'tubingen_000105_000019', 'tubingen_000116_000019', 'tubingen_000023_000019', 'tubingen_000100_000019', 'tubingen_000017_000019', 'tubingen_000058_000019', 'tubingen_000021_000019', 'tubingen_000041_000019', 'tubingen_000117_000019', 'tubingen_000035_000019', 'tubingen_000030_000019', 'tubingen_000133_000019', 'tubingen_000043_000019', 'tubingen_000091_000019', 'tubingen_000059_000019', 'tubingen_000134_000019', 'tubingen_000051_000019', 'tubingen_000001_000019', 'tubingen_000010_000019', 'tubingen_000033_000019', 'tubingen_000076_000019', 'tubingen_000064_000019', 'tubingen_000131_000019', 'tubingen_000065_000019', 'hanover_000000_012675', 'hanover_000000_039021', 'hanover_000000_049465', 'hanover_000000_049269', 'hanover_000000_049005', 'hanover_000000_007342']
#fewshot_name_list =['krefeld_000000_032390', 'krefeld_000000_018747', 'krefeld_000000_032845', 'krefeld_000000_009926', 'krefeld_000000_015687', 'krefeld_000000_024362', 'krefeld_000000_010329', 'krefeld_000000_020033', 'krefeld_000000_005503', 'krefeld_000000_014673', 'krefeld_000000_003707', 'krefeld_000000_011655', 'krefeld_000000_029050', 'krefeld_000000_000926', 'krefeld_000000_031257', 'krefeld_000000_004447', 'krefeld_000000_023510', 'krefeld_000000_030560', 'krefeld_000000_017489', 'krefeld_000000_036299', 'krefeld_000000_013257', 'krefeld_000000_035124', 'krefeld_000000_012353', 'krefeld_000000_024921', 'krefeld_000000_024604', 'krefeld_000000_012505', 'krefeld_000000_013139']
fewshot_name_list =['b08cff3e-21c6ff47', '6bb5312d-96366827', '71f3dd79-bc3d6197', '2c783792-bf36f7fd', '0c113798-00000000', '5ffe9db5-bcf70001', '392d943b-00000000', '4318b758-b3f85c2a', '00e9be89-00001325', '7a4e7466-bb01f12a', '8dacce2d-7310ed9a', '6ddc9481-fe6982d2', '7ce0f7a8-8cb3ed4f', 'bd8d420d-ec34c634', '288b8fc3-00000000', '2bf177ce-b7d8d5ff', '3d1a6305-3980e3bc', '2a052d49-00000000', '047e715f-3e4790fd', '196fd10e-21e3fbdf', 'bae92945-99223ba6', '00e9be89-00001070', '3ffb33b8-98640000', '541e537b-91d6ed10', '87154a3d-65d8e0c4', '381d0a50-01ecc550', '06e583b2-39c5a69d', '00e9be89-00001795', '48ba7adf-a70c7480', '75883d91-b779423c', '25abdc75-5a7b6af1', '1d5ab341-50d3ec06', 'c695fbd3-0a91a52f', 'bbeefd23-dbf4d60e', '212289f5-732658d4', '0bcc752f-fd302777', '11d0db2b-d0b29fec', '065a3868-00000000', '73fd9d6f-ba1ccb6d']
class CityGTA5(Dataset):
      
      def __init__(self, train_path,  support_path, train_size=None, transform=None, use_val=False):
          super(CityGTA5, self).__init__()
          self.train_path = train_path
          self.train_size = train_size
          self.transform = transform
          self.validlabel= [7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33]
          self.train_list = self.get_train_list(train_path)
          self.support_list = self.get_support_list(support_path)
          #self.transform = transforms.Compose(transforms_)
          self.dataset_size = len(self.train_list)
          self.support_dataset_size = len (self.support_list )

      def __len__(self):
         return self.dataset_size


      def __getitem__(self, index):

          q_label_list =[]
          index1 = index
          while len(list (set(q_label_list) & set(self.validlabel))) <= 1:#some images in GTA5 contains only one valid label, we ignore these images during training
                q_image_path = self.train_list[index1]['image_path']
                q_label_path = self.train_list[index1]['label_path']
                q_image_name = self.train_list[index1]['image_name']
                q_image = cv2.imread(q_image_path, cv2.IMREAD_COLOR) 
                q_image = cv2.cvtColor(q_image, cv2.COLOR_BGR2RGB)  

                q_label = np.array(Image.open(q_label_path))
                #q_label = cv2.imread(q_label_path, cv2.IMREAD_GRAYSCALE)
                #print(q_image.shape,q_label.shape)
                if q_image.shape[0] != q_label.shape[0] or q_image.shape[1] != q_label.shape[1]:
                    #print('aaaaaaaaaaaaaaaaaa')
                    q_label= cv2.resize(q_label, dsize=(q_image.shape[1],q_image.shape[0]) , interpolation=cv2.INTER_NEAREST)
          
                q_image = np.float32(q_image)
                #  
               
                #print(q_image.shape,q_label.shape)
                q_label_list = np.unique(q_label).tolist()
                #print(q_label_list)
                index1 = index1 +random.randint(1,5)


          if q_image.shape[0] != q_label.shape[0] or q_image.shape[1] != q_label.shape[1]:

                raise (RuntimeError("Query Image & label shape mismatch: " + q_image_path + " " + q_label_path + "\n"))
          
          #perform data augmentation on query images
          if self.transform:
                q_image,q_label = self.transform(q_image,q_label)
          #q_label_temp = q_label.cpu().numpy()
          #q_label_temp = cv2.resize(q_label_temp, dsize=(int(q_label.shape[0]/8), int(q_label.shape[1]/8)),interpolation=cv2.INTER_NEAREST) 
          q_label_list = np.unique(q_label).tolist()#最后计算原型的特征会进行8倍下采样 某些标签可能会消失

          train_label_list= []
          while len(train_label_list) <= 1: #query image and support image must contain at least 2 same labels
                s_image_idx = random.randint(1,self.support_dataset_size)-1
                s_image_path = self.support_list[s_image_idx]['image_path']
                s_label_path = self.support_list[s_image_idx ]['label_path']

                # while(s_image_path == q_image_path):#support image and query image must be different 
                #    s_image_idx = random.randint(1,self.dataset_size)-1
                #    s_image_path = self.train_list[s_image_idx ]['image_path']
                #    s_label_path = self.train_list[s_image_idx ]['label_path']
                #end
               #print(q_image_path)
               #print(s_image_path)

                s_image = cv2.imread(s_image_path, cv2.IMREAD_COLOR) 
                s_image = cv2.cvtColor(s_image, cv2.COLOR_BGR2RGB)  

                s_image = np.float32(s_image)
                s_label = np.array(Image.open(s_label_path))#cv2.imread(s_label_path, cv2.IMREAD_GRAYSCALE)
                if self.transform:
                    s_image,s_label = self.transform(s_image,s_label)
            
                # s_label_temp = s_label
                # s_label_temp = F.interpolate( s_label_temp.unsqueeze(0).unsqueeze(0).float, size=(129, 129), mode='nearest')
                # print(s_label_temp.shape) 
                # print(np.unique(s_label_temp).tolist())
                s_label_list = np.unique(s_label).tolist()
                # print(s_label_list)
                qscommon_label = list(set(q_label_list) & set(s_label_list))
                        
                #if 0 in train_label:
                #  train_label.remove(0)#0 is the ignored label
                #print(train_label)

                train_label_list = list (set(qscommon_label) & set(self.validlabel))#we only consider label that contained in 19 labels of cityscape dataset
                #print(len(train_label_list))

           
                train_label_list_temp = train_label_list
                #poptimes = 0
                for k, c in enumerate(train_label_list):#range(n_classes):
                #print('ccccccccccccccccccc',c,train_label_list)
                #print(k,c)
                    mask = (s_label[:,:] == c).float().unsqueeze(0).unsqueeze(0)

                    mask = F.interpolate(mask, size=(129,129), mode='bilinear', align_corners=True)
                    #print('mask.shape after interpolation',np.unique(mask.cpu().numpy()))# 降采样后 MASK中有些小目标可能干没了
                    #qmask = (q_label[:,:] == c).float().unsqueeze(0).unsqueeze(0)
                    #qmask = F.interpolate(qmask, size=(129,129), mode='bilinear', align_corners=True)    

                    if 1 in np.unique(mask.cpu().numpy()) : #in general, an image may not contribute prototype to all the classes
                        #if 1 in np.unique(qmask.cpu().numpy()) :
                            pass
                        #else:
                            #print('in data loading, query disapper', k,c)
                         #   train_label_list_temp.remove(c)#pop(k-poptimes)
                            #poptimes = poptimes+1
                    else:
                         print('in data loading, support disapper', k,c)
                         train_label_list_temp.remove(c)#pop(k-poptimes)
                         #poptimes = poptimes+1
                    #print('end for ')
                    
          
                        
                train_label_list = train_label_list_temp
                if (len(train_label_list)<=1):
                    print(q_image_path)
                    print(s_image_path)
            # end of  while

            # labels contained both in support image and query image will be used for few-shot learning, other labels are set to invalid 
          for i in s_label_list:
                if i not in train_label_list:
                    pixelind_i = np.where(s_label == i)
                    s_label[pixelind_i[0],pixelind_i[1]] = 255  

          
          for i in q_label_list:
                if i not in train_label_list:
                    pixelind_i = np.where(q_label == i)
                    q_label[pixelind_i[0],pixelind_i[1]] = 255
          #print(train_label_list)
          #According to the label numbers (n) in train_label_list, formulate the few shot learning into a n-class classification problem
          for i in train_label_list:
                #print(i,train_label_list.index(i))
                pixelind_i = np.where(s_label==i)
                s_label[pixelind_i[0],pixelind_i[1]] = train_label_list.index(i) 

                pixelind_i = np.where(q_label==i)
                q_label[pixelind_i[0],pixelind_i[1]] = train_label_list.index(i) 

          #print(q_label_path,q_label_list)
          #print(s_label_path,s_label_list)
         
          #im=Image.frombytes('L', (s_label.shape[1],s_label.shape[0]),s_label)
          #im.save('slabelsaved.png','png')
          train_label_list = [i for i in range(len(train_label_list))]
          
          return q_image,q_label,s_image.unsqueeze(0),s_label.unsqueeze(0),train_label_list

     


      def get_train_list(self, train_path):

         image_dir = train_path[0]
         label_dir = train_path[1]



         if not os.path.exists(label_dir):
              raise IOError("No such training direcotry exist!")
         match_str=os.path.join(label_dir,  "*.png")
         #print(match_str)
         image_label_path_list=[]
         image_record_list=[]
         image_label_path_list.extend(glob.glob(match_str))
         #print(len(image_label_path_list))
    
         for f in image_label_path_list:                   
             image=os.path.splitext(f.split('/')[-1])[0]
        
             image_name=image
            
             image_path=os.path.join(image_dir, image_name+'.png')
        
             if not os.path.exists(image_path):
                print("Image %s is not exist %s" % (image_name,image_dir))
                raise IOError("Please Check")

             else:
                image_record={'label_path':f,'image_path':image_path,'image_name':image_name}
                                                                                                
                image_record_list.append(image_record)


         #print(image_record_list)
         print (len(image_record_list))
         
         return image_record_list


      def get_support_list(self, support_path):

            image_dir = support_path[0]
            label_dir = support_path[1]
            if not os.path.exists(label_dir):
                raise IOError("No such training direcotry exist!")

            image_record_list=[]
        

            
            for image_name in fewshot_name_list:                   
                
                cityname=image_name.split('_')
                cityname=cityname[0]
                #print(city)
                
                #print(image_name)
                label_path = os.path.join(label_dir, cityname, image_name+'_gtFine_labelIds.png')
                image_path=os.path.join(image_dir, cityname, image_name+'_leftImg8bit.png')
            
                if not os.path.exists(image_path):
                    #print("Image %s is not exist %s" % (image_name,image_dir))
                    raise IOError("Please Check")

                else:
                    image_record={'label_path':label_path,'image_path':image_path,'image_name':image_name}
                                                                                                    
                image_record_list.append(image_record)
           
            print ('the length of support images',len(image_record_list))
            return image_record_list

class CityGTA5NEW(Dataset):
      
      def __init__(self, train_path,  support_path, train_size=None, transform=None, use_val=False):
          super(CityGTA5NEW, self).__init__()
          self.train_path = train_path
          self.train_size = train_size
          self.transform = transform
          self.validlabel= [7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33]
          self.train_list = self.get_train_list(train_path)
          self.support_list = self.get_support_list(support_path)
          #self.transform = transforms.Compose(transforms_)
          self.dataset_size = len(self.train_list)
          self.support_dataset_size = len (self.support_list )

      def __len__(self):
         return self.dataset_size


      def __getitem__(self, index):

          q_label_list =[]
          index1 = index
          while len(list (set(q_label_list) & set(self.validlabel))) <= 1:#some images in GTA5 contains only one valid label, we ignore these images during training
                q_image_path = self.train_list[index1]['image_path']
                q_label_path = self.train_list[index1]['label_path']
                q_image_name = self.train_list[index1]['image_name']
                q_image = cv2.imread(q_image_path, cv2.IMREAD_COLOR) 
                q_image = cv2.cvtColor(q_image, cv2.COLOR_BGR2RGB)  

                q_label = np.array(Image.open(q_label_path))
                #q_label = cv2.imread(q_label_path, cv2.IMREAD_GRAYSCALE)
                #print(q_image.shape,q_label.shape)
                if q_image.shape[0] != q_label.shape[0] or q_image.shape[1] != q_label.shape[1]:
                    #print('aaaaaaaaaaaaaaaaaa')
                    q_label= cv2.resize(q_label, dsize=(q_image.shape[1],q_image.shape[0]) , interpolation=cv2.INTER_NEAREST)
          
                q_image = np.float32(q_image)
                #  
               
                #print(q_image.shape,q_label.shape)
                q_label_list = np.unique(q_label).tolist()
                #print(q_label_list)
                index1 = index1 +random.randint(1,5)


          if q_image.shape[0] != q_label.shape[0] or q_image.shape[1] != q_label.shape[1]:

                raise (RuntimeError("Query Image & label shape mismatch: " + q_image_path + " " + q_label_path + "\n"))
          
          #perform data augmentation on query images
          if self.transform:
                q_image,q_label = self.transform(q_image,q_label)
          #q_label_temp = q_label.cpu().numpy()
          #q_label_temp = cv2.resize(q_label_temp, dsize=(int(q_label.shape[0]/8), int(q_label.shape[1]/8)),interpolation=cv2.INTER_NEAREST) 
          q_label_list = np.unique(q_label).tolist()#最后计算原型的特征会进行8倍下采样 某些标签可能会消失

          train_label_list= []
          while len(train_label_list) <= 1: #query image and support image must contain at least 2 same labels
                s_image_idx = random.randint(1,self.support_dataset_size)-1
                s_image_path = self.support_list[s_image_idx]['image_path']
                s_label_path = self.support_list[s_image_idx ]['label_path']

                # while(s_image_path == q_image_path):#support image and query image must be different 
                #    s_image_idx = random.randint(1,self.dataset_size)-1
                #    s_image_path = self.train_list[s_image_idx ]['image_path']
                #    s_label_path = self.train_list[s_image_idx ]['label_path']
                #end
               #print(q_image_path)
               #print(s_image_path)

                s_image = cv2.imread(s_image_path, cv2.IMREAD_COLOR) 
                s_image = cv2.cvtColor(s_image, cv2.COLOR_BGR2RGB)  

                s_image = np.float32(s_image)
                s_label = np.array(Image.open(s_label_path))#cv2.imread(s_label_path, cv2.IMREAD_GRAYSCALE)
                if self.transform:
                    s_image,s_label = self.transform(s_image,s_label)
            
                # s_label_temp = s_label
                # s_label_temp = F.interpolate( s_label_temp.unsqueeze(0).unsqueeze(0).float, size=(129, 129), mode='nearest')
                # print(s_label_temp.shape) 
                # print(np.unique(s_label_temp).tolist())
                s_label_list = np.unique(s_label).tolist()
                # print(s_label_list)
                #qscommon_label = list(set(q_label_list) & set(s_label_list))
                        
                #if 0 in train_label:
                #  train_label.remove(0)#0 is the ignored label
                #print(train_label)

                train_label_list = list (set(s_label_list) & set(self.validlabel))#we only consider label that contained in 19 labels of cityscape dataset
                #print(len(train_label_list))

           
                train_label_list_temp = train_label_list
                #poptimes = 0
                for k, c in enumerate(train_label_list):#range(n_classes):
                #print('ccccccccccccccccccc',c,train_label_list)
                #print(k,c)
                    mask = (s_label[:,:] == c).float().unsqueeze(0).unsqueeze(0)

                    mask = F.interpolate(mask, size=(129,129), mode='bilinear', align_corners=True)
                    #print('mask.shape after interpolation',np.unique(mask.cpu().numpy()))# 降采样后 MASK中有些小目标可能干没了
                    #qmask = (q_label[:,:] == c).float().unsqueeze(0).unsqueeze(0)
                    #qmask = F.interpolate(qmask, size=(129,129), mode='bilinear', align_corners=True)    

                    if 1 in np.unique(mask.cpu().numpy()) : #in general, an image may not contribute prototype to all the classes
                        #if 1 in np.unique(qmask.cpu().numpy()) :
                            pass
                        #else:
                            #print('in data loading, query disapper', k,c)
                         #   train_label_list_temp.remove(c)#pop(k-poptimes)
                            #poptimes = poptimes+1
                    else:
                         print('in data loading, support disapper', k,c)
                         train_label_list_temp.remove(c)#pop(k-poptimes)
                         #poptimes = poptimes+1
                    #print('end for ')
                    
          
                        
                train_label_list = train_label_list_temp
                if (len(train_label_list)<=1):
                    print(q_image_path)
                    print(s_image_path)
            # end of  while

            # labels contained both in support image and query image will be used for few-shot learning, other labels are set to invalid 
          for i in s_label_list:
                if i not in train_label_list:
                    pixelind_i = np.where(s_label == i)
                    s_label[pixelind_i[0],pixelind_i[1]] = 255  

          
          for i in q_label_list:
                if i not in train_label_list:
                    pixelind_i = np.where(q_label == i)
                    q_label[pixelind_i[0],pixelind_i[1]] = 255
          #print(train_label_list)
          #According to the label numbers (n) in train_label_list, formulate the few shot learning into a n-class classification problem
          
          pixelind_s=[]
          pixelind_q=[]
          for i, k in enumerate(train_label_list):
                #print(i,train_label_list.index(i))
                pixelind_i = np.where(s_label==k)
                pixelind_s.append(pixelind_i)

                pixelind_i = np.where(q_label==k)
                pixelind_q.append(pixelind_i)
          
        
          for i, k in enumerate(train_label_list):
               s_label[pixelind_s[i][0],pixelind_s[i][1]] = i           # train_label_list.index(i) 
               q_label[pixelind_q[i][0],pixelind_q[i][1]] =i           #train_label_list.index(i)

          #print(q_label_path,q_label_list)
          #print(s_label_path,s_label_list)
         
          #im=Image.frombytes('L', (s_label.shape[1],s_label.shape[0]),s_label)
          #im.save('slabelsaved.png','png')
          train_label_list = [i for i in range(len(train_label_list))]
          
          return q_image,q_label, q_image_name, s_image.unsqueeze(0),s_label.unsqueeze(0),train_label_list

     


      def get_train_list(self, train_path):

         image_dir = train_path[0]
         label_dir = train_path[1]



         if not os.path.exists(label_dir):
              raise IOError("No such training direcotry exist!")
         match_str=os.path.join(label_dir,  "*.png")
         #print(match_str)
         image_label_path_list=[]
         image_record_list=[]
         image_label_path_list.extend(glob.glob(match_str))
         #print(len(image_label_path_list))
    
         for f in image_label_path_list:                   
             image=os.path.splitext(f.split('/')[-1])[0]
        
             image_name=image
            
             image_path=os.path.join(image_dir, image_name+'.png')
        
             if not os.path.exists(image_path):
                print("Image %s is not exist %s" % (image_name,image_dir))
                raise IOError("Please Check")

             else:
                image_record={'label_path':f,'image_path':image_path,'image_name':image_name}
                                                                                                
                image_record_list.append(image_record)


         #print(image_record_list)
         print (len(image_record_list))
         
         return image_record_list


      def get_support_list(self, support_path):

            image_dir = support_path[0]
            label_dir = support_path[1]
            if not os.path.exists(label_dir):
                raise IOError("No such training direcotry exist!")

            image_record_list=[]
        

            
            for image_name in fewshot_name_list:                   
                
                cityname=image_name.split('_')
                cityname=cityname[0]
                #print(city)
                
                #print(image_name)
                label_path = os.path.join(label_dir, cityname, image_name+'_gtFine_labelIds.png')
                image_path=os.path.join(image_dir, cityname, image_name+'_leftImg8bit.png')
            
                if not os.path.exists(image_path):
                    #print("Image %s is not exist %s" % (image_name,image_dir))
                    raise IOError("Please Check")

                else:
                    image_record={'label_path':label_path,'image_path':image_path,'image_name':image_name}
                                                                                                    
                image_record_list.append(image_record)
           
            print ('the length of support images',len(image_record_list))
            return image_record_list

class City2BDD(Dataset):
      
      def __init__(self, train_path,  support_path, train_size=None, transform=None, use_val=False):
          super(City2BDD, self).__init__()
          self.train_path = train_path
          self.train_size = train_size
          self.transform = transform
          self.validlabel= [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
          self.train_list = self.get_train_list(train_path)
          self.support_list = self.get_support_list(support_path)
          #self.transform = transforms.Compose(transforms_)
          self.dataset_size = len(self.train_list)
          self.support_dataset_size = len (self.support_list )

      def __len__(self):
         return self.dataset_size


      def __getitem__(self, index):

          q_label_list =[]
          index1 = index
          while len(list (set(q_label_list) & set(self.validlabel))) <= 2:#some images in GTA5 contains only one valid label, we ignore these images during training
                q_image_path = self.train_list[index1]['image_path']
                q_label_path = self.train_list[index1]['label_path']
                q_image_name = self.train_list[index1]['image_name']
                q_image = cv2.imread(q_image_path, cv2.IMREAD_COLOR) 
                q_image = cv2.cvtColor(q_image, cv2.COLOR_BGR2RGB)  

                q_label = np.array(Image.open(q_label_path))
                #q_label = cv2.imread(q_label_path, cv2.IMREAD_GRAYSCALE)
                #print(q_image.shape,q_label.shape)
                if q_image.shape[0] != q_label.shape[0] or q_image.shape[1] != q_label.shape[1]:
                    #print('aaaaaaaaaaaaaaaaaa')
                    q_label= cv2.resize(q_label, dsize=(q_image.shape[1],q_image.shape[0]) , interpolation=cv2.INTER_NEAREST)
          
                q_image = np.float32(q_image)
                #  
               
                #print(q_image.shape,q_label.shape)
                q_label_list = np.unique(q_label).tolist()
                #print(q_label_list)
                index1 = index1 +random.randint(1,5)


          if q_image.shape[0] != q_label.shape[0] or q_image.shape[1] != q_label.shape[1]:

                raise (RuntimeError("Query Image & label shape mismatch: " + q_image_path + " " + q_label_path + "\n"))
          
          #perform data augmentation on query images
          if self.transform:
                q_image,q_label = self.transform(q_image,q_label)
          #q_label_temp = q_label.cpu().numpy()
          #q_label_temp = cv2.resize(q_label_temp, dsize=(int(q_label.shape[0]/8), int(q_label.shape[1]/8)),interpolation=cv2.INTER_NEAREST) 
          q_label_list = np.unique(q_label).tolist()#最后计算原型的特征会进行8倍下采样 某些标签可能会消失

          train_label_list= []
          while len(train_label_list) <= 1: #query image and support image must contain at least 2 same labels
                s_image_idx = random.randint(1,self.support_dataset_size)-1
                s_image_path = self.support_list[s_image_idx]['image_path']
                s_label_path = self.support_list[s_image_idx ]['label_path']

                # while(s_image_path == q_image_path):#support image and query image must be different 
                #    s_image_idx = random.randint(1,self.dataset_size)-1
                #    s_image_path = self.train_list[s_image_idx ]['image_path']
                #    s_label_path = self.train_list[s_image_idx ]['label_path']
                #end
               #print(q_image_path)
               #print(s_image_path)

                s_image = cv2.imread(s_image_path, cv2.IMREAD_COLOR) 
                s_image = cv2.cvtColor(s_image, cv2.COLOR_BGR2RGB)  

                s_image = np.float32(s_image)
                s_label = np.array(Image.open(s_label_path))#cv2.imread(s_label_path, cv2.IMREAD_GRAYSCALE)
                if self.transform:
                    s_image,s_label = self.transform(s_image,s_label)
            
                # s_label_temp = s_label
                # s_label_temp = F.interpolate( s_label_temp.unsqueeze(0).unsqueeze(0).float, size=(129, 129), mode='nearest')
                # print(s_label_temp.shape) 
                # print(np.unique(s_label_temp).tolist())
                s_label_list = np.unique(s_label).tolist()
                # print(s_label_list)
                #qscommon_label = list(set(q_label_list) & set(s_label_list))
                        
                #if 0 in train_label:
                #  train_label.remove(0)#0 is the ignored label
                #print(train_label)

                train_label_list = list (set(s_label_list) & set(self.validlabel))#we only consider label that contained in 19 labels of cityscape dataset
                #print(len(train_label_list))

           
                train_label_list_temp = train_label_list
                #poptimes = 0
                for k, c in enumerate(train_label_list):#range(n_classes):
                #print('ccccccccccccccccccc',c,train_label_list)
                #print(k,c)
                    mask = (s_label[:,:] == c).float().unsqueeze(0).unsqueeze(0)

                    mask = F.interpolate(mask, size=(129,129), mode='bilinear', align_corners=True)
                    #print('mask.shape after interpolation',np.unique(mask.cpu().numpy()))# 降采样后 MASK中有些小目标可能干没了
                    #time.sleep(20)
                    #qmask = (q_label[:,:] == c).float().unsqueeze(0).unsqueeze(0)
                    #qmask = F.interpolate(qmask, size=(129,129), mode='bilinear', align_corners=True)    

                    if 1 in np.unique(mask.cpu().numpy()) : #in general, an image may not contribute prototype to all the classes
                        #if 1 in np.unique(qmask.cpu().numpy()) :
                            pass
                        #else:
                            #print('in data loading, query disapper', k,c)
                         #   train_label_list_temp.remove(c)#pop(k-poptimes)
                            #poptimes = poptimes+1
                    else:
                         print('in data loading, support disapper', k,c)
                         train_label_list_temp.remove(c)#pop(k-poptimes)
                         #poptimes = poptimes+1
                    #print('end for ')
                    
          
                        
                train_label_list = train_label_list_temp
                if (len(train_label_list)<=1):
                    print(q_image_path)
                    print(s_image_path)
            # end of  while

            # labels contained both in support image and query image will be used for few-shot learning, other labels are set to invalid 
          for i in s_label_list:
                if i not in train_label_list:
                    pixelind_i = np.where(s_label == i)
                    s_label[pixelind_i[0],pixelind_i[1]] = 255  

          
          for i in q_label_list:
                if i not in train_label_list:
                    pixelind_i = np.where(q_label == i)
                    q_label[pixelind_i[0],pixelind_i[1]] = 255
          #print(train_label_list)
          #According to the label numbers (n) in train_label_list, formulate the few shot learning into a n-class classification problem
          
          pixelind_s=[]
          pixelind_q=[]
          for i, k in enumerate(train_label_list):
                #print(i,train_label_list.index(i))
                pixelind_i = np.where(s_label==k)
                pixelind_s.append(pixelind_i)

                pixelind_i = np.where(q_label==k)
                pixelind_q.append(pixelind_i)
          
        
          for i, k in enumerate(train_label_list):
               s_label[pixelind_s[i][0],pixelind_s[i][1]] = i           # train_label_list.index(i) 
               q_label[pixelind_q[i][0],pixelind_q[i][1]] =i           #train_label_list.index(i)

          #print(q_label_path,q_label_list)
          #print(s_label_path,s_label_list)
         
          #im=Image.frombytes('L', (s_label.shape[1],s_label.shape[0]),s_label)
          #im.save('slabelsaved.png','png')
          train_label_list = [i for i in range(len(train_label_list))]
          
          return q_image,q_label, q_image_name, s_image.unsqueeze(0),s_label.unsqueeze(0),train_label_list

     


      def get_train_list(self, train_path):

         image_dir = train_path[0]
         label_dir = train_path[1]

         if not os.path.exists(label_dir):
              raise IOError("No such training direcotry exist!")
         match_str=os.path.join(label_dir, '*', "*labelTrainIds.png")
         print(match_str)
         image_label_path_list=[]
         image_record_list=[]
         image_label_path_list.extend(glob.glob(match_str))

         
         for f in image_label_path_list:                   
             image=os.path.splitext(f.split('/')[-1])[0]
             image=image.split('_')
             city=image[0]
             #print(city)
             image_name=image[0]+'_'+image[1]+'_'+image[2]
             #print(image_name)
             image_path=os.path.join(image_dir, city, image_name+'_leftImg8bit.png')
        
             if not os.path.exists(image_path):
                #print("Image %s is not exist %s" % (image_name,image_dir))
                raise IOError("Please Check")

             else:
                image_record={'label_path':f,'image_path':image_path,'image_name':image_name}
                                                                                                
                image_record_list.append(image_record)

      
         print (len(image_record_list))
         
         return image_record_list


      def get_support_list(self, support_path):

            image_dir = support_path[0]
            label_dir = support_path[1]
            if not os.path.exists(label_dir):
                raise IOError("No such training direcotry exist!")

            image_record_list=[]
                   
            for image_name in fewshot_name_list:                   
                
            
                
                #print(image_name)
                label_path = os.path.join(label_dir, image_name+'_train_id.png')
                image_path=os.path.join(image_dir, image_name+'.jpg')
            
                if not os.path.exists(image_path):
                    #print("Image %s is not exist %s" % (image_name,image_dir))
                    raise IOError("Please Check")

                else:
                    image_record={'label_path':label_path,'image_path':image_path,'image_name':image_name}
                                                                                                    
                image_record_list.append(image_record)
           
            print ('the length of support images',len(image_record_list))
            return image_record_list

class BDD2City(Dataset):
      
      def __init__(self, train_path,  support_path, train_size=None, transform=None, use_val=False):
          super(BDD2City, self).__init__()
          self.train_path = train_path
          self.train_size = train_size
          self.transform = transform
          self.validlabel= [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
          self.train_list = self.get_train_list(train_path)
          self.support_list = self.get_support_list(support_path)
          #self.transform = transforms.Compose(transforms_)
          self.dataset_size = len(self.train_list)
          self.support_dataset_size = len (self.support_list )

      def __len__(self):
         return self.dataset_size


      def __getitem__(self, index):

          q_label_list =[]
          index1 = index
          while len(list (set(q_label_list) & set(self.validlabel))) <= 2:#some images in GTA5 contains only one valid label, we ignore these images during training
                q_image_path = self.train_list[index1]['image_path']
                q_label_path = self.train_list[index1]['label_path']
                q_image_name = self.train_list[index1]['image_name']
                q_image = cv2.imread(q_image_path, cv2.IMREAD_COLOR) 
                q_image = cv2.cvtColor(q_image, cv2.COLOR_BGR2RGB)  

                q_label = np.array(Image.open(q_label_path))
                #q_label = cv2.imread(q_label_path, cv2.IMREAD_GRAYSCALE)
                #print(q_image.shape,q_label.shape)
                if q_image.shape[0] != q_label.shape[0] or q_image.shape[1] != q_label.shape[1]:
                    #print('aaaaaaaaaaaaaaaaaa')
                    q_label= cv2.resize(q_label, dsize=(q_image.shape[1],q_image.shape[0]) , interpolation=cv2.INTER_NEAREST)
          
                q_image = np.float32(q_image)
                #  
               
                #print(q_image.shape,q_label.shape)
                q_label_list = np.unique(q_label).tolist()
                #print(q_label_list)
                index1 = index1 +random.randint(1,5)


          if q_image.shape[0] != q_label.shape[0] or q_image.shape[1] != q_label.shape[1]:

                raise (RuntimeError("Query Image & label shape mismatch: " + q_image_path + " " + q_label_path + "\n"))
          
          #perform data augmentation on query images
          if self.transform:
                q_image,q_label = self.transform(q_image,q_label)
          #q_label_temp = q_label.cpu().numpy()
          #q_label_temp = cv2.resize(q_label_temp, dsize=(int(q_label.shape[0]/8), int(q_label.shape[1]/8)),interpolation=cv2.INTER_NEAREST) 
          q_label_list = np.unique(q_label).tolist()#最后计算原型的特征会进行8倍下采样 某些标签可能会消失

          train_label_list= []
          while len(train_label_list) <= 1: #query image and support image must contain at least 2 same labels
                s_image_idx = random.randint(1,self.support_dataset_size)-1
                s_image_path = self.support_list[s_image_idx]['image_path']
                s_label_path = self.support_list[s_image_idx ]['label_path']

                # while(s_image_path == q_image_path):#support image and query image must be different 
                #    s_image_idx = random.randint(1,self.dataset_size)-1
                #    s_image_path = self.train_list[s_image_idx ]['image_path']
                #    s_label_path = self.train_list[s_image_idx ]['label_path']
                #end
               #print(q_image_path)
               #print(s_image_path)

                s_image = cv2.imread(s_image_path, cv2.IMREAD_COLOR) 
                s_image = cv2.cvtColor(s_image, cv2.COLOR_BGR2RGB)  

                s_image = np.float32(s_image)
                s_label = np.array(Image.open(s_label_path))#cv2.imread(s_label_path, cv2.IMREAD_GRAYSCALE)
                if self.transform:
                    s_image,s_label = self.transform(s_image,s_label)
            
                # s_label_temp = s_label
                # s_label_temp = F.interpolate( s_label_temp.unsqueeze(0).unsqueeze(0).float, size=(129, 129), mode='nearest')
                # print(s_label_temp.shape) 
                # print(np.unique(s_label_temp).tolist())
                s_label_list = np.unique(s_label).tolist()
                # print(s_label_list)
                #qscommon_label = list(set(q_label_list) & set(s_label_list))
                        
                #if 0 in train_label:
                #  train_label.remove(0)#0 is the ignored label
                #print(train_label)

                train_label_list = list (set(s_label_list) & set(self.validlabel))#we only consider label that contained in 19 labels of cityscape dataset
                #print(len(train_label_list))

           
                train_label_list_temp = train_label_list
                #poptimes = 0
                for k, c in enumerate(train_label_list):#range(n_classes):
                #print('ccccccccccccccccccc',c,train_label_list)
                #print(k,c)
                    mask = (s_label[:,:] == c).float().unsqueeze(0).unsqueeze(0)

                    mask = F.interpolate(mask, size=(129,129), mode='bilinear', align_corners=True)
                    #print('mask.shape after interpolation',np.unique(mask.cpu().numpy()))# 降采样后 MASK中有些小目标可能干没了
                    #time.sleep(20)
                    #qmask = (q_label[:,:] == c).float().unsqueeze(0).unsqueeze(0)
                    #qmask = F.interpolate(qmask, size=(129,129), mode='bilinear', align_corners=True)    

                    if 1 in np.unique(mask.cpu().numpy()) : #in general, an image may not contribute prototype to all the classes
                        #if 1 in np.unique(qmask.cpu().numpy()) :
                            pass
                        #else:
                            #print('in data loading, query disapper', k,c)
                         #   train_label_list_temp.remove(c)#pop(k-poptimes)
                            #poptimes = poptimes+1
                    else:
                         print('in data loading, support disapper', k,c)
                         train_label_list_temp.remove(c)#pop(k-poptimes)
                         #poptimes = poptimes+1
                    #print('end for ')
                    
          
                        
                train_label_list = train_label_list_temp
                if (len(train_label_list)<=1):
                    print(q_image_path)
                    print(s_image_path)
            # end of  while

            # labels contained both in support image and query image will be used for few-shot learning, other labels are set to invalid 
          for i in s_label_list:
                if i not in train_label_list:
                    pixelind_i = np.where(s_label == i)
                    s_label[pixelind_i[0],pixelind_i[1]] = 255  

          
          for i in q_label_list:
                if i not in train_label_list:
                    pixelind_i = np.where(q_label == i)
                    q_label[pixelind_i[0],pixelind_i[1]] = 255
          #print(train_label_list)
          #According to the label numbers (n) in train_label_list, formulate the few shot learning into a n-class classification problem
          
          pixelind_s=[]
          pixelind_q=[]
          for i, k in enumerate(train_label_list):
                #print(i,train_label_list.index(i))
                pixelind_i = np.where(s_label==k)
                pixelind_s.append(pixelind_i)

                pixelind_i = np.where(q_label==k)
                pixelind_q.append(pixelind_i)
          
        
          for i, k in enumerate(train_label_list):
               s_label[pixelind_s[i][0],pixelind_s[i][1]] = i           # train_label_list.index(i) 
               q_label[pixelind_q[i][0],pixelind_q[i][1]] =i           #train_label_list.index(i)

          #print(q_label_path,q_label_list)
          #print(s_label_path,s_label_list)
         
          #im=Image.frombytes('L', (s_label.shape[1],s_label.shape[0]),s_label)
          #im.save('slabelsaved.png','png')
          train_label_list = [i for i in range(len(train_label_list))]
          
          return q_image,q_label, q_image_name, s_image.unsqueeze(0),s_label.unsqueeze(0),train_label_list

     


      def get_train_list(self, train_path):

         image_dir = train_path[0]
         label_dir = train_path[1]



         if not os.path.exists(label_dir):
              raise IOError("No such training direcotry exist!")
         match_str=os.path.join(label_dir,  "*.png")
         #print(match_str)
         image_label_path_list=[]
         image_record_list=[]
         image_label_path_list.extend(glob.glob(match_str))
         #print(len(image_label_path_list))
    
         for f in image_label_path_list:                   
             image=os.path.splitext(f.split('/')[-1])[0]
        
             image_name=image.split('_')[0]
            
             image_path=os.path.join(image_dir, image_name+'.jpg')
        
             if not os.path.exists(image_path):
                print("Image %s is not exist %s" % (image_name,image_dir))
                raise IOError("Please Check")

             else:
                image_record={'label_path':f,'image_path':image_path,'image_name':image_name}
                                                                                                
                image_record_list.append(image_record)


         #print(image_record_list)
         print (len(image_record_list))
         
         return image_record_list


      def get_support_list(self, support_path):

            image_dir = support_path[0]
            label_dir = support_path[1]
            if not os.path.exists(label_dir):
                raise IOError("No such training direcotry exist!")

            image_record_list=[]
        

            
            for image_name in fewshot_name_list:                   
                
                cityname=image_name.split('_')
                cityname=cityname[0]
                #print(city)
                
                #print(image_name)
                label_path = os.path.join(label_dir, cityname, image_name+'_gtFine_labelTrainIds.png')
                image_path=os.path.join(image_dir, cityname, image_name+'_leftImg8bit.png')
            
                if not os.path.exists(image_path):
                    #print("Image %s is not exist %s" % (image_name,image_dir))
                    raise IOError("Please Check")

                else:
                    image_record={'label_path':label_path,'image_path':image_path,'image_name':image_name}
                                                                                                    
                image_record_list.append(image_record)
           
            print ('the length of support images',len(image_record_list))
            return image_record_list


class CityFEW(Dataset):
      
      def __init__(self, train_path,   train_size=None, transform=None, use_val=False):
          super(CityFEW, self).__init__()
          self.train_path = train_path
          self.train_size = train_size
          self.transform = transform
        
          self.train_list = self.get_train_list(train_path)
          
          self.dataset_size = len(self.train_list)
          

      def __len__(self):
         return self.dataset_size


      def __getitem__(self, index):

       
          if True:#while len(list (set(q_label_list) & set(self.validlabel))) <= 1:#some images in GTA5 contains only one valid label, we ignore these images during training
            q_image_path = self.train_list[index]['image_path']
            q_label_path = self.train_list[index]['label_path']
            q_image_name = self.train_list[index]['image_name']
            q_image = cv2.imread(q_image_path, cv2.IMREAD_COLOR) 
            q_image = cv2.cvtColor(q_image, cv2.COLOR_BGR2RGB)  

            q_image = np.float32(q_image)
            #q_label = cv2.imread(q_label_path, cv2.IMREAD_GRAYSCALE)  
            q_label = np.array(Image.open(q_label_path))
             
           
           # print(q_label_list)
           # index1 = index1 +random.randint(1,5)


          if q_image.shape[0] != q_label.shape[0] or q_image.shape[1] != q_label.shape[1]:
            raise (RuntimeError("Query Image & label shape mismatch: " + q_image_path + " " + q_label_path + "\n"))
          
          #perform data augmentation on query images
          if self.transform:
              q_image,q_label = self.transform(q_image,q_label)
        #   q_label1=trainid2labelid_efficient(q_label)
        #   color_pred=decode_labels_cityscape(q_label1)
        #   im=Image.fromarray(color_pred)
        #   im.show()
         
          return q_image,q_label,q_image_name
                 
      def get_train_list(self, support_path):

            image_dir = support_path[0]
            label_dir = support_path[1]
            if not os.path.exists(label_dir):
                raise IOError("No such training direcotry exist!")

            image_record_list=[]
        

            
            for image_name in fewshot_name_list:                   
                
                cityname=image_name.split('_')
                cityname=cityname[0]
                #print(city)
                
                #print(image_name)
                label_path = os.path.join(label_dir, cityname, image_name+'_gtFine_labelTrainIds.png')
                image_path=os.path.join(image_dir, cityname, image_name+'_leftImg8bit.png')
            
                if not os.path.exists(image_path):
                    #print("Image %s is not exist %s" % (image_name,image_dir))
                    raise IOError("Please Check")

                else:
                    image_record={'label_path':label_path,'image_path':image_path,'image_name':image_name}
                                                                                                    
                image_record_list.append(image_record)
           
            print ('the length of support images',len(image_record_list))
            return image_record_list 
          

label_colours_cityscape = [(0,0,0), (0,  0,  0),(  0,  0,  0),(  0,  0,  0),( 0,  0,  0),
               
                (111, 74,  0),( 81,  0, 81) ,(128, 64,128),(244, 35,232) ,(250,170,160),
                
                (230,150,140),( 70, 70, 70),(102,102,156),(190,153,153),(180,165,180),
               
                (150,100,100),(150,120, 90),(153,153,153),(153,153,153),(250,170, 30),
                
                (220,220,  0),(107,142, 35) ,(152,251,152) ,( 70,130,180),(220, 20, 60),
                (255,  0,  0),(  0,  0,142),(  0,  0, 70),(  0, 60,100),(  0,  0, 90),
                (  0,  0,110),(  0, 80,100),(  0,  0,230),(119, 11, 32)]


def decode_labels_cityscape(mask):
    """Decode segmentation masks.
    
    Args:
      mask: result of inference after taking argmax.

    
    Returns:
      A  RGB image of the same size as the input. 
    """
    h, w = mask.shape
    #assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((h, w, 3), dtype=np.uint8)
    #for i in range(num_images):
    #img = Image.new('RGB', (len(mask[0]), len(mask)))
    #pixels = img.load()
    for i in range(h):
        for j in range(w):
            if mask[i,j]<255:  
                 outputs[i,j,:] = label_colours_cityscape[mask[i,j]]
            else:
                 outputs[i,j,:] = [0,0,0]      
    #outputs = np.array(img)
    return outputs

def trainid2labelid_efficient(prediction):
    shape=np.shape(prediction)
    prediction=prediction+13
    prediction=prediction.reshape(-1)
    #color=np.zeros(prediction.shape,3)
    index=np.where(prediction==13)
    #print(index[0].shape)
    index1=np.where(prediction==14)
    index=np.concatenate((index[0],index1[0]))
    #print(index.shape)
    
    prediction[index]=prediction[index]-6
    index=np.where(prediction==15) 
    index1=np.where(prediction==16) 
    index2=np.where(prediction==17) 
    index=np.concatenate((index[0],index1[0],index2[0]))
    prediction[index]=prediction[index]-4

    index=np.where(prediction==18)
    prediction[index[0]]=prediction[index[0]]-1

    index=np.where(prediction==29) 
    index1=np.where(prediction==30) 
    index2=np.where(prediction==31) 
    index=np.concatenate((index[0],index1[0],index2[0]))
    
    prediction[index]=prediction[index]+2
    prediction=prediction.reshape(shape[0],shape[1])
    return prediction

class GTA5Source(Dataset):
      
      def __init__(self, train_path,  support_path, train_size=None, transform=None, use_val=False):
          super(GTA5Source, self).__init__()
          self.train_path = train_path
          self.train_size = train_size
          self.transform = transform
          self.validlabel= [7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33]
          self.train_list = self.get_train_list(train_path)
          self.support_list = self.get_support_list(support_path)
          #self.transform = transforms.Compose(transforms_)
          self.dataset_size = len(self.train_list)
          self.support_dataset_size = len (self.support_list )

      def __len__(self):
         return self.dataset_size


      def __getitem__(self, index):

          q_label_list =[]
          index1 = index
          if True:#while len(list (set(q_label_list) & set(self.validlabel))) <= 1:#some images in GTA5 contains only one valid label, we ignore these images during training
            q_image_path = self.train_list[index1]['image_path']
            q_label_path = self.train_list[index1]['label_path']
            q_image_name = self.train_list[index1]['image_name']
            q_image = cv2.imread(q_image_path, cv2.IMREAD_COLOR) 
            q_image = cv2.cvtColor(q_image, cv2.COLOR_BGR2RGB)  

            q_image = np.float32(q_image)
            #q_label = cv2.imread(q_label_path, cv2.IMREAD_GRAYSCALE)  
            q_label = np.array(Image.open(q_label_path))
             
           
           # print(q_label_list)
           # index1 = index1 +random.randint(1,5)


          if q_image.shape[0] != q_label.shape[0] or q_image.shape[1] != q_label.shape[1]:
                   
                    #print('aaaaaaaaaaaaaaaaaa')
                    q_label= cv2.resize(q_label, dsize=(q_image.shape[1],q_image.shape[0]) , interpolation=cv2.INTER_NEAREST)
            #raise (RuntimeError("Query Image & label shape mismatch: " + q_image_path + " " + q_label_path + "\n"))
          
          #perform data augmentation on query images
          if self.transform:
              q_image,q_label = self.transform(q_image,q_label)
          
          q_label_list = np.unique(q_label).tolist() 
          train_label_list= []
       
    

          train_label_list = list (set(q_label_list) & set(self.validlabel))#we only consider label that contained in 19 labels of cityscape dataset
          #print(len(train_label_list))
    
          

         
          
          for i in q_label_list:
             if i not in train_label_list:
                pixelind_i = np.where(q_label == i)
                q_label[pixelind_i[0],pixelind_i[1]] = 255
          #print(train_label_list)
          #According to the label numbers (n) in train_label_list, formulate the few shot learning into a n-class classification problem
          q_label_temp = q_label
          for i in train_label_list:
            #print(i,train_label_list.index(i))
            pixelind_i = np.where(q_label_temp==i)
            q_label[pixelind_i[0],pixelind_i[1]] = self.validlabel.index(i) 


        #   q_label1=trainid2labelid_efficient(q_label)
        #   color_pred=decode_labels_cityscape(q_label1)
        #   im=Image.fromarray(color_pred)
        #   im.show()
         
          return q_image,q_label,q_image_name

     


      def get_train_list(self, train_path):

         image_dir = train_path[0]
         label_dir = train_path[1]



         if not os.path.exists(label_dir):
              raise IOError("No such training direcotry exist!")
         match_str=os.path.join(label_dir,  "*.png")
         #print(match_str)
         image_label_path_list=[]
         image_record_list=[]
         image_label_path_list.extend(glob.glob(match_str))
         #print(len(image_label_path_list))
    
         for f in image_label_path_list:                   
             image=os.path.splitext(f.split('/')[-1])[0]
        
             image_name=image
            
             image_path=os.path.join(image_dir, image_name+'.png')
        
             if not os.path.exists(image_path):
                print("Image %s is not exist %s" % (image_name,image_dir))
                raise IOError("Please Check")

             else:
                image_record={'label_path':f,'image_path':image_path,'image_name':image_name}
                                                                                                
                image_record_list.append(image_record)


         #print(image_record_list)
         print (len(image_record_list))
         
         return image_record_list


      def get_support_list(self, support_path):

            image_dir = support_path[0]
            label_dir = support_path[1]
            if not os.path.exists(label_dir):
                raise IOError("No such training direcotry exist!")

            image_record_list=[]
        

            
            for image_name in fewshot_name_list:                   
                
                cityname=image_name.split('_')
                cityname=cityname[0]
                #print(city)
                
                #print(image_name)
                label_path = os.path.join(label_dir, cityname, image_name+'_gtFine_labelIds.png')
                image_path=os.path.join(image_dir, cityname, image_name+'_leftImg8bit.png')
            
                if not os.path.exists(image_path):
                    #print("Image %s is not exist %s" % (image_name,image_dir))
                    raise IOError("Please Check")

                else:
                    image_record={'label_path':label_path,'image_path':image_path,'image_name':image_name}
                                                                                                    
                image_record_list.append(image_record)
           
            print ('the length of support images',len(image_record_list))
            return image_record_list

class BDDSource(Dataset):
      
      def __init__(self, train_path,  support_path, train_size=None, transform=None, use_val=False):
          super(BDDSource, self).__init__()
          self.train_path = train_path
          self.train_size = train_size
          self.transform = transform
          self.validlabel= [7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33]
          self.train_list = self.get_train_list(train_path)
          #self.support_list = self.get_support_list(support_path)
          #self.transform = transforms.Compose(transforms_)
          self.dataset_size = len(self.train_list)
          #self.support_dataset_size = len (self.support_list )

      def __len__(self):
         return self.dataset_size


      def __getitem__(self, index):

          q_label_list =[]
          index1 = index
          if True:#while len(list (set(q_label_list) & set(self.validlabel))) <= 1:#some images in GTA5 contains only one valid label, we ignore these images during training
            q_image_path = self.train_list[index1]['image_path']
            q_label_path = self.train_list[index1]['label_path']
            q_image_name = self.train_list[index1]['image_name']
            q_image = cv2.imread(q_image_path, cv2.IMREAD_COLOR) 
            q_image = cv2.cvtColor(q_image, cv2.COLOR_BGR2RGB)  

            q_image = np.float32(q_image)
            #q_label = cv2.imread(q_label_path, cv2.IMREAD_GRAYSCALE)  
            q_label = np.array(Image.open(q_label_path))
             
           
           # print(q_label_list)
           # index1 = index1 +random.randint(1,5)


          if q_image.shape[0] != q_label.shape[0] or q_image.shape[1] != q_label.shape[1]:
                   
                    #print('aaaaaaaaaaaaaaaaaa')
                    #q_label= cv2.resize(q_label, dsize=(q_image.shape[1],q_image.shape[0]) , interpolation=cv2.INTER_NEAREST)
                  raise (RuntimeError("Query Image & label shape mismatch: " + q_image_path + " " + q_label_path + "\n"))
          
          #perform data augmentation on query images
          if self.transform:
              q_image,q_label = self.transform(q_image,q_label)
          
         
         
          return q_image,q_label,q_image_name

     


      def get_train_list(self, train_path):

         image_dir = train_path[0]
         label_dir = train_path[1]



         if not os.path.exists(label_dir):
              raise IOError("No such training direcotry exist!")
         match_str=os.path.join(label_dir,  "*.png")
         #print(match_str)
         image_label_path_list=[]
         image_record_list=[]
         image_label_path_list.extend(glob.glob(match_str))
         #print(len(image_label_path_list))
    
         for f in image_label_path_list:                   
             image=os.path.splitext(f.split('/')[-1])[0]
        
             image_name=image.split('_')[0]
            
             image_path=os.path.join(image_dir, image_name+'.jpg')
        
             if not os.path.exists(image_path):
                print("Image %s is not exist %s" % (image_name,image_dir))
                raise IOError("Please Check")

             else:
                image_record={'label_path':f,'image_path':image_path,'image_name':image_name}
                                                                                                
                image_record_list.append(image_record)


         #print(image_record_list)
         print (len(image_record_list))
         
         return image_record_list


      

class CitySource(Dataset):
      
      def __init__(self, train_path,  support_path, train_size=None, transform=None, use_val=False):
          super(CitySource, self).__init__()
          self.train_path = train_path
          self.train_size = train_size
          self.transform = transform
          self.validlabel= [7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33]
          self.train_list = self.get_train_list(train_path)
          #self.support_list = self.get_support_list(support_path)
          #self.transform = transforms.Compose(transforms_)
          self.dataset_size = len(self.train_list)
         # self.support_dataset_size = len (self.support_list )

      def __len__(self):
         return self.dataset_size


      def __getitem__(self, index):

          q_label_list =[]
          index1 = index
          if True:#while len(list (set(q_label_list) & set(self.validlabel))) <= 1:#some images in GTA5 contains only one valid label, we ignore these images during training
            q_image_path = self.train_list[index1]['image_path']
            q_label_path = self.train_list[index1]['label_path']
            q_image_name = self.train_list[index1]['image_name']
            q_image = cv2.imread(q_image_path, cv2.IMREAD_COLOR) 
            q_image = cv2.cvtColor(q_image, cv2.COLOR_BGR2RGB)  

            q_image = np.float32(q_image)
            #q_label = cv2.imread(q_label_path, cv2.IMREAD_GRAYSCALE)  
            q_label = np.array(Image.open(q_label_path))
             
           
           # print(q_label_list)
           # index1 = index1 +random.randint(1,5)


          if q_image.shape[0] != q_label.shape[0] or q_image.shape[1] != q_label.shape[1]:
                   
                    #print('aaaaaaaaaaaaaaaaaa')
                    #q_label= cv2.resize(q_label, dsize=(q_image.shape[1],q_image.shape[0]) , interpolation=cv2.INTER_NEAREST)
                  raise (RuntimeError("Query Image & label shape mismatch: " + q_image_path + " " + q_label_path + "\n"))
          
          #perform data augmentation on query images
          if self.transform:
              q_image,q_label = self.transform(q_image,q_label)
          
         
         
          return q_image,q_label,q_image_name

     
      def get_train_list(self, train_path):

         image_dir = train_path[0]
         label_dir = train_path[1]

         if not os.path.exists(label_dir):
              raise IOError("No such training direcotry exist!")
         match_str=os.path.join(label_dir, '*', "*labelTrainIds.png")
         print(match_str)
         image_label_path_list=[]
         image_record_list=[]
         image_label_path_list.extend(glob.glob(match_str))

         
         for f in image_label_path_list:                   
             image=os.path.splitext(f.split('/')[-1])[0]
             image=image.split('_')
             city=image[0]
             #print(city)
             image_name=image[0]+'_'+image[1]+'_'+image[2]
             #print(image_name)
             image_path=os.path.join(image_dir, city, image_name+'_leftImg8bit.png')
        
             if not os.path.exists(image_path):
                #print("Image %s is not exist %s" % (image_name,image_dir))
                raise IOError("Please Check")

             else:
                image_record={'label_path':f,'image_path':image_path,'image_name':image_name}
                                                                                                
                image_record_list.append(image_record)

      
         print (len(image_record_list))
         
         return image_record_list

    
      

class CityscapesTest(Dataset):
      
      def __init__(self, support_path, query_path, classes = 19,transform=None,computeproto=False):
          super(CityscapesTest, self).__init__()
          self.support_path = support_path
          #self.train_size = train_size
          self.query_path = query_path
          #self.use_val = use_val
          self.num_classes = classes
          self.train_label_list = [i for i in range(classes)]
          self.query_image_list, self.support_image_list = self.get_support_and_query_list(support_path, query_path)
          self.transform = transform
          #self.simage_tensor,self.simage_labeltensor = self.get_support_image_label_tensor(self.support_image_list)
          self.computeproto = computeproto
          
          if self.computeproto:
             self.file_length = len(self.support_image_list)
          else:
             self.file_length = len(self.query_image_list)


      def __len__(self):
         return self.file_length


      def __getitem__(self, index):

          if self.computeproto:

            image_path = self.support_image_list[index]['image_path']
            label_path = self.support_image_list[index]['label_path']
            image_name = self.support_image_list[index]['image_name']

          

          else:

            image_path = self.query_image_list[index]['image_path']
            label_path = self.query_image_list[index]['label_path']
            image_name = self.query_image_list[index]['image_name']

          image = cv2.imread(image_path, cv2.IMREAD_COLOR) 
          image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  

          image = np.float32(image)
          #q_label = cv2.imread(q_label_path, cv2.IMREAD_GRAYSCALE)  
          label = np.array(Image.open(label_path))
          if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Query Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
          if self.transform:
            image, label = self.transform(image, label)
          
          #print(image_name)

          return image, label, self.train_label_list, image_name
     

      def get_support_image_label_tensor(self, support_image_list):
          simage_tensor_list = []
          simage_label_tensor_list = []

          for i in range(len(support_image_list)):
              s_image = cv2.imread(support_image_list[i]['image_path'], cv2.IMREAD_COLOR) 
              s_image = cv2.cvtColor(s_image, cv2.COLOR_BGR2RGB)  

              s_image = np.float32(s_image)
              s_label = np.array(Image.open(support_image_list[i]['label_path']))
              if s_image.shape[0] != s_label.shape[0] or s_image.shape[1] != s_label.shape[1]:
                raise (RuntimeError("Query Image & label shape mismatch: " "\n"))
              if self.transform:
                  s_image, s_label = self.transform(s_image, s_label)
              simage_tensor_list.append(s_image) 
              simage_label_tensor_list.append(s_label)
          simage_rgbtensor = torch.stack(simage_tensor_list,dim=0)
          simage_labeltensor = torch.stack(simage_label_tensor_list,dim=0)
          return simage_rgbtensor, simage_labeltensor#simage_tensor_list, simage_label_tensor_list
              

      
      def get_support_and_query_list(self, support_path, query_path):

         s_image_dir = support_path[0]
         s_label_dir = support_path[1]

         q_image_dir = query_path[0]
         q_label_dir = query_path[1]




         if not os.path.exists(s_label_dir):
              raise IOError("No such training direcotry exist!")
         match_str=os.path.join(s_label_dir, '*', "*labelTrainIds.png")
         print(match_str)
         simage_label_path_list=[]
         s_image_list=[]
         simage_label_path_list.extend(glob.glob(match_str))

         
         for f in simage_label_path_list:                   
             image=os.path.splitext(f.split('/')[-1])[0]
             image=image.split('_')
             city=image[0]
             #print(city)
             image_name=image[0]+'_'+image[1]+'_'+image[2]
             #print(image_name)
             image_path=os.path.join(s_image_dir, city, image_name+'_leftImg8bit.png')
        
             if not os.path.exists(image_path):
                #print("Image %s is not exist %s" % (image_name,image_dir))
                raise IOError("Please Check")

             else:
                image_record={'label_path':f,'image_path':image_path,'image_name':image_name}
                                                                                                
                s_image_list.append(image_record)


         


         if not os.path.exists(q_label_dir):
              raise IOError("No such training direcotry exist!")
         match_str=os.path.join(q_label_dir, '*', "*labelTrainIds.png")
         print(match_str)
         qimage_label_path_list=[]
         q_image_list=[]
         qimage_label_path_list.extend(glob.glob(match_str))
         print(len(qimage_label_path_list))
    
         for f in qimage_label_path_list:                   
             image=os.path.splitext(f.split('/')[-1])[0]
             image=image.split('_')
             city=image[0]
             #print(city)
             image_name=image[0]+'_'+image[1]+'_'+image[2]
             #print(image_name)
             image_path=os.path.join(q_image_dir, city, image_name+'_leftImg8bit.png')
        
             if not os.path.exists(image_path):
                #print("Image %s is not exist %s" % (image_name,image_dir))
                raise IOError("Please Check")

             else:
                image_record={'label_path':f,'image_path':image_path,'image_name':image_name}
                                                                                                
                q_image_list.append(image_record)


         return q_image_list, s_image_list

class BDDTest(Dataset):
      
      def __init__(self, support_path, query_path, classes = 19,transform=None,computeproto=False):
          super(BDDTest, self).__init__()
          self.support_path = support_path
          #self.train_size = train_size
          self.query_path = query_path
          #self.use_val = use_val
          self.num_classes = classes
          self.train_label_list = [i for i in range(classes)]
          self.query_image_list, self.support_image_list = self.get_support_and_query_list(support_path, query_path)
          self.transform = transform
          #self.simage_tensor,self.simage_labeltensor = self.get_support_image_label_tensor(self.support_image_list)
          self.computeproto = computeproto
          
          if self.computeproto:
             self.file_length = len(self.support_image_list)
          else:
             self.file_length = len(self.query_image_list)


      def __len__(self):
         return self.file_length


      def __getitem__(self, index):

          if self.computeproto:

            image_path = self.support_image_list[index]['image_path']
            label_path = self.support_image_list[index]['label_path']
            image_name = self.support_image_list[index]['image_name']

          

          else:

            image_path = self.query_image_list[index]['image_path']
            label_path = self.query_image_list[index]['label_path']
            image_name = self.query_image_list[index]['image_name']

          image = cv2.imread(image_path, cv2.IMREAD_COLOR) 
          image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  

          image = np.float32(image)
          #q_label = cv2.imread(q_label_path, cv2.IMREAD_GRAYSCALE)  
          label = np.array(Image.open(label_path))
          if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Query Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
          if self.transform:
            image, label = self.transform(image, label)
          
          #print(image_name)

          return image, label, self.train_label_list, image_name
     

      
      def get_support_and_query_list(self, support_path, query_path):

         s_image_dir = support_path[0]
         s_label_dir = support_path[1]

         q_image_dir = query_path[0]
         q_label_dir = query_path[1]




         if not os.path.exists(s_label_dir):
              raise IOError("No such training direcotry exist!")
         match_str=os.path.join(s_label_dir,  "*.png")
         #print(match_str)
         image_label_path_list=[]
         s_image_list=[]
         image_label_path_list.extend(glob.glob(match_str))
         #print(len(image_label_path_list))
    
         for f in image_label_path_list:                   
             image=os.path.splitext(f.split('/')[-1])[0]
        
             image_name=image.split('_')[0]
            
             image_path=os.path.join(s_image_dir, image_name+'.jpg')
        
             if not os.path.exists(image_path):
                print("Image %s is not exist %s" % (image_name,s_image_dir))
                raise IOError("Please Check")

             else:
                image_record={'label_path':f,'image_path':image_path,'image_name':image_name}
                                                                                                
                s_image_list.append(image_record)


         if not os.path.exists(q_label_dir):
              raise IOError("No such training direcotry exist!")
         match_str=os.path.join(q_label_dir,  "*.png")
         #print(match_str)
         image_label_path_list=[]
         q_image_list=[]
         image_label_path_list.extend(glob.glob(match_str))
         #print(len(image_label_path_list))
    
         for f in image_label_path_list:                   
             image=os.path.splitext(f.split('/')[-1])[0]
        
             image_name=image.split('_')[0]
            
             image_path=os.path.join(q_image_dir, image_name+'.jpg')
        
             if not os.path.exists(image_path):
                print("Image %s is not exist %s" % (image_name,q_image_dir))
                raise IOError("Please Check")

             else:
                image_record={'label_path':f,'image_path':image_path,'image_name':image_name}
                                                                                                
                q_image_list.append(image_record)
         
         return q_image_list, s_image_list


class GTATest(Dataset):
      
      def __init__(self, support_path, query_path, classes = 19,transform=None):
          super(GTATest, self).__init__()
          self.support_path = support_path
          #self.train_size = train_size
          self.query_path = query_path
          #self.use_val = use_val
          self.num_classes = classes
          self.query_image_list, self.support_image_list = self.get_support_and_query_list(support_path, query_path)
          self.transform = transform
          self.simage_tensor,self.simage_labeltensor = self.get_support_image_label_tensor(self.support_image_list)
          self.file_length = len(self.query_image_list)

      def __len__(self):
         return self.file_length


      def __getitem__(self, index):
          q_image_path = self.query_image_list[index]['image_path']
          q_label_path = self.query_image_list[index]['label_path']
          q_image_name = self.query_image_list[index]['image_name']
          q_image = cv2.imread(q_image_path, cv2.IMREAD_COLOR) 
          q_image = cv2.cvtColor(q_image, cv2.COLOR_BGR2RGB)  

          q_image = np.float32(q_image)
          #q_label = cv2.imread(q_label_path, cv2.IMREAD_GRAYSCALE)  
          q_label = np.array(Image.open(q_label_path))
          if q_image.shape[0] != q_label.shape[0] or q_image.shape[1] != q_label.shape[1]:
            raise (RuntimeError("Query Image & label shape mismatch: " + q_image_path + " " + q_label_path + "\n"))
          if self.transform:
            q_image, q_label = self.transform(q_image, q_label)
          
          

          return q_image, self.simage_tensor,self.simage_labeltensor,q_image_name
     

      def get_support_image_label_tensor(self, support_image_list):
          simage_tensor_list = []
          simage_label_tensor_list = []

          for i in range(len(support_image_list)):
              s_image = cv2.imread(support_image_list[i]['image_path'], cv2.IMREAD_COLOR) 
              s_image = cv2.cvtColor(s_image, cv2.COLOR_BGR2RGB)  

              s_image = np.float32(s_image)
              s_label = np.array(Image.open(support_image_list[i]['label_path']))
              if s_image.shape[0] != s_label.shape[0] or s_image.shape[1] != s_label.shape[1]:
                raise (RuntimeError("Query Image & label shape mismatch: " "\n"))
              if self.transform:
                  s_image, s_label = self.transform(s_image, s_label)
              simage_tensor_list.append(s_image) 
              simage_label_tensor_list.append(s_label)
          simage_rgbtensor = torch.stack(simage_tensor_list,dim=0)
          simage_labeltensor = torch.stack(simage_label_tensor_list,dim=0)
          return simage_rgbtensor, simage_labeltensor#simage_tensor_list, simage_label_tensor_list
              

      
      def get_support_and_query_list(self, support_path, query_path):

         s_image_dir = support_path[0]
         s_label_dir = support_path[1]

         q_image_dir = query_path[0]
         q_label_dir = query_path[1]

         if not os.path.exists(s_label_dir):
              raise IOError("No such training direcotry exist!")
         match_str=os.path.join(s_label_dir, "*.png")
         print(match_str)
         simage_label_path_list=[]
         s_image_list=[]
         simage_label_path_list.extend(glob.glob(match_str))
         

         for f in simage_label_path_list:                   
             image_name=os.path.splitext(f.split('/')[-1])[0]
             
             
             #print(city)
             
             #print(image_name)
             image_path=os.path.join(s_image_dir, image_name+'.png')
        
             if not os.path.exists(image_path):
                print("Image %s is not exist %s" % (image_name,s_image_dir))
                raise IOError("Please Check")

             else:
                image_record={'label_path':f,'image_path':image_path,'image_name':image_name}
                                                                                                
                s_image_list.append(image_record)
         
          

         if not os.path.exists(q_label_dir):
              raise IOError("No such training direcotry exist!")
         match_str=os.path.join(q_label_dir, "*.png")
         print(match_str)
         qimage_label_path_list=[]
         q_image_list=[]
         qimage_label_path_list.extend(glob.glob(match_str))
         print(len(qimage_label_path_list))
    
         for f in qimage_label_path_list:                   
             image_name=os.path.splitext(f.split('/')[-1])[0]
             
             #print(image_name)
             image_path=os.path.join(q_image_dir, image_name+'.png')
        
             if not os.path.exists(image_path):
                #print("Image %s is not exist %s" % (image_name,image_dir))
                raise IOError("Please Check")

             else:
                image_record={'label_path':f,'image_path':image_path,'image_name':image_name}
                                                                                                
                q_image_list.append(image_record)


         return q_image_list, s_image_list



class PASCALTest(Dataset):
      
      def __init__(self, support_path, query_path, classes = 21,transform=None,computeproto=False):
          super(PASCALTest, self).__init__()
          self.support_path = support_path
          #self.train_size = train_size
          self.query_path = query_path
          #self.use_val = use_val
          self.num_classes = classes
          self.train_label_list = [i for i in range(classes)]
          self.query_image_list, self.support_image_list = self.get_support_and_query_list(support_path, query_path)
          self.transform = transform
          #self.simage_tensor,self.simage_labeltensor = self.get_support_image_label_tensor(self.support_image_list)
          self.computeproto = computeproto
          if self.computeproto:
             self.file_length = len(self.support_image_list)
          else:
             self.file_length = len(self.query_image_list)


      def __len__(self):
         return self.file_length


      def __getitem__(self, index):

          if self.computeproto:

            image_path = self.support_image_list[index]['image_path']
            label_path = self.support_image_list[index]['label_path']
            image_name = self.support_image_list[index]['image_name']

          

          else:

            image_path = self.query_image_list[index]['image_path']
            label_path = self.query_image_list[index]['label_path']
            image_name = self.query_image_list[index]['image_name']

          image = cv2.imread(image_path, cv2.IMREAD_COLOR) 
          image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  

          image = np.float32(image)
          #q_label = cv2.imread(q_label_path, cv2.IMREAD_GRAYSCALE)  
          label = np.array(Image.open(label_path))
          print(np.unique(label))
          if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Query Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
          if self.transform:
            image, label = self.transform(image, label)
          
          #print(image_name)

          return image, label, self.train_label_list, image_name
     


      def get_image_and_label_list_from_txt(self, label_dir,image_dir,txt_path):
            
        
            image_record_list=[]
            
            f=open(txt_path,'r')
            for line in f:
                image_name=line.strip('\n')#去掉TXT每行末尾的回车
                image_path=os.path.join(image_dir,image_name+'.jpg')
                label_path=os.path.join(label_dir,image_name+'.png')
                if not os.path.exists(image_path) or not os.path.exists(label_path):
                    print("Image %s is not exist in %s or label is not exist in %s" % (image_name,image_dir,label_dir))
                    raise IOError("Please Check")

                else:
                    image_record={'label_path':label_path,'image_path':image_path,'image_name':image_name}#将每张图片对应的真值路径、图像路径、图像名称放到一个字典中
                                                                                                    #这样做的好处是在于，可以对返回的字典列表随机乱序。而不会破坏图片真值之间的对应关系         
                    image_record_list.append(image_record)
            
                
            
            return image_record_list
                

      
      def get_support_and_query_list(self, support_path, query_path):

         s_image_dir = support_path[0]
         s_label_dir = support_path[1]
         s_image_txt = support_path[2]

         q_image_dir = query_path[0]
         q_label_dir = query_path[1]
         q_image_txt = query_path[2]

         s_image_list = self.get_image_and_label_list_from_txt(s_label_dir,s_image_dir,s_image_txt)
         q_image_list = self.get_image_and_label_list_from_txt(q_label_dir,q_image_dir,q_image_txt)

        


         return q_image_list, s_image_list






class SythSource(Dataset):
      
      def __init__(self, train_path,  support_path, train_size=None, transform=None, use_val=False):
          super(SythSource, self).__init__()
          self.train_path = train_path
          self.train_size = train_size
          self.transform = transform
          self.cityvalidlabel=[7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33]# [7,8,11,12,13,17,19,20,21,23,24,25,26,28,32,33] # 
          self.sythvalidlabel= [3,4,2,21,5,7,15,9,6,16,1,10,17,8,18,19,20,12,11]# [3,4,2,21,5,7,15,9,6,1,10,17,8,19,12,11]#
          self.train_list = self.get_train_list(train_path)
          #self.support_list = self.get_support_list(support_path)
          #self.transform = transforms.Compose(transforms_)
          self.dataset_size = len(self.train_list)
          #self.support_dataset_size = len (self.support_list )

      def __len__(self):
         return self.dataset_size


      def __getitem__(self, index):

          q_label_list =[]
          index1 = index
          if  True: #while len(list (set(q_label_list) & set(self.sythvalidlabel))) <= 1:#some images in GTA5 contains only one valid label, we ignore these images during training
                q_image_path = self.train_list[index1]['image_path']
                q_label_path = self.train_list[index1]['label_path']
                q_image_name = self.train_list[index1]['image_name']
                q_image = cv2.imread(q_image_path, cv2.IMREAD_COLOR) 
                q_image = cv2.cvtColor(q_image, cv2.COLOR_BGR2RGB)  

                q_label =  np.uint8 (cv2.imread(q_label_path, cv2.IMREAD_UNCHANGED)[:,:,2])# sythisa dataset use unit16 as label
                
         
           
                #q_label = cv2.imread(q_label_path, cv2.IMREAD_GRAYSCALE)   np.array(Image.open(q_label_path))
              
               
          
                q_image = np.float32(q_image)
                #  
               
                #print(q_image.shape,q_label.shape)
                q_label_list = np.unique(q_label).tolist()
                #print(q_label_list)
                index1 = index1 +random.randint(1,5)


          if q_image.shape[0] != q_label.shape[0] or q_image.shape[1] != q_label.shape[1]:

                raise (RuntimeError("Query Image & label shape mismatch: " + q_image_path + " " + q_label_path + "\n"))
          
          #perform data augmentation on query images
          if self.transform:
                q_image,q_label = self.transform(q_image,q_label)
          #q_label_temp = q_label.cpu().numpy()
          #q_label_temp = cv2.resize(q_label_temp, dsize=(int(q_label.shape[0]/8), int(q_label.shape[1]/8)),interpolation=cv2.INTER_NEAREST) 
          q_label_list_o = np.unique(q_label).tolist()#
          q_label_list =  list (set(q_label_list_o) & set(self.sythvalidlabel))
               
          for i in q_label_list_o:
                 if i not in q_label_list:         # replace all invalid label to 255  
                   # print('qqqqqqq',i) 
                    pixelind_i = np.where(q_label == i)
                    q_label[pixelind_i[0],pixelind_i[1]] = 255         

          #print('q_label_list',  q_label_list, np.unique(q_label).tolist())          
          #print(train_label_list)
          #According to the label numbers (n) in train_label_list, formulate the few shot learning into a n-class classification problem
          #q_label_temp = q_label
         # print('q_label_listf', np.unique(q_label).tolist(),np.unique(q_label_temp).tolist())
          pixelind=[]
          for i, k in enumerate(q_label_list):
                 
                #index_ofcity = self.sythvalidlabel.index(k)
                
                #index_ofsyth = self.cityvalidlabel.index(i)# get the index  of  co
                pixelind_i = np.where(q_label==k)
                
                #print(i, index_ofcity, len(pixelind_i[0]))
                pixelind.append(pixelind_i)
                #print(pixelind_i.shape)
          for i, k in enumerate(q_label_list):     
                q_label[pixelind[i][0],pixelind[i][1]] = self.sythvalidlabel.index(k)#index_ofcity
          #print('q_label_listf', np.unique(q_label).tolist())
          #print(pixelind_i.shape)
          return q_image,q_label, q_image_name#train_label_list

     
      def syth2city(self, labellist):
           result = []
           for i in labellist:
               
                    i_index =  self.sythvalidlabel.index(i)
                    result.append(self.cityvalidlabel[i_index])
           return result     


      def get_train_list(self, train_path):

         image_dir = train_path[0]
         label_dir = train_path[1]



         if not os.path.exists(label_dir):
              raise IOError("No such training direcotry exist!")
         match_str=os.path.join(label_dir,  "*.png")
         #print(match_str)
         image_label_path_list=[]
         image_record_list=[]
         image_label_path_list.extend(glob.glob(match_str))
         #print(len(image_label_path_list))
    
         for f in image_label_path_list:                   
             image=os.path.splitext(f.split('/')[-1])[0]
        
             image_name=image
            
             image_path=os.path.join(image_dir, image_name+'.png')
        
             if not os.path.exists(image_path):
                print("Image %s is not exist %s" % (image_name,image_dir))
                raise IOError("Please Check")

             else:
                image_record={'label_path':f,'image_path':image_path,'image_name':image_name}
                                                                                                
                image_record_list.append(image_record)


         #print(image_record_list)
         print (len(image_record_list))
         
         return image_record_list


      def get_support_list(self, support_path):

            image_dir = support_path[0]
            label_dir = support_path[1]
            if not os.path.exists(label_dir):
                raise IOError("No such training direcotry exist!")

            image_record_list=[]
        

            
            for image_name in fewshot_name_list:                   
                
                cityname=image_name.split('_')
                cityname=cityname[0]
                #print(city)
                
                #print(image_name)
                label_path = os.path.join(label_dir, cityname, image_name+'_gtFine_labelIds.png')
                image_path=os.path.join(image_dir, cityname, image_name+'_leftImg8bit.png')
            
                if not os.path.exists(image_path):
                    #print("Image %s is not exist %s" % (image_name,image_dir))
                    raise IOError("Please Check")

                else:
                    image_record={'label_path':label_path,'image_path':image_path,'image_name':image_name}
                                                                                                    
                image_record_list.append(image_record)
           
            print ('the length of support images',len(image_record_list))
            return image_record_list



class SythTest(Dataset):
      
      def __init__(self, train_path,   transform=None, use_val=False):
          super(SythTest, self).__init__()
          self.train_path = train_path
          #self.train_size = train_size
          self.transform = transform
          self.cityvalidlabel=[7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33]# [7,8,11,12,13,17,19,20,21,23,24,25,26,28,32,33] # 
          self.sythvalidlabel= [3,4,2,21,5,7,15,9,6,16,1,10,17,8,18,19,20,12,11]# [3,4,2,21,5,7,15,9,6,1,10,17,8,19,12,11]#
          self.train_list = self.get_train_list(train_path)
          #self.support_list = self.get_support_list(support_path)
          #self.transform = transforms.Compose(transforms_)
          self.dataset_size = len(self.train_list)
          #self.support_dataset_size = len (self.support_list )

      def __len__(self):
         return self.dataset_size


      def __getitem__(self, index):

          q_label_list =[]
          index1 = index
          if  True: #while len(list (set(q_label_list) & set(self.sythvalidlabel))) <= 1:#some images in GTA5 contains only one valid label, we ignore these images during training
                q_image_path = self.train_list[index1]['image_path']
                q_label_path = self.train_list[index1]['label_path']
                q_image_name = self.train_list[index1]['image_name']
                q_image = cv2.imread(q_image_path, cv2.IMREAD_COLOR) 
                q_image = cv2.cvtColor(q_image, cv2.COLOR_BGR2RGB)  

                q_label =  np.uint8 (cv2.imread(q_label_path, cv2.IMREAD_UNCHANGED)[:,:,2])# sythisa dataset use unit16 as label
                
         
           
                #q_label = cv2.imread(q_label_path, cv2.IMREAD_GRAYSCALE)   np.array(Image.open(q_label_path))
              
               
          
                q_image = np.float32(q_image)
                #  
               
                #print(q_image.shape,q_label.shape)
                q_label_list = np.unique(q_label).tolist()
                #print(q_label_list)
                index1 = index1 +random.randint(1,5)


          if q_image.shape[0] != q_label.shape[0] or q_image.shape[1] != q_label.shape[1]:

                raise (RuntimeError("Query Image & label shape mismatch: " + q_image_path + " " + q_label_path + "\n"))
          
          #perform data augmentation on query images
          if self.transform:
                q_image,q_label = self.transform(q_image,q_label)
          #q_label_temp = q_label.cpu().numpy()
          #q_label_temp = cv2.resize(q_label_temp, dsize=(int(q_label.shape[0]/8), int(q_label.shape[1]/8)),interpolation=cv2.INTER_NEAREST) 
          q_label_list_o = np.unique(q_label).tolist()#
          q_label_list =  list (set(q_label_list_o) & set(self.sythvalidlabel))
               
          for i in q_label_list_o:
                 if i not in q_label_list:         # replace all invalid label to 255  
                   # print('qqqqqqq',i) 
                    pixelind_i = np.where(q_label == i)
                    q_label[pixelind_i[0],pixelind_i[1]] = 255         

          #print('q_label_list',  q_label_list, np.unique(q_label).tolist())          
          #print(train_label_list)
          #According to the label numbers (n) in train_label_list, formulate the few shot learning into a n-class classification problem
          #q_label_temp = q_label
         # print('q_label_listf', np.unique(q_label).tolist(),np.unique(q_label_temp).tolist())
          pixelind=[]
          for i, k in enumerate(q_label_list):
                 
                #index_ofcity = self.sythvalidlabel.index(k)
                
                #index_ofsyth = self.cityvalidlabel.index(i)# get the index  of  co
                pixelind_i = np.where(q_label==k)
                
                #print(i, index_ofcity, len(pixelind_i[0]))
                pixelind.append(pixelind_i)
                #print(pixelind_i.shape)
          for i, k in enumerate(q_label_list):     
                q_label[pixelind[i][0],pixelind[i][1]] = self.sythvalidlabel.index(k)#index_ofcity
          #print('q_label_listf', np.unique(q_label).tolist())
          #print(pixelind_i.shape)
          return q_image,q_label, q_label_list, q_image_name#train_label_list

     
      def syth2city(self, labellist):
           result = []
           for i in labellist:
               
                    i_index =  self.sythvalidlabel.index(i)
                    result.append(self.cityvalidlabel[i_index])
           return result     


      def get_train_list(self, train_path):

         image_dir = train_path[0]
         label_dir = train_path[1]



         if not os.path.exists(label_dir):
              raise IOError("No such training direcotry exist!")
         match_str=os.path.join(label_dir,  "*.png")
         #print(match_str)
         image_label_path_list=[]
         image_record_list=[]
         image_label_path_list.extend(glob.glob(match_str))
         #print(len(image_label_path_list))
    
         for f in image_label_path_list:                   
             image=os.path.splitext(f.split('/')[-1])[0]
        
             image_name=image
            
             image_path=os.path.join(image_dir, image_name+'.png')
        
             if not os.path.exists(image_path):
                print("Image %s is not exist %s" % (image_name,image_dir))
                raise IOError("Please Check")

             else:
                image_record={'label_path':f,'image_path':image_path,'image_name':image_name}
                                                                                                
                image_record_list.append(image_record)


         #print(image_record_list)
         print (len(image_record_list))
         
         return image_record_list


      def get_support_list(self, support_path):

            image_dir = support_path[0]
            label_dir = support_path[1]
            if not os.path.exists(label_dir):
                raise IOError("No such training direcotry exist!")

            image_record_list=[]
        

            
            for image_name in fewshot_name_list:                   
                
                cityname=image_name.split('_')
                cityname=cityname[0]
                #print(city)
                
                #print(image_name)
                label_path = os.path.join(label_dir, cityname, image_name+'_gtFine_labelIds.png')
                image_path=os.path.join(image_dir, cityname, image_name+'_leftImg8bit.png')
            
                if not os.path.exists(image_path):
                    #print("Image %s is not exist %s" % (image_name,image_dir))
                    raise IOError("Please Check")

                else:
                    image_record={'label_path':label_path,'image_path':image_path,'image_name':image_name}
                                                                                                    
                image_record_list.append(image_record)
           
            print ('the length of support images',len(image_record_list))
            return image_record_list

class CitySyth(Dataset):
      
      def __init__(self, train_path,  support_path, train_size=None, transform=None, use_val=False):
          super(CitySyth, self).__init__()
          self.train_path = train_path
          self.train_size = train_size
          self.transform = transform
          self.cityvalidlabel= [7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33]
          self.sythvalidlabel= [3,4,2,21,5,7,15,9,6,16,1,10,17,8,18,19,20,12,11]
          self.train_list = self.get_train_list(train_path)
          self.support_list = self.get_support_list(support_path)
          #self.transform = transforms.Compose(transforms_)
          self.dataset_size = len(self.train_list)
          self.support_dataset_size = len (self.support_list )

      def __len__(self):
         return self.dataset_size


      def __getitem__(self, index):

          q_label_list =[]
          index1 = index
          while len(list (set(q_label_list) & set(self.sythvalidlabel))) <= 1:#some images in GTA5 contains only one valid label, we ignore these images during training
                q_image_path = self.train_list[index1]['image_path']
                q_label_path = self.train_list[index1]['label_path']
                q_image_name = self.train_list[index1]['image_name']
                q_image = cv2.imread(q_image_path, cv2.IMREAD_COLOR) 
                q_image = cv2.cvtColor(q_image, cv2.COLOR_BGR2RGB)  

                q_label =  np.uint8 (cv2.imread(q_label_path, cv2.IMREAD_UNCHANGED)[:,:,2])# sythisa dataset use unit16 as label
                
         
           
                #q_label = cv2.imread(q_label_path, cv2.IMREAD_GRAYSCALE)   np.array(Image.open(q_label_path))
              
               
          
                q_image = np.float32(q_image)
                #  
               
                #print(q_image.shape,q_label.shape)
                q_label_list = np.unique(q_label).tolist()
                #print(q_label_list)
                index1 = index1 +random.randint(1,5)


          if q_image.shape[0] != q_label.shape[0] or q_image.shape[1] != q_label.shape[1]:

                raise (RuntimeError("Query Image & label shape mismatch: " + q_image_path + " " + q_label_path + "\n"))
          
          #perform data augmentation on query images
          if self.transform:
                q_image,q_label = self.transform(q_image,q_label)
          #q_label_temp = q_label.cpu().numpy()
          #q_label_temp = cv2.resize(q_label_temp, dsize=(int(q_label.shape[0]/8), int(q_label.shape[1]/8)),interpolation=cv2.INTER_NEAREST) 
          q_label_list_o = np.unique(q_label).tolist()#
          q_label_list =  list (set(q_label_list_o) & set(self.sythvalidlabel))
          #print('q_label_list_o',q_label_list_o)
         # q_label_list_in_city = self.syth2city(q_label_list)
          #print(q_label_list_in_city )

          train_label_list= []
          while len(train_label_list) <= 1: #query image and support image must contain at least 2 same labels
                s_image_idx = random.randint(1,self.support_dataset_size)-1
                s_image_path = self.support_list[s_image_idx]['image_path']
                s_label_path = self.support_list[s_image_idx ]['label_path']

                # while(s_image_path == q_image_path):#support image and query image must be different 
                #    s_image_idx = random.randint(1,self.dataset_size)-1
                #    s_image_path = self.train_list[s_image_idx ]['image_path']
                #    s_label_path = self.train_list[s_image_idx ]['label_path']
                #end
               #print(q_image_path)
               #print(s_image_path)

                s_image = cv2.imread(s_image_path, cv2.IMREAD_COLOR) 
                s_image = cv2.cvtColor(s_image, cv2.COLOR_BGR2RGB)  

                s_image = np.float32(s_image)
                s_label = np.array(Image.open(s_label_path))#cv2.imread(s_label_path, cv2.IMREAD_GRAYSCALE)
                if self.transform:
                    s_image,s_label = self.transform(s_image,s_label)
            
                # s_label_temp = s_label
                # s_label_temp = F.interpolate( s_label_temp.unsqueeze(0).unsqueeze(0).float, size=(129, 129), mode='nearest')
                # print(s_label_temp.shape) 
                # print(np.unique(s_label_temp).tolist())
                s_label_list = np.unique(s_label).tolist()
                # print(s_label_list)
                #qscommon_label = list(set(q_label_list) & set(s_label_list))
                        
                #if 0 in train_label:
                #  train_label.remove(0)#0 is the ignored label
                #print(train_label)

                train_label_list = list (set(s_label_list) & set(self.cityvalidlabel))#we only consider label that contained in 19 labels of cityscape dataset
                #print(len(train_label_list))

           
                train_label_list_temp = train_label_list
                #poptimes = 0
                for k, c in enumerate(train_label_list):#range(n_classes):
                #print('ccccccccccccccccccc',c,train_label_list)
                #print(k,c)
                    mask = (s_label[:,:] == c).float().unsqueeze(0).unsqueeze(0)

                    mask = F.interpolate(mask, size=(129,129), mode='bilinear', align_corners=True)
                    #print('mask.shape after interpolation',np.unique(mask.cpu().numpy()))# 降采样后 MASK中有些小目标可能干没了
                    #qmask = (q_label[:,:] == c).float().unsqueeze(0).unsqueeze(0)
                    #qmask = F.interpolate(qmask, size=(129,129), mode='bilinear', align_corners=True)    

                    if 1 in np.unique(mask.cpu().numpy()) : #in general, an image may not contribute prototype to all the classes
                        #if 1 in np.unique(qmask.cpu().numpy()) :
                            pass
                        #else:
                            #print('in data loading, query disapper', k,c)
                         #   train_label_list_temp.remove(c)#pop(k-poptimes)
                            #poptimes = poptimes+1
                    else:
                         print('in data loading, support disapper', k,c)
                         train_label_list_temp.remove(c)#pop(k-poptimes)
                         #poptimes = poptimes+1
                    #print('end for ')
                    
          
                        
                train_label_list = train_label_list_temp
                if (len(train_label_list)<=1):
                    print(q_image_path)
                    print(s_image_path)
            # end of  while

            # labels contained both in support image and query image will be used for few-shot learning, other labels are set to invalid 
          for i in s_label_list:
                if i not in train_label_list:
                    pixelind_i = np.where(s_label == i)
                    s_label[pixelind_i[0],pixelind_i[1]] = 255  

          
          for i in q_label_list_o:
                 if i not in q_label_list:         # replace all invalid label to 255   
                    pixelind_i = np.where(q_label == i)
                    q_label[pixelind_i[0],pixelind_i[1]] = 255         
                 else:     
                    index_ofcity =  self.sythvalidlabel.index(i)

                    if self.cityvalidlabel[index_ofcity] not in train_label_list:     # also replace label that not in support image to 255
                        pixelind_i = np.where(q_label == i)
                        q_label[pixelind_i[0],pixelind_i[1]] = 255     
          #print('q_label_list', q_label_list)          
          #print(train_label_list)
          #According to the label numbers (n) in train_label_list, formulate the few shot learning into a n-class classification problem
          pixelind_s=[]
          pixelind_q=[]
          for i, k in enumerate(train_label_list):
                #print(i,train_label_list.index(i))
                pixelind_i = np.where(s_label==k)
                pixelind_s.append(pixelind_i)
                #s_label[pixelind_i[0],pixelind_i[1]] = train_label_list.index(i) 

                index_ofsyth = self.cityvalidlabel.index(k)# get the index  of  co
                pixelind_i = np.where(q_label==self.sythvalidlabel[index_ofsyth])
                pixelind_q.append(pixelind_i)

          for i, k in enumerate(train_label_list):
               s_label[pixelind_s[i][0],pixelind_s[i][1]] = i           # train_label_list.index(i) 
               q_label[pixelind_q[i][0],pixelind_q[i][1]] =i           #train_label_list.index(i) 
          #print(np.unique(q_label))
          #print(q_label_path,q_label_list)
          #print(s_label_path,s_label_list)
         
          #im=Image.frombytes('L', (s_label.shape[1],s_label.shape[0]),s_label)
          #im.save('slabelsaved.png','png')
          train_label_list = [i for i in range(len(train_label_list))]
          #print(train_label_list)
          return q_image,q_label, q_image_name , s_image.unsqueeze(0),s_label.unsqueeze(0),train_label_list

     
      def syth2city(self, labellist):
           result = []
           for i in labellist:
               
                    i_index =  self.sythvalidlabel.index(i)
                    result.append(self.cityvalidlabel[i_index])
           return result     


      def get_train_list(self, train_path):

         image_dir = train_path[0]
         label_dir = train_path[1]



         if not os.path.exists(label_dir):
              raise IOError("No such training direcotry exist!")
         match_str=os.path.join(label_dir,  "*.png")
         #print(match_str)
         image_label_path_list=[]
         image_record_list=[]
         image_label_path_list.extend(glob.glob(match_str))
         #print(len(image_label_path_list))
    
         for f in image_label_path_list:                   
             image=os.path.splitext(f.split('/')[-1])[0]
        
             image_name=image
            
             image_path=os.path.join(image_dir, image_name+'.png')
        
             if not os.path.exists(image_path):
                print("Image %s is not exist %s" % (image_name,image_dir))
                raise IOError("Please Check")

             else:
                image_record={'label_path':f,'image_path':image_path,'image_name':image_name}
                                                                                                
                image_record_list.append(image_record)


         #print(image_record_list)
         print (len(image_record_list))
         
         return image_record_list


      def get_support_list(self, support_path):

            image_dir = support_path[0]
            label_dir = support_path[1]
            if not os.path.exists(label_dir):
                raise IOError("No such training direcotry exist!")

            image_record_list=[]
        

            
            for image_name in fewshot_name_list:                   
                
                cityname=image_name.split('_')
                cityname=cityname[0]
                #print(city)
                
                #print(image_name)
                label_path = os.path.join(label_dir, cityname, image_name+'_gtFine_labelIds.png')
                image_path=os.path.join(image_dir, cityname, image_name+'_leftImg8bit.png')
            
                if not os.path.exists(image_path):
                    #print("Image %s is not exist %s" % (image_name,image_dir))
                    raise IOError("Please Check")

                else:
                    image_record={'label_path':label_path,'image_path':image_path,'image_name':image_name}
                                                                                                    
                image_record_list.append(image_record)
           
            print ('the length of support images',len(image_record_list))
            return image_record_list


train_path = ['/home/wjw/yzg/RAND_CITYSCAPES/RGB',
                  '/home/wjw/yzg/RAND_CITYSCAPES/GT/LABELS']

if __name__ == '__main__':
  train_path = ['/home/wjw/yzg/RAND_CITYSCAPES/RGB',
                  '/home/wjw/yzg/RAND_CITYSCAPES/GT/LABELS']
  support_path =  ['/home/wjw/yzg/Cityscapes/leftImg8bit/train',
                                        '/home/wjw/yzg/Cityscapes/gtFine_trainvaltest/gtFine/train']                  
  train_data = CitySyth(train_path,support_path)