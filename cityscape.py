import torch
import numpy as np
import glob
import os
import time
import cv2
import random
import math

import torch.utils.data as data
import torchvision.transforms as transforms
import time

from PIL import Image




class Cityscapes(data.Dataset):
      
      def __init__(self, support_path, query_path, classes = 19):
          super(Cityscapes, self).__init__()
          self.support_path = support_path
          #self.train_size = train_size
          self.query_path = query_path
          #self.use_val = use_val
          self.num_classes = classes
          self.query_image_list, self.support_image_list, self.few_simage_list= self.get_train_list(support_path, query_path)
          #self.transform = transforms.Compose(transforms_)
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


          

          return q_image,self.support_image_list
     


      def get_train_list(self, support_path, query_path):

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
                print("Image %s is not exist %s" % (image_name,s_image_dir))
                raise IOError("Please Check")

             else:
                image_record={'label_path':f,'image_path':image_path,'image_name':image_name}
                                                                                                
                s_image_list.append(image_record)
         
         print(len(s_image_list))

         few_support_image_index_list = []

         s_image_idx = random.randint(1,len(simage_label_path_list))-1
         few_support_image_index_list.append(s_image_idx)

         s_label = np.array(Image.open(s_image_list[s_image_idx]['label_path']))
         s_label_list = np.unique(s_label).tolist()

         if 255 in s_label_list:
                s_label_list.remove(255)
         print(s_label_list)
         while (len(s_label_list)<19): 
           in_lenth = len(s_label_list)        
           s_image_idx = random.randint(1,len(simage_label_path_list))-1
           
           s_label = np.array(Image.open(s_image_list[s_image_idx]['label_path']))
           s_label_list1 = np.unique(s_label).tolist()
           print('labels contained in singe image', s_label_list1)
           if 255 in s_label_list1:
                s_label_list1.remove(255)
           
           union_label = list(set(s_label_list) | set(s_label_list1))
           
           
           if len(union_label) > in_lenth: #we only add images that contained new labels with respect to previously collected images as the new support image
              few_support_image_index_list.append(s_image_idx)
              
              s_label_list = union_label

         print(s_label_list)
         print(len(few_support_image_index_list))
         for i in range(len(few_support_image_index_list)):
           print(s_image_list[few_support_image_index_list[i]]['label_path'])


          

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
                print("Image %s is not exist %s" % (image_name,image_dir))
                raise IOError("Please Check")

             else:
                image_record={'label_path':f,'image_path':image_path,'image_name':image_name}
                                                                                                
                q_image_list.append(image_record)


         return q_image_list, s_image_list, few_support_image_index_list


class GTA(data.Dataset):
      
      def __init__(self, support_path, classes = 19):
          super(GTA, self).__init__()
          self.support_path = support_path
          #self.train_size = train_size
          
          #self.use_val = use_val
          self.num_classes = classes
          self.validlabel= [7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33]
          self.few_simage_list= self.get_train_list(support_path)
          #self.transform = transforms.Compose(transforms_)
          self.file_length = 2000

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


          

          return q_image,self.support_image_list
     


      def get_train_list(self, support_path):

         image_dir = support_path[0]
         label_dir = support_path[1]



         if not os.path.exists(label_dir):
              raise IOError("No such training direcotry exist!")
         match_str=os.path.join(label_dir,  "*.png")
         #print(match_str)
         simage_label_path_list=[]
         s_image_list=[]
         simage_label_path_list.extend(glob.glob(match_str))
         #print(len(image_label_path_list))
    
         for f in simage_label_path_list:                   
             image=os.path.splitext(f.split('/')[-1])[0]
        
             image_name=image
            
             image_path=os.path.join(image_dir, image_name+'.png')
        
             if not os.path.exists(image_path):
                print("Image %s is not exist %s" % (image_name,image_dir))
                raise IOError("Please Check")

             else:
                image_record={'label_path':f,'image_path':image_path,'image_name':image_name}
                                                                                                
                s_image_list.append(image_record)
         
         print(len(s_image_list))

         few_support_image_index_list = []

         s_image_idx = random.randint(1,len(simage_label_path_list))-1
         few_support_image_index_list.append(s_image_idx)

         s_label = np.array(Image.open(s_image_list[s_image_idx]['label_path']))
         s_label_list = np.unique(s_label).tolist()
         print('labels contained in singe image', s_label_list, s_image_list[s_image_idx]['label_path'])
         s_label_list = list(set(s_label_list) & set(self.validlabel))

         if 255 in s_label_list:
                s_label_list.remove(255)
         print(s_label_list)
         while (len(s_label_list)<19): 
           in_lenth = len(s_label_list)        
           s_image_idx = random.randint(1,len(simage_label_path_list))-1
           
           s_label = np.array(Image.open(s_image_list[s_image_idx]['label_path']))
           s_label_list1 = np.unique(s_label).tolist()
           s_label_list1 = list(set(s_label_list1) & set(self.validlabel))
           print('labels contained in singe image', s_label_list1, s_image_list[s_image_idx]['label_path'])
           if 255 in s_label_list1:
                s_label_list1.remove(255)
           
           union_label = list(set(s_label_list) | set(s_label_list1))
           
           
           if len(union_label) > in_lenth: #we only add images that contained new labels with respect to previously collected images as the new support image
              few_support_image_index_list.append(s_image_idx)
              
              s_label_list = union_label

         print(s_label_list)
         print(len(few_support_image_index_list))
         for i in range(len(few_support_image_index_list)):
           print(s_image_list[few_support_image_index_list[i]]['label_path'])


          




         return  few_support_image_index_list


#the following code is used to test the Class of Cityscape, you can igore these codes if you do not want to test it by yourself

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
      num_images: number of images to decode from the batch.
    
    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """
    h, w = mask.shape
    #assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((h, w, 3), dtype=np.uint8)
    #for i in range(num_images):
    #img = Image.new('RGB', (len(mask[0]), len(mask)))
    #pixels = img.load()
    for i in range(h):
        for j in range(w):
            if mask[i,j]< 255:
               outputs[i,j,:] = label_colours_cityscape[mask[i,j]]
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

if __name__ == "__main__":
    train_path = ['/opt/Cityscapes/leftImg8bit/train',
                  '/opt/Cityscapes/gtFine_trainvaltest/gtFine/train']
    fewshot_path = ['/opt/Cityscapes/few-shot/images',
                  '/opt/Cityscapes/few-shot/labels']
    val_path = ['/opt/Cityscapes/leftImg8bit/val',
                '/opt/Cityscapes/gtFine_trainvaltest/gtFine/val']
    train_path1 = ['/opt/GTA5/images',
                  '/opt/GTA5/labels']
    transforms_ = [transforms.RandomCrop((1024, 1024), pad_if_needed=True)]
    bd = GTA(train_path1)
    for i in range(30):
      start_time=time.time()
      out=bd.__getitem__(i)
      print(time.time()-start_time)
      print(bd.file_length)
      print(out['image'].shape)
      #k=transforms.ToPILImage()k(out['data'])#
      im=out['image'].numpy()
      print(im.shape,out['image_name'])
      im=Image.fromarray(np.uint8(im))
      im.show()
      im=Image.fromarray(decode_labels_cityscape(trainid2labelid_efficient(out['label'].numpy())))
      im.show()
      time.sleep(10)

