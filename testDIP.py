import os
import random
import time
import cv2
import numpy as np
import logging
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from tensorboardX import SummaryWriter

from model.PFENet4FEW import FEWDomain  
from util import dataset
from util import transform, config
from util.util import AverageMeter, poly_learning_rate, intersectionAndUnion
from PIL import Image

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)



label_colours_cityscape = [(0,0,0), (0,  0,  0),(  0,  0,  0),(  0,  0,  0),( 0,  0,  0),
               
                (111, 74,  0),( 81,  0, 81) ,(128, 64,128),(244, 35,232) ,(250,170,160),
                
                (230,150,140),( 70, 70, 70),(102,102,156),(190,153,153),(180,165,180),
               
                (150,100,100),(150,120, 90),(153,153,153),(153,153,153),(250,170, 30),
                
                (220,220,  0),(107,142, 35) ,(152,251,152) ,( 70,130,180),(220, 20, 60),
                (255,  0,  0),(  0,  0,142),(  0,  0, 70),(  0, 60,100),(  0,  0, 90),
                (  0,  0,110),(  0, 80,100),(  0,  0,230),(119, 11, 32)]
               
label_colours_pascal = [[0, 0, 0], [0, 0, 128], [0, 128, 0], [0, 128, 128],
                        [128, 0, 0], [128, 0, 128], [128, 128, 0],
                        [128, 128, 128],
                        [0, 0, 64], [0, 0, 192], [0, 128, 64],
                        [0, 128, 192],
                        [128, 0, 64], [128, 0, 192], [128, 128, 64],
                        [128, 128, 192], [0, 64, 0], [0, 64, 128],
                        [0, 192, 0],
                        [0, 192, 128], [128, 64, 0]]

label_list= [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]

label_listgta= [7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33]

def load_model(model, model_file, is_restore=False):
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict, strict=False)

    del state_dict 
    return model


def decode_labels_pascal(mask):
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
            outputs[i,j,:] = label_colours_pascal[mask[i,j]]
          else:
            outputs[i,j,:] = [255,255,255]  
    #outputs = np.array(img)
    return outputs





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

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/pascal/pascal_split0_resnet50.yaml', help='config file')
    parser.add_argument('opts', help='see config/ade20k/ade20k_pspnet50.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def main():
    args = get_parser()
    assert args.classes > 1
    assert args.zoom_factor in [1, 2, 4, 8]
    assert (args.train_h - 1) % 8 == 0 and (args.train_w - 1) % 8 == 0
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    if args.manual_seed is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        random.seed(args.manual_seed)
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
    if args.multiprocessing_distributed:
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        with torch.no_grad(): 
           main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args
    args = argss

    BatchNorm = nn.BatchNorm2d

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

    model = FEWDomain(layers=args.layers, classes=2, zoom_factor=8, \
        criterion=nn.CrossEntropyLoss(ignore_index=255), BatchNorm=BatchNorm, \
        pretrained=True, shot=args.shot, ppm_scales=args.ppm_scales, vgg=args.vgg)

    global logger, writer
    logger = get_logger()
    writer = SummaryWriter(args.save_path)
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))
    logger.info(model)
    print(args)

    #model = torch.nn.DataParallel(model.cuda(), device_ids=[0])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if args.weight:
        if os.path.isfile(args.weight):
            logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    assert args.split in [0, 1, 2, 3, 999]

    if args.resized_val:
        val_transform = transform.Compose([
            transform.Resize(size=args.val_size),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)])    
    else:
        val_transform = transform.Compose([
            #transform.test_Resize(size=args.val_size),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)])   
    fewshot_path = ['/opt/Cityscapes/few-shot/images',
                  '/opt/Cityscapes/few-shot/labels']
    fewshot_pathGTA = ['/opt/GTA5/few-shot/images',
                  '/opt/GTA5/few-shot/labels']

    train_path = ['/home/vision/yzg/Cityscapes/leftImg8bit/train',
                '/home/vision/yzg/Cityscapes/gtFine_trainvaltest/gtFine/train']              
    val_path = ['/home/vision/yzg/Cityscapes/leftImg8bit/val',
                '/home/vision/yzg/Cityscapes/gtFine_trainvaltest/gtFine/val']
    train_pathbdd = ['/home/vision/yzg/bdd/bdd100k/seg/images/train',
                  '/home/vision/yzg/bdd/bdd100k/seg/labels/train'] 
    val_pathbdd = ['/home/vision/yzg/bdd/bdd100k/seg/images/val',
                  '/home/vision/yzg/bdd/bdd100k/seg/labels/val'] 

    val_pathGTA = ['/opt/GTA5/images',
                '/opt/GTA5/labels']      

    train_pathsyth = ['/home/wjw/yzg/RAND_CITYSCAPES/RGB',
                  '/home/wjw/yzg/RAND_CITYSCAPES/GT/LABELS']               

    pascal_train_path = ['/opt/voc2012/JPEGImages',
                         '/opt/voc2012/SegmentationClass',
                         '/opt/voc2012/train.txt']    
    pascal_val_path = ['/opt/voc2012/JPEGImages',
                         '/opt/voc2012/SegmentationClass',
                         '/opt/voc2012/val.txt']                 
    #val_data = dataset.SythTest(train_pathsyth, transform=val_transform)
    # = dataset.CityscapesTest(train_path,val_path, transform=val_transform)
    val_data = dataset.BDDTest(train_pathbdd,val_pathbdd, transform=val_transform)
    #val_data = dataset.PASCALTest(pascal_train_path,pascal_val_path, transform=val_transform)



    val_sampler = None
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    validate(val_loader, model, criterion) 

def validate(val_loader, model, criterion):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    model_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    if args.use_coco:
        split_gap = 20
    else:
        split_gap = 5
    class_intersection_meter = [0]*split_gap
    class_union_meter = [0]*split_gap  

    if args.manual_seed is not None and args.fix_random_seed_val:
        torch.cuda.manual_seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        random.seed(args.manual_seed)

    model.eval()
    end = time.time()

    test_num = len(val_loader)
    print(test_num)
    assert test_num % args.batch_size_val == 0    
    iter_num = 0
    total_time = 0
    

    for i, (input, label,train_label_list,image_name) in enumerate(val_loader):
            if (iter_num-1) * args.batch_size_val >= test_num:
                break
            iter_num += 1    
            data_time.update(time.time() - end)
            input = input.cuda(non_blocking=True)
            #target = target.cuda(non_blocking=True)
            #ori_label = ori_label.cuda(non_blocking=True)
            start_time = time.time()
            output = model(x=input, train_label_list=train_label_list)#,, train_label_list=train_label_list
            total_time = total_time + 1
            model_time.update(time.time() - start_time)

            

            output = F.interpolate(output, size=input.size()[2:], mode='bilinear', align_corners=True)         
            #loss = criterion(output, target)    

            n = input.size(0)
            #loss = torch.mean(loss)

            output = output.max(1)[1]
            #print(output.shape)
            prediction=torch.squeeze(output)


            intersection, union, target = intersectionAndUnion(prediction, label.squeeze(0), 19, args.ignore_label)
        
        #intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy() umcomment if use intersectionAndUnionGPU
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)
       
            prediction=np.uint8(prediction.cpu().numpy())

            prediction=trainid2labelid_efficient(prediction)

       
            shape=np.shape(prediction)

            print(i,image_name[0])
            saved_path_pred = os.path.join('./citys_val_deep', image_name[0]+'.png')
            saved_path_predcolor = os.path.join('./savedcolor', image_name[0]+'.jpg')
            saved_path_predcolorgt = os.path.join('./savedcolor', image_name[0]+'gt'+'.jpg')
            im=Image.frombytes('L', (shape[1],shape[0]),prediction)
            im.save(saved_path_pred,'png') #the saved png can be used to compute the miou value with the code provided by the official  
            if True:
               color_pred=decode_labels_cityscape(prediction)
               im=Image.fromarray(color_pred)
               im.save(saved_path_predcolor,'jpeg')
               target=trainid2labelid_efficient(label.squeeze(0).numpy())
               color_pred = decode_labels_cityscape(target)
               im=Image.fromarray(color_pred)
               im.save(saved_path_predcolorgt,'jpeg')

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)   
    print('class iou',iou_class) 
    print('mean iou',mIoU)                 


    





if __name__ == '__main__':
    main()
