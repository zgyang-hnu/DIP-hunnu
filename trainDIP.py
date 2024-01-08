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
from PIL import Image
from model.PFENet import FEWDomain   
from util import dataset
from util import transform, config
from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU, intersectionAndUnion

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

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



def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    #parser.add_argument : 添加可通过命令行输入的参数，并可设置默认参数（即命令行没有新的输入用默认的）
    parser.add_argument('--config', type=str, default='config/pascal/pascal_split0_resnet50.yaml', help='config file')
    parser.add_argument('opts', help='see config/ade20k/ade20k_pspnet50.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)#从yaml文件中 读取配置参数
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
    args = get_parser()#获取程序运行所需的一系列参数，定义在yaml文件中
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
    
    ### multi-processing training is deprecated
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False    # sync_bn is deprecated 
        args.distributed = False
        args.multiprocessing_distributed = False
    if args.multiprocessing_distributed:
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args
    args = argss

    BatchNorm = nn.BatchNorm2d

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

    model = FEWDomain(layers=args.layers, classes=2, zoom_factor=8, \
        criterion=nn.CrossEntropyLoss(ignore_index=255), BatchNorm=BatchNorm, \
        pretrained=True, shot=args.shot, ppm_scales=args.ppm_scales, vgg=args.vgg)

    for param in model.layer0.parameters():
        param.requires_grad = False
    for param in model.layer1.parameters():
        param.requires_grad = False
    for param in model.layer2.parameters():
        param.requires_grad = True
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True
        
    optimizer = torch.optim.SGD(
        [
        #{'params': model.deeplabv1head.parameters()},
        {'params': model.down.parameters()},
       # {'params': model.aspp.parameters()},
        {'params': model.ppm.parameters()},
        {'params': model.projector.parameters()},
        #{'params': model.layer0.parameters()},
       #{'params': model.layer1.parameters()},
        {'params': model.layer2.parameters()},
        {'params': model.layer3.parameters()},
         {'params': model.layer4.parameters()}],
        #{'params': model.res2.parameters()},        
        #{'params': model.cls.parameters()}],
        lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay) 

    global logger, writer
    logger = get_logger()
    writer = SummaryWriter(args.save_path)
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))
    logger.info(model)
    print(args)

    #model = torch.nn.DataParallel(model.cuda())
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

    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))


    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    assert args.split in [0, 1, 2, 3, 999]
    #下面的transform大类 是作者自己定义的 不是pytorch系统里的 基本上 还是通过cv2 工具包来实现的
    train_transform = [
        transform.RandScale([args.scale_min, args.scale_max]),
        transform.RandRotate([args.rotate_min, args.rotate_max], padding=mean, ignore_label=args.padding_label),
        transform.RandomGaussianBlur(),
        transform.RandomHorizontalFlip(),
        transform.Crop([args.train_h, args.train_w], crop_type='rand', padding=mean, ignore_label=args.padding_label),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)]
    train_transform = transform.Compose(train_transform)
    #train_data = dataset.SemData(split=args.split, shot=args.shot, data_root=args.data_root, \
    #                            data_list=args.train_list, transform=train_transform, mode='train', \
    #                            use_coco=args.use_coco, use_split_coco=args.use_split_coco)
    train_pathsyth = ['/home/vision/yzg/RAND_CITYSCAPES/RGB',
                  '/home/vision/yzg/RAND_CITYSCAPES/GT/LABELS']
    train_path = ['/home/vision/yzg/GTA5/images',
                  '/home/vision/yzg/GTA5/labels']
    train_pathbdd = ['/home/vision/yzg/bdd/bdd100k/seg/images/train',
                  '/home/vision/yzg/bdd/bdd100k/seg/labels/train']               
    support_path =  ['/home/vision/yzg/Cityscapes/leftImg8bit/train',
                                        '/home/vision/yzg/Cityscapes/gtFine_trainvaltest/gtFine/train']           
    #train_data = dataset.GTA5(train_path,[1024,1024],transform=train_transform)
   # train_data = dataset.CityGTA5NEW(train_path,support_path,[1024,1024],transform=train_transform)
    train_data = dataset.City2BDD(support_path,train_pathbdd,[1024,1024],transform=train_transform)
    #train_data = dataset.BDD2City(train_pathbdd,support_path,[1024,1024],transform=train_transform)
    #train_data = dataset.GTA5Source(train_path,support_path,transform=train_transform)
    #train_data = dataset.BDDSource(train_pathbdd,support_path,transform=train_transform)
    #train_data = dataset.CitySource(support_path,train_pathbdd,transform=train_transform)
    #train_data = dataset.CitySyth(train_pathsyth,support_path,transform=train_transform)
    #train_data = dataset.SythSource(train_pathsyth,support_path,transform=train_transform)
    #train_data = dataset.CityFEW(support_path,[1024,2048],transform=train_transform)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    if args.evaluate:
        if args.resized_val:
            val_transform = transform.Compose([
                transform.Resize(size=args.val_size),
                transform.ToTensor(),
                transform.Normalize(mean=mean, std=std)])    
        else:
            val_transform = transform.Compose([
                transform.test_Resize(size=args.val_size),
                transform.ToTensor(),
                transform.Normalize(mean=mean, std=std)])           
        val_data = dataset.SemData(split=args.split, shot=args.shot, data_root=args.data_root, \
                                data_list=args.val_list, transform=val_transform, mode='val', \
                                use_coco=args.use_coco, use_split_coco=args.use_split_coco)
        val_sampler = None
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    max_iou = 0.
    filename = 'PFENet.pth'

    for epoch in range(args.start_epoch, args.epochs):
        if args.fix_random_seed_val:
            torch.cuda.manual_seed(args.manual_seed + epoch)
            np.random.seed(args.manual_seed + epoch)
            torch.manual_seed(args.manual_seed + epoch)
            torch.cuda.manual_seed_all(args.manual_seed + epoch)
            random.seed(args.manual_seed + epoch)
        
        epoch_log = epoch + 1
        #此处 训练 一个epoch
        loss_train, mIoU_train, mAcc_train, allAcc_train = train(train_loader, model, optimizer, epoch)

        if main_process():
            writer.add_scalar('loss_train', loss_train, epoch_log)
            writer.add_scalar('mIoU_train', mIoU_train, epoch_log)
            writer.add_scalar('mAcc_train', mAcc_train, epoch_log)
            writer.add_scalar('allAcc_train', allAcc_train, epoch_log)     

        if False:#args.evaluate and (epoch % 2 == 0 or (args.epochs<=50 and epoch%1==0)):
            loss_val, mIoU_val, mAcc_val, allAcc_val, class_miou = validate(val_loader, model, criterion)
            if main_process():
                writer.add_scalar('loss_val', loss_val, epoch_log)
                writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
                writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
                writer.add_scalar('class_miou_val', class_miou, epoch_log)
                writer.add_scalar('allAcc_val', allAcc_val, epoch_log)
            if class_miou > max_iou:
                max_iou = class_miou
                if os.path.exists(filename):
                    os.remove(filename)            
                filename = args.save_path + '/train_epoch_' + str(epoch) + '_'+str(max_iou)+'.pth'
                logger.info('Saving checkpoint to: ' + filename)
                torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)
        
        if (epoch+1) % 5==0:
           filename = args.save_path +'/epoch'+str(epoch) +'.pth'
           logger.info('Saving checkpoint to: ' + filename)
           torch.save({'epoch': args.epochs, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)
 
    filename = args.save_path + '/final.pth'
    logger.info('Saving checkpoint to: ' + filename)
    torch.save({'epoch': args.epochs, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)                


def train(train_loader, model, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    main_loss_meter = AverageMeter()
    aux_loss_meter = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    vis_key = 0
    print('Warmup: {}'.format(args.warmup))
    
    for i, (input, target, name, s_input, s_mask, labellist) in enumerate(train_loader):#, s_input, s_mask, labellist
        data_time.update(time.time() - end)
        current_iter = epoch * len(train_loader) + i + 1
        index_split = -1
        if args.base_lr > 1e-6:
            poly_learning_rate(optimizer, args.base_lr, current_iter, max_iter, power=args.power, index_split=index_split, warmup=args.warmup, warmup_step=len(train_loader)//2)

        s_input = s_input.cuda(non_blocking=True)
        s_mask = s_mask.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        #print(s_input.shape,s_mask.shape,input.shape,target.shape,labellist)
        
        #执行深度学习模型，获得输出 s_x=s_input, s_y=s_mask, train_label_list=labellist, epoch=epoch
        output, main_loss = model( x=input,  y=target,s_x=s_input, s_y=s_mask, train_label_list=labellist, epoch=epoch)
        #if main_loss>1000:
        #    print(image_name)
        if not args.multiprocessing_distributed:
            main_loss= torch.mean(main_loss)
        loss = main_loss #+ args.aux_weight * aux_loss
        optimizer.zero_grad()
        #print('loss',loss)
        loss.backward()
        optimizer.step()
        n = input.size(0)
       

        intersection, union, target1 = intersectionAndUnion(output, target, 2, args.ignore_label)
        
        #intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy() umcomment if use intersectionAndUnionGPU
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target1)
        
        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        main_loss_meter.update(main_loss.item(), n)
        #aux_loss_meter.update(aux_loss.item(), n)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()

        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'MainLoss {main_loss_meter.val:.4f} '             
                        'Loss {loss_meter.val:.4f} '
                        'Accuracy {accuracy:.4f}.'.format(epoch+1, args.epochs, i + 1, len(train_loader),
                                                          batch_time=batch_time,
                                                          data_time=data_time,
                                                          remain_time=remain_time,
                                                          main_loss_meter=main_loss_meter,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))
        if main_process():
            writer.add_scalar('loss_train_batch', main_loss_meter.val, current_iter)
            writer.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), current_iter)
            writer.add_scalar('mAcc_train_batch', np.mean(intersection / (target1 + 1e-10)), current_iter)
            writer.add_scalar('allAcc_train_batch', accuracy, current_iter)
        if (i+1) %1000 ==0:
           print('sleep for 100 s')
           time.sleep(20)
        if i %500 ==0:#save intermediate output of training process to files
             prediction=torch.squeeze(output[0])
             prediction=np.uint8(prediction.cpu().numpy())
             prediction=trainid2labelid_efficient(prediction)
             color_pred=decode_labels_cityscape(prediction)
             im=Image.fromarray(color_pred)
             spath = os.path.join('./trainsave', str(epoch)+'_'+str(i)+'.jpg')
             im.save(spath, 'jpeg')
             target=torch.squeeze(target[0])
             target=np.uint8(target.cpu().numpy())
             target=trainid2labelid_efficient(target)
             color_pred=decode_labels_cityscape(target)
             im=Image.fromarray(color_pred)
             spath = os.path.join('./trainsave', str(epoch)+'_'+str(i)+'_'+name[0]+'gt.jpg')
             im.save(spath, 'jpeg')
            #  s_mask=torch.squeeze(s_mask[0])
            #  s_mask=np.uint8(s_mask.cpu().numpy())
            #  s_mask=trainid2labelid_efficient(s_mask)
            #  color_pred=decode_labels_cityscape(s_mask)
            #  im=Image.fromarray(color_pred)
            #  spath = os.path.join('./trainsave', str(epoch)+'_'+str(i)+'_'+name[0]+'supportgt.jpg')
            #  im.save(spath, 'jpeg')
             
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    if main_process():
        logger.info('Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch, args.epochs, mIoU, mAcc, allAcc))
        for i in range(args.classes):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))        
    return main_loss_meter.avg, mIoU, mAcc, allAcc


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
    if args.split != 999:
        if args.use_coco:
            test_num = 20000
        else:
            test_num = 5000 
    else:
        test_num = len(val_loader)
    assert test_num % args.batch_size_val == 0    
    iter_num = 0
    total_time = 0
    for e in range(10):
        for i, (input, target, s_input, s_mask, subcls, ori_label) in enumerate(val_loader):
            if (iter_num-1) * args.batch_size_val >= test_num:
                break
            iter_num += 1    
            data_time.update(time.time() - end)
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            ori_label = ori_label.cuda(non_blocking=True)
            start_time = time.time()
            output = model(s_x=s_input, s_y=s_mask, x=input, y=target)
            total_time = total_time + 1
            model_time.update(time.time() - start_time)

            if args.ori_resize:
                longerside = max(ori_label.size(1), ori_label.size(2))
                backmask = torch.ones(ori_label.size(0), longerside, longerside).cuda()*255
                backmask[0, :ori_label.size(1), :ori_label.size(2)] = ori_label
                target = backmask.clone().long()

            output = F.interpolate(output, size=target.size()[1:], mode='bilinear', align_corners=True)         
            loss = criterion(output, target)    

            n = input.size(0)
            loss = torch.mean(loss)

            output = output.max(1)[1]

            intersection, union, new_target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
            intersection, union, target, new_target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy(), new_target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(new_target)
                
            subcls = subcls[0].cpu().numpy()[0]
            class_intersection_meter[(subcls-1)%split_gap] += intersection[1]
            class_union_meter[(subcls-1)%split_gap] += union[1] 

            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            loss_meter.update(loss.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if ((i + 1) % (test_num/100) == 0) and main_process():
                logger.info('Test: [{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                            'Accuracy {accuracy:.4f}.'.format(iter_num* args.batch_size_val, test_num,
                                                              data_time=data_time,
                                                              batch_time=batch_time,
                                                              loss_meter=loss_meter,
                                                              accuracy=accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    
    class_iou_class = []
    class_miou = 0
    for i in range(len(class_intersection_meter)):
        class_iou = class_intersection_meter[i]/(class_union_meter[i]+ 1e-10)
        class_iou_class.append(class_iou)
        class_miou += class_iou
    class_miou = class_miou*1.0 / len(class_intersection_meter)
    logger.info('meanIoU---Val result: mIoU {:.4f}.'.format(class_miou))
    for i in range(split_gap):
        logger.info('Class_{} Result: iou {:.4f}.'.format(i+1, class_iou_class[i]))            
    

    if main_process():
        logger.info('FBIoU---Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(args.classes):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

    print('avg inference time: {:.4f}, count: {}'.format(model_time.avg, test_num))
    return loss_meter.avg, mIoU, mAcc, allAcc, class_miou


if __name__ == '__main__':
    main()
