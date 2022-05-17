import os
import numpy as np
import torch
import torch.distributed as dist
from functools import partial

from utils.gr_augmentation import gr_train_aug, gr_val_aug, gr_train_aug_jacquard, gr_val_aug_jacquard

os.makedirs('results/images', exist_ok=True)
os.makedirs('results/videos', exist_ok=True)
os.makedirs('results/onnx_images', exist_ok=True)
os.makedirs('results/onnx_videos', exist_ok=True)
os.makedirs('results/trt_images', exist_ok=True)
os.makedirs('results/trt_videos', exist_ok=True)
os.makedirs('weights/', exist_ok=True)
os.makedirs('onnx_files/', exist_ok=True)
os.makedirs('trt_files/', exist_ok=True)
os.makedirs('tensorboard_log/', exist_ok=True)

COLORS = np.array([[0, 0, 0], [244, 67, 54], [233, 30, 99], [156, 39, 176], [103, 58, 183], [100, 30, 60],
                   [63, 81, 181], [33, 150, 243], [3, 169, 244], [0, 188, 212], [20, 55, 200],
                   [0, 150, 136], [76, 175, 80], [139, 195, 74], [205, 220, 57], [70, 25, 100],
                   [255, 235, 59], [255, 193, 7], [255, 152, 0], [255, 87, 34], [90, 155, 50],
                   [121, 85, 72], [158, 158, 158], [96, 125, 139], [15, 67, 34], [98, 55, 20],
                   [21, 82, 172], [58, 128, 255], [196, 125, 39], [75, 27, 134], [90, 125, 120],
                   [121, 82, 7], [158, 58, 8], [96, 25, 9], [115, 7, 234], [8, 155, 220],
                   [221, 25, 72], [188, 58, 158], [56, 175, 19], [215, 67, 64], [198, 75, 20],
                   [62, 185, 22], [108, 70, 58], [160, 225, 39], [95, 60, 144], [78, 155, 120],
                   [101, 25, 142], [48, 198, 28], [96, 225, 200], [150, 167, 134], [18, 185, 90],
                   [21, 145, 172], [98, 68, 78], [196, 105, 19], [215, 67, 84], [130, 115, 170],
                   [255, 0, 255], [255, 255, 0], [196, 185, 10], [95, 167, 234], [18, 25, 190],
                   [0, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], [155, 0, 0],
                   [0, 155, 0], [0, 0, 155], [46, 22, 130], [255, 0, 155], [155, 0, 255],
                   [255, 155, 0], [155, 255, 0], [0, 155, 255], [0, 255, 155], [18, 5, 40],
                   [120, 120, 255], [255, 58, 30], [60, 45, 60], [75, 27, 244], [128, 25, 70], [128, 25, 70],
                   [128, 25, 70], [128, 25, 70], [128, 25, 70], [128, 25, 70], [128, 25, 70], [128, 25, 70]], dtype='uint8')

# 7 classes per row
COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
                'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
                'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
                'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
                'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
                'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush')

PASCAL_CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                  'bus', 'car', 'cat', 'chair', 'cow',
                  'diningtable', 'dog', 'horse', 'motorbike', 'person',
                  'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

CUSTOM_CLASSES = ('dog', 'person', 'bear', 'sheep')

COCO_LABEL_MAP = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8,
                  9: 9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16,
                  18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24,
                  27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32,
                  37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40,
                  46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48,
                  54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56,
                  62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64,
                  74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72,
                  82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}

CORNELL_LABEL_MAP = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 9: 8, 11: 9, 12: 10,
                     13: 11, 14: 12, 15: 13, 16: 14, 18: 15, 19: 16, 20: 17, 21: 18,
                     22: 19, 23: 20, 24: 21, 25: 22, 26: 23, 27: 24, 28: 25, 29: 26,
                     30: 27, 31: 28, 32: 29, 33: 30, 34: 31, 35: 32, 36: 33, 37: 34,
                     38: 35, 39: 36, 40: 37, 41: 38, 42: 39, 43: 40, 44: 41, 45: 42,
                     46: 43, 47: 44, 48: 45, 49: 46, 50: 47, 51: 48, 52: 49, 53: 50,
                     54: 51, 55: 52, 56: 53, 57: 54, 58: 55, 59: 56, 60: 57, 61: 58,
                     62: 59, 63: 60, 64: 61, 65: 62, 66: 63}


CORNELL_CLASSES = ['background', 'remote control', 'stapler', 'glasses', 'shoes', 'flashlight', 'mouse', 'tape', 'food box', 'comb', 'shampoo', 'toothbrush', 'paste', 'camera', 'soap', 'shaver', 'plate', 'scissor', 'calculator', 'card box', 'bag', 'bottle', 'mobile phone', 'cup', 'mug', 'bowl', 'headset', 'marker', 'orange', 'can', 'power adapter', 'spoon', 'clamp', 'opener', 'bag clip', 'rolling pin', 'screwdriver', 'book', 'slipper', 'snacks', 'stirrer', 'umbrella', 'brush', 'ball', 'lamp', 'corn', 'onion', 'tomato', 'apple', 'lemon', 'mango', 'pepper', 'banana', 'potato', 'cucumber', 'kiwi', 'hat', 'lock', 'candy', 'unknow', 'seasoning', 'toy', 'glue', 'ropes']


GRASPNET_DATASET = ['background', 'cracker box', 'sugar box', 'tomato soup can', 'mustard bottle', 'potted meat can', 'banana', 'bowl', 'mug', 'power drill',
                    'scissors', 'chips can', 'strawberry', 'apple', 'lemon', 'peach', 'pear', 'orange', 'plum', 'knife', 'phillips screw driver',
                    'flat screwdriver', 'racquetball', 'b cups', 'd cups', 'toy a', 'toy c', 'toy d', 'toy f',
                    'toy h', 'toy i', 'toy j', 'toy k', 'padlock', 'dragon', 'secret', 'cleansing foam', 'wash soup', 'skincare mouth rinse',
                    'dabao sod', 'soap box', 'kispa cleanser', 'tooth paste', 'nivea', 'marker', 'hosjam', 'pitcher cap', 'dish', 'white mouse', 'camel',
                    'deer', 'zebra', 'big elephant', 'rhino', 'small elephant', 'monkey', 'giraffe', 'gorilla', 'weiquan', 'charlie box', 'soap', 'black mouse',
                    'dabao face wash', 'pantene', 'head shoulder supreme', 'thera med', 'dove', 'head shoulder care', 'lion', 'coconut juice', 'hippo', 'tape',
                    'rubiks cube', 'peeler cover', 'peeler', 'ice cube mould', 'bar clamp', 'climbing hold', 'endstop holder', 'gear box', 'mount1', 'mount2', 'nozzle',
                    'part1', 'part3', 'pawn', 'pipe connector', 'turbine housing', 'vase']


JACQUARD_DATASET = ['background', 'object']



from datasets.OCID_class_dict import cls_list as OCID_GRASP_DATASET

PER_CLASS_MAX_GRASP_WIDTH = [65, 83, 45, 64, 43, 23, 140, 62, 29, 107, 147, 70, 34, 103, 112, 118, 101, 70, 41, 51, 80, 61, 77, 74, 57, 56, 74, 42, 54, 49, 75]


# from graspnet_dataset import GraspNetDataset
from datasets.ocid_grasp import OCIDGraspDataset
from datasets.jacquard_dataset import JacquardGraspDataset


class res101_jacquard:
    def __init__(self, args):
        self.mode = args.mode
        self.cuda = args.cuda
        self.gpu_id = args.gpu_id
        assert args.img_size % 32 == 0, f'Img_size must be divisible by 32, got {args.img_size}.'
        self.img_size = args.img_size
        self.class_names = JACQUARD_DATASET
        self.num_classes = len(JACQUARD_DATASET)
        # self.class_names = CORNELL_CLASSES
        # self.num_classes = len(CORNELL_CLASSES)
        self.continuous_id = COCO_LABEL_MAP
        self.scales = [int(self.img_size / 544 * aa) for aa in (24, 48, 96, 192, 384)]
        self.aspect_ratios = [1, 1 / 2, 2]


        self.summary_comment = "JACQUARD"
        self.weight_dir = "CoGr-JACQUARD"
        os.makedirs('weights/{}'.format(self.weight_dir), exist_ok=True)

        self.dataset = JacquardGraspDataset(
                        "/home/puzek/sdb/dataset/JACQUARD/jacquard",
                        origin_size=1024, 
                        target_size=544, 
                        transform=gr_train_aug_jacquard,
                        multi_obj_aug=True,
                        mode="train"
                    )
        self.val_dataset = JacquardGraspDataset(
                        "/home/puzek/sdb/dataset/JACQUARD/jacquard",
                        origin_size=1024, 
                        target_size=544, 
                        transform=gr_val_aug_jacquard,
                        multi_obj_aug=False,
                        mode="test"
                    )

        if self.mode == 'train':
            self.weight = args.resume if args.resume else 'weights/backbone_res101.pth'
        else:
            self.weight = args.weight


        if self.mode == 'train':
            self.train_bs = args.train_bs
            self.bs_per_gpu = args.bs_per_gpu
            self.val_interval = args.val_interval

            self.bs_factor = self.train_bs / 8
            self.lr = 0.001 * self.bs_factor
            self.warmup_init = self.lr * 0.1
            self.warmup_until = 500  # If adapted with bs_factor, inifinte loss may appear.
            self.lr_steps = tuple([int(aa / self.bs_factor) for aa in (0, 280000, 560000, 620000, 680000)])

            self.pos_iou_thre = 0.5
            self.neg_iou_thre = 0.4

            self.conf_alpha = 1
            self.bbox_alpha = 2.5
            self.mask_alpha = 6.125
            self.grasp_alpha = 6.125
            self.semantic_alpha = 1

            # The max number of masks to train for one image.
            self.masks_to_train = 100

        if self.mode in ('train', 'val'):
            self.val_bs = 1
            self.val_num = args.val_num
            self.coco_api = args.coco_api

        self.traditional_nms = args.traditional_nms
        self.nms_score_thre = 0.05
        self.nms_iou_thre = 0.5
        self.top_k = 200
        self.max_detections = 100

        if self.mode == 'detect':
            for k, v in vars(args).items():
                self.__setattr__(k, v)

    def print_cfg(self):
        print()
        print('-' * 30 + self.__class__.__name__ + '-' * 30)
        for k, v in vars(self).items():
            if k not in ('continuous_id', 'data_root', 'cfg'):
                print(f'{k}: {v}')
        print()


# class res101_graspnet:
#     def __init__(self, args):
#         self.mode = args.mode
#         self.cuda = args.cuda
#         self.gpu_id = args.gpu_id
#         assert args.img_size % 32 == 0, f'Img_size must be divisible by 32, got {args.img_size}.'
#         self.img_size = args.img_size
#         self.class_names = GRASPNET_DATASET
#         self.num_classes = len(GRASPNET_DATASET)
#         # self.class_names = CORNELL_CLASSES
#         # self.num_classes = len(CORNELL_CLASSES)
#         self.continuous_id = COCO_LABEL_MAP
#         self.scales = [int(self.img_size / 544 * aa) for aa in (24, 48, 96, 192, 384)]
#         self.aspect_ratios = [1, 1 / 2, 2]


#         self.summary_comment = "graspnet"
#         self.weight_dir = "CoGr-GraspNet"
#         os.makedirs('weights/{}'.format(self.weight_dir), exist_ok=True)

#         self.dataset = GraspNetDataset(
#                         "/home/puzek/sdb/dataset/graspnet",
#                         transform=partial(gr_train_aug, self.img_size),
#                         mode="train"
#                     )[:16000]
#         self.val_dataset = GraspNetDataset(
#                         "/home/puzek/sdb/dataset/graspnet",
#                         transform=partial(gr_val_aug, self.img_size),
#                         mode="test"
#                     )[16000:]

#         if self.mode == 'train':
#             self.weight = args.resume if args.resume else 'weights/backbone_res101.pth'
#         else:
#             self.weight = args.weight

#         self.data_root = '/home/hilbertxu/dataset'

#         if self.mode == 'train':
#             self.train_imgs = self.data_root + 'coco2017/train2017/'
#             self.train_ann = self.data_root + 'coco2017/annotations/instances_train2017.json'
#             self.train_bs = args.train_bs
#             self.bs_per_gpu = args.bs_per_gpu
#             self.val_interval = args.val_interval

#             self.bs_factor = self.train_bs / 8
#             self.lr = 0.001 * self.bs_factor
#             self.warmup_init = self.lr * 0.1
#             self.warmup_until = 500  # If adapted with bs_factor, inifinte loss may appear.
#             self.lr_steps = tuple([int(aa / self.bs_factor) for aa in (0, 280000, 560000, 620000, 680000)])

#             self.pos_iou_thre = 0.5
#             self.neg_iou_thre = 0.4

#             self.conf_alpha = 1
#             self.bbox_alpha = 1.5
#             self.mask_alpha = 6.125
#             self.grasp_alpha = 6.125
#             self.semantic_alpha = 1

#             # The max number of masks to train for one image.
#             self.masks_to_train = 100

#         if self.mode in ('train', 'val'):
#             self.val_imgs = self.data_root + 'coco2017/val2017/'
#             self.val_ann = self.data_root + 'coco2017/annotations/instances_val2017.json'
#             self.val_bs = 1
#             self.val_num = args.val_num
#             self.coco_api = args.coco_api

#         self.traditional_nms = args.traditional_nms
#         self.nms_score_thre = 0.05
#         self.nms_iou_thre = 0.5
#         self.top_k = 200
#         self.max_detections = 100

#         if self.mode == 'detect':
#             for k, v in vars(args).items():
#                 self.__setattr__(k, v)

#     def print_cfg(self):
#         print()
#         print('-' * 30 + self.__class__.__name__ + '-' * 30)
#         for k, v in vars(self).items():
#             if k not in ('continuous_id', 'data_root', 'cfg'):
#                 print(f'{k}: {v}')
#         print()


class res101_ocid:
    def __init__(self, args):
        self.mode = args.mode
        self.cuda = args.cuda
        self.gpu_id = args.gpu_id
        assert args.img_size % 32 == 0, f'Img_size must be divisible by 32, got {args.img_size}.'
        self.img_size = args.img_size
        self.class_names = OCID_GRASP_DATASET
        self.num_classes = len(OCID_GRASP_DATASET)
        # self.class_names = CORNELL_CLASSES
        # self.num_classes = len(CORNELL_CLASSES)
        self.continuous_id = COCO_LABEL_MAP
        self.scales = [int(self.img_size / 544 * aa) for aa in (24, 48, 96, 192, 384)]
        self.aspect_ratios = [1, 1 / 2, 2]

        if self.mode == 'train':
            self.weight = args.resume if args.resume else 'weights/backbone_res101.pth'
        else:
            self.weight = args.weight

        self.data_root = '/home/hilbertxu/dataset'

        self.summary_comment = "OCID"
        self.weight_dir = "CoGr-OCID"
        os.makedirs('weights/{}'.format(self.weight_dir), exist_ok=True)

        self.dataset = OCIDGraspDataset(
                        "/home/puzek/sdb/dataset/OCID_grasp",
                        "training_0",
                        transform=partial(gr_train_aug, self.img_size)
                    )
        self.val_dataset = OCIDGraspDataset(
                        "/home/puzek/sdb/dataset/OCID_grasp",
                        "validation_0",
                        mode='test',
                        transform=partial(gr_val_aug, self.img_size)
                    )


        if self.mode == 'train':
            self.train_imgs = self.data_root + 'coco2017/train2017/'
            self.train_ann = self.data_root + 'coco2017/annotations/instances_train2017.json'
            self.train_bs = args.train_bs
            self.bs_per_gpu = args.bs_per_gpu
            self.val_interval = args.val_interval

            self.bs_factor = self.train_bs / 8
            self.lr = 0.001 * self.bs_factor
            self.warmup_init = self.lr * 0.1
            self.warmup_until = 500  # If adapted with bs_factor, inifinte loss may appear.
            self.lr_steps = tuple([int(aa / self.bs_factor) for aa in (0, 280000, 560000, 620000, 680000)])

            self.pos_iou_thre = 0.5
            self.neg_iou_thre = 0.4

            self.conf_alpha = 1
            self.bbox_alpha = 1.5
            self.mask_alpha = 6.125
            self.grasp_alpha = 6.125
            self.semantic_alpha = 1

            # The max number of masks to train for one image.
            self.masks_to_train = 100

        if self.mode in ('train', 'val'):
            self.val_imgs = self.data_root + 'coco2017/val2017/'
            self.val_ann = self.data_root + 'coco2017/annotations/instances_val2017.json'
            self.val_bs = 1
            self.val_num = args.val_num
            self.coco_api = args.coco_api

        self.traditional_nms = args.traditional_nms
        self.nms_score_thre = 0.05
        self.nms_iou_thre = 0.5
        self.top_k = 200
        self.max_detections = 100

        if self.mode == 'detect':
            for k, v in vars(args).items():
                self.__setattr__(k, v)

    def print_cfg(self):
        print()
        print('-' * 30 + self.__class__.__name__ + '-' * 30)
        for k, v in vars(self).items():
            if k not in ('continuous_id', 'data_root', 'cfg'):
                print(f'{k}: {v}')
        print()



class res101_coco:
    def __init__(self, args):
        self.mode = args.mode
        self.cuda = args.cuda
        self.gpu_id = args.gpu_id
        assert args.img_size % 32 == 0, f'Img_size must be divisible by 32, got {args.img_size}.'
        self.img_size = args.img_size
        self.class_names = OCID_GRASP_DATASET
        self.num_classes = len(OCID_GRASP_DATASET)
        # self.class_names = CORNELL_CLASSES
        # self.num_classes = len(CORNELL_CLASSES)
        self.continuous_id = COCO_LABEL_MAP
        self.scales = [int(self.img_size / 544 * aa) for aa in (24, 48, 96, 192, 384)]
        self.aspect_ratios = [1, 1 / 2, 2]

        if self.mode == 'train':
            self.weight = args.resume if args.resume else 'weights/backbone_res101.pth'
        else:
            self.weight = args.weight

        self.data_root = '/home/hilbertxu/dataset'

        if self.mode == 'train':
            self.train_imgs = self.data_root + 'coco2017/train2017/'
            self.train_ann = self.data_root + 'coco2017/annotations/instances_train2017.json'
            self.train_bs = args.train_bs
            self.bs_per_gpu = args.bs_per_gpu
            self.val_interval = args.val_interval

            self.bs_factor = self.train_bs / 8
            self.lr = 0.001 * self.bs_factor
            self.warmup_init = self.lr * 0.1
            self.warmup_until = 500  # If adapted with bs_factor, inifinte loss may appear.
            self.lr_steps = tuple([int(aa / self.bs_factor) for aa in (0, 280000, 560000, 620000, 680000)])

            self.pos_iou_thre = 0.5
            self.neg_iou_thre = 0.4

            self.conf_alpha = 1
            self.bbox_alpha = 1.5
            self.mask_alpha = 6.125
            self.grasp_alpha = 6.125
            self.semantic_alpha = 1

            # The max number of masks to train for one image.
            self.masks_to_train = 100

        if self.mode in ('train', 'val'):
            self.val_imgs = self.data_root + 'coco2017/val2017/'
            self.val_ann = self.data_root + 'coco2017/annotations/instances_val2017.json'
            self.val_bs = 1
            self.val_num = args.val_num
            self.coco_api = args.coco_api

        self.traditional_nms = args.traditional_nms
        self.nms_score_thre = 0.05
        self.nms_iou_thre = 0.5
        self.top_k = 200
        self.max_detections = 100

        if self.mode == 'detect':
            for k, v in vars(args).items():
                self.__setattr__(k, v)

    def print_cfg(self):
        print()
        print('-' * 30 + self.__class__.__name__ + '-' * 30)
        for k, v in vars(self).items():
            if k not in ('continuous_id', 'data_root', 'cfg'):
                print(f'{k}: {v}')
        print()


class res50_coco(res101_coco):
    def __init__(self, args):
        super().__init__(args)
        if self.mode == 'train':
            self.weight = args.resume if args.resume else 'weights/backbone_res50.pth'
        else:
            self.weight = args.weight


class swin_tiny_coco(res101_coco):
    def __init__(self, args):
        super().__init__(args)
        if self.mode == 'train':
            self.weight = args.resume if args.resume else 'weights/swin_tiny.pth'
            self.lr = 0.00005 * self.bs_factor
        else:
            self.weight = args.weight


class res50_pascal(res101_coco):
    def __init__(self, args):
        super().__init__(args)
        self.class_names = PASCAL_CLASSES
        self.num_classes = len(PASCAL_CLASSES) + 1
        self.continuous_id = {(aa + 1): (aa + 1) for aa in range(self.num_classes - 1)}
        self.use_square_anchors = False
        if self.mode == 'train':
            self.weight = args.resume if args.resume else 'weights/backbone_res50.pth'
        else:
            self.weight = args.weight

        if self.mode == 'train':
            self.train_imgs = self.data_root + 'pascal_sbd/img'
            self.train_ann = self.data_root + 'pascal_sbd/pascal_sbd_train.json'
            self.lr_steps = tuple([int(aa / self.bs_factor) for aa in (0, 60000, 100000, 120000)])
            self.scales = [int(self.img_size / 544 * aa) for aa in (32, 64, 128, 256, 512)]

        if self.mode in ('train', 'val'):
            self.val_imgs = self.data_root + 'pascal_sbd/img'
            self.val_ann = self.data_root + 'pascal_sbd/pascal_sbd_val.json'


class res101_custom(res101_coco):
    def __init__(self, args):
        super().__init__(args)
        self.class_names = CUSTOM_CLASSES
        self.num_classes = len(self.class_names) + 1
        self.continuous_id = {(aa + 1): (aa + 1) for aa in range(self.num_classes - 1)}

        if self.mode == 'train':
            self.train_imgs = 'custom_dataset/'
            self.train_ann = 'custom_dataset/custom_ann.json'
            self.warmup_until = 100  # just an example
            self.lr_steps = (0, 1200, 1600, 2000)  # just an example

        if self.mode in ('train', 'val'):
            self.val_imgs = ''  # decide by yourself
            self.val_ann = ''


class res50_custom(res101_coco):
    def __init__(self, args):
        super().__init__(args)
        self.class_names = CUSTOM_CLASSES
        self.num_classes = len(self.class_names) + 1
        self.continuous_id = {(aa + 1): (aa + 1) for aa in range(self.num_classes - 1)}
        if self.mode == 'train':
            self.weight = args.resume if args.resume else 'weights/backbone_res50.pth'
        else:
            self.weight = args.weight

        if self.mode == 'train':
            self.train_imgs = 'custom_dataset/'
            self.train_ann = 'custom_dataset/custom_ann.json'
            self.warmup_until = 100  # just an example
            self.lr_steps = (0, 1200, 1600, 2000)  # just an example

        if self.mode in ('train', 'val'):
            self.val_imgs = ''  # decide by yourself
            self.val_ann = ''


def get_config(args, mode):
    args.cuda = torch.cuda.is_available()
    args.mode = mode

    if args.cuda:
        args.gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES') if os.environ.get('CUDA_VISIBLE_DEVICES') else '0'
        if args.mode == 'train':
            torch.cuda.set_device(args.local_rank)
            dist.init_process_group(backend='nccl', init_method='env://')

            # Only launched by torch.distributed.launch, 'WORLD_SIZE' can be add to environment variables.
            num_gpus = int(os.environ['WORLD_SIZE'])
            assert args.train_bs % num_gpus == 0, 'Total training batch size must be divisible by GPU number.'
            args.bs_per_gpu = int(args.train_bs / num_gpus)
        else:
            assert args.gpu_id.isdigit(), f'Only one GPU can be used in val/detect mode, got {args.gpu_id}.'
    else:
        args.gpu_id = None
        if args.mode == 'train':
            args.bs_per_gpu = args.train_bs
            print('\n-----No GPU found, training on CPU.-----')
        else:
            print('\n-----No GPU found, validate on CPU.-----')

    cfg = globals()[args.cfg](args)

    if not args.cuda or args.mode != 'train':
        cfg.print_cfg()
    elif dist.get_rank() == 0:
        cfg.print_cfg()

    return cfg
