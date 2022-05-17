import os
import torch
import argparse
import numpy as np

from modules.gr_yolact import GraspYolact
from utils.output_utils import after_nms, nms
from utils.common_utils import ProgressBar, MakeJson, APDataObject, prep_metrics, calc_map
from config import get_config

parser = argparse.ArgumentParser(description='YOLACT COCO Evaluation')
parser.add_argument('--img_size', type=int, default=544, help='The image size for validation.')
parser.add_argument('--weight', type=str, default='weights/CoGr-GraspNet/latest_CoGrv2_212000.pth')
parser.add_argument('--traditional_nms', default=False, action='store_true', help='Whether to use traditional nms.')
parser.add_argument('--val_num', default=-1, type=int, help='The number of images for test, set to -1 for all.')
parser.add_argument('--coco_api', action='store_true', help='Whether to use cocoapi to evaluate results.')

iou_thres = [x / 100 for x in range(50, 100, 5)]
make_json = MakeJson()


norm_mean = np.array([103.94, 116.78, 123.68], dtype=np.float32)
norm_std = np.array([57.38, 57.12, 58.40], dtype=np.float32)


if __name__ == '__main__':
    args = parser.parse_args()
    
    # # net.cuda()
    # print("Loading weights...")
    
   
    
    # net.module.load_state_dict(state_dict, strict=True)
    # net.eval()
    # net = net.cuda()

    # print("Loading weights...Done!")

    weight = args.weight
    state_dict = torch.load(weight, map_location='cpu')

    correct_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    args.cfg = 'res101_graspnet'
    cfg = get_config(args, mode='val')

    net = GraspYolact(cfg)
    net.eval()
    with torch.no_grad():
        import torch.nn as nn
        net.backbone.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    net.load_weights(correct_state_dict, cfg.cuda)

    print("Load state dict... Done!")

    net = net.cuda()

    from ocid_grasp import OCIDGraspDataset
    from functools import partial
    from utils.gr_augmentation import gr_val_aug
    from utils.output_utils import gr_nms_v2, draw_lincomb, gr_post_processing, calculate_grasp_iou_match
    from utils.box_utils import crop
    from skimage.feature import peak_local_max
    from skimage.filters import gaussian
    from tqdm import tqdm


    root_dir = "/home/puzek/sdb/dataset/OCID_grasp"

    dataset = cfg.val_dataset

    with torch.no_grad():
        total_obj_num_count = np.array([0 for i in range(31)])
        total_obj_success_count = np.array([0 for i in range(31)])

        pbar = tqdm(range(len(dataset)))
        # pbar = tqdm(range(191))

        for i in pbar:
            rgbd, bboxes, rects, ins_masks, pos_masks, qua_masks, ang_masks, wid_masks = dataset[100]

            rgbd_tensor = torch.tensor(rgbd).unsqueeze(0).cuda()

            img = rgbd.transpose((1,2,0))[:, :, :3]
            depth = rgbd.transpose((1,2,0))[:, :, 3]
            
            # print(rgbd_tensor.shape)

            class_pred, box_pred, coef_pred, gr_pos_coef_pred, gr_sin_coef_pred, gr_cos_coef_pred, gr_wid_coef_pred, proto_out = net(rgbd_tensor)

            ids_p, class_p, box_p, coef_p, pos_coef_p, sin_coef_p, cos_coef_p, wid_coef_p, proto_p = gr_nms_v2(
                class_pred, box_pred, coef_pred, proto_out,
                gr_pos_coef_pred, gr_sin_coef_pred, gr_cos_coef_pred, gr_wid_coef_pred,
                net.anchors, cfg
            )

            per_object_width = [360 for i in range(cfg.num_classes)]

            img, depth, box_p, instance_masks, grasps, pos_masks, ang_masks, wid_masks, ids_p = gr_post_processing(
                img, depth, ids_p, class_p, box_p, coef_p, pos_coef_p, sin_coef_p, cos_coef_p, wid_coef_p, proto_p, ori_h=720, ori_w=1280, visualize_lincomb=False, visualize_results=False,
                num_grasp_per_object=10,
                per_object_width=per_object_width
            )

            
            # np.savez(
            #     tgt_f,
            #     ins_masks=ins_masks,
            #     pos_masks=pos_masks,
            #     qua_masks=qua_masks,
            #     ang_masks=ang_masks,
            #     wid_masks=wid_masks
            # )
            # print(ids_p)
            # print(grasps)

            tmp = []
            for obj_rects in grasps:
                tmp.extend(obj_rects)


            dataset.show_data(img, depth, box_p, ids_p, instance_masks, grasps=tmp, pos_masks=pos_masks, ang_masks=ang_masks, wid_masks=wid_masks, tgt_file="results/grasps/cogrv2_graspnet_test_results_{}.png".format(i))

            # obj_num_count, obj_success_count = calculate_grasp_iou_match(grasps, rects)
            # total_obj_num_count += np.array(obj_num_count)
            # total_obj_success_count += np.array(obj_success_count)

            break

        # print(np.array(total_obj_num_count))
        # print(np.array(total_obj_success_count))
        # print(np.array(total_obj_success_count) / np.array(total_obj_num_count))
        # class_rate = np.array(total_obj_success_count) / np.array(total_obj_num_count)
        # overrall_rate = class_rate.mean()
        # print(overrall_rate)

