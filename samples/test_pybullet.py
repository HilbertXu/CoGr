import os
import cv2
import torch
import argparse
import numpy as np

from modules.gr_yolact import GraspYolact
from utils.output_utils import after_nms, nms, calculate_iou
from utils.common_utils import ProgressBar, MakeJson, APDataObject, prep_metrics, calc_map
from config import get_config
from skimage.draw import polygon
from utils.gr_augmentation import normalize_and_toRGB
from utils.output_utils import gr_nms_v2, draw_lincomb, gr_post_processing_jacquard, calculate_grasp_iou_match

parser = argparse.ArgumentParser(description='YOLACT COCO Evaluation')
parser.add_argument('--img_size', type=int, default=544, help='The image size for validation.')
parser.add_argument('--weight', type=str, default='weights/CoGr-JACQUARD/CoGrv2_99.61.pth')
# parser.add_argument('--weight', type=str, default='weights/CoGr-JACQUARD/latest_CoGrv2_340000.pth')
parser.add_argument('--traditional_nms', default=False, action='store_true', help='Whether to use traditional nms.')
parser.add_argument('--val_num', default=-1, type=int, help='The number of images for test, set to -1 for all.')
parser.add_argument('--coco_api', action='store_true', help='Whether to use cocoapi to evaluate results.')

iou_thres = [x / 100 for x in range(50, 100, 5)]
make_json = MakeJson()


norm_mean = np.array([103.94, 116.78, 123.68], dtype=np.float32)
norm_std = np.array([57.38, 57.12, 58.40], dtype=np.float32)


if __name__ == "__main__":
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

    args.cfg = 'res101_jacquard'
    cfg = get_config(args, mode='val')

    net = GraspYolact(cfg)
    net.eval()
    with torch.no_grad():
        import torch.nn as nn
        net.backbone.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    net.load_weights(correct_state_dict, cfg.cuda)

    print("Load state dict... Done!")

    net = net.cuda()
    net = net.eval()

    dataset = cfg.dataset

    idx = 3
    num_object=1
    path = "./dataset/{:04d}".format(idx)
    rgb = cv2.imread(os.path.join(path, "{}_objects_rgb.png".format(num_object)))
    dep = cv2.imread(os.path.join(path, "{}_objects_depth.png".format(num_object)), cv2.IMREAD_UNCHANGED)
    print(np.max(dep), np.min(dep), np.mean(dep))
    dep = 1 - (dep / np.max(dep))
    rgb_norm = normalize_and_toRGB(rgb)

    rgbd = np.concatenate([rgb_norm, np.expand_dims(dep, -1)], axis=-1)

    img = rgbd.transpose((1,2,0))[:, :, :3]
    depth = rgbd.transpose((1,2,0))[:, :, 3]

    print(rgbd.shape)
    with torch.no_grad():

        rgbd_tensor = torch.tensor(rgbd).float().permute(2,0,1).unsqueeze(0).cuda()

        class_pred, box_pred, coef_pred, gr_pos_coef_pred, gr_sin_coef_pred, gr_cos_coef_pred, gr_wid_coef_pred, proto_out = net(rgbd_tensor)


        ids_p, class_p, box_p, coef_p, pos_coef_p, sin_coef_p, cos_coef_p, wid_coef_p, proto_p = gr_nms_v2(
                        class_pred, box_pred, coef_pred, proto_out,
                        gr_pos_coef_pred, gr_sin_coef_pred, gr_cos_coef_pred, gr_wid_coef_pred,
                        net.anchors, cfg
                    )

        img, depth, box_p, instance_masks, grasps, pos_masks, ang_masks, wid_masks, ids_p = gr_post_processing_jacquard(
            img, depth, ids_p, class_p, box_p, coef_p, pos_coef_p, sin_coef_p, cos_coef_p, wid_coef_p, proto_p, ori_h=544, ori_w=544, visualize_lincomb=False, visualize_results=False,
            num_grasp_per_object=1
        )


        tmp = []
        for obj_rects in grasps:
            tmp.extend(obj_rects)
        dataset._show_data(rgb, dep, instance_masks, box_p, tmp, pos_masks, pos_masks, ang_masks, wid_masks, tgt_file="./results/pybullet/{:04d}_{}_results.png".format(idx, num_object))