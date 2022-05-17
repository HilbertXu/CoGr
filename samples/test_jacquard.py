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

    from ocid_grasp import OCIDGraspDataset
    from functools import partial
    from utils.gr_augmentation import gr_val_aug
    from utils.output_utils import gr_nms_v2, draw_lincomb, gr_post_processing_jacquard, calculate_grasp_iou_match
    from utils.box_utils import crop
    from skimage.feature import peak_local_max
    from skimage.filters import gaussian
    from tqdm import tqdm
    from gr_eval import evaluate_jacquard


    dataset = cfg.val_dataset
    # dataset = cfg.dataset
    

    # with torch.no_grad():
    #     class_rate, overrall_rate = evaluate_jacquard(net, dataset, cfg)
    data = []
    with open("grasps.txt", "w") as txt:
    
        with torch.no_grad():
            
            total_obj_num_count = 0
            total_obj_success_count = 0

            pbar = tqdm(range(len(dataset)))

            for idx in pbar:
                
                f, rgbd, bboxes, rects, ins_masks, pos_masks, qua_masks, ang_masks, wid_masks = dataset[idx]
                # print(f)

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

                if ids_p is None:
                    total_obj_num_count += 1
                    total_obj_success_count += 1
                    continue

                img, depth, box_p, instance_masks, grasps, pos_masks, ang_masks, wid_masks, ids_p = gr_post_processing_jacquard(
                    img, depth, ids_p, class_p, box_p, coef_p, pos_coef_p, sin_coef_p, cos_coef_p, wid_coef_p, proto_p, ori_h=1024, ori_w=1024, visualize_lincomb=False, visualize_results=False,
                    num_grasp_per_object=1
                )

                tmp = []
                for obj_rects in grasps:
                    tmp.extend(obj_rects)

                if len(tmp) == 0:
                    total_obj_num_count += 1
                    continue

                pred_rect = tmp[0]
                max_idx = 0
                max_iou = 0
                scale = 1024 / 544
                for r_idx, rect in enumerate(rects):
                    rect_gt = [rect[0]*scale, rect[1]*scale, rect[2]*scale, rect[3]*scale, rect[4], 1]

                    pred_rect[3] = rect_gt[3]
                    
                    iou = calculate_iou(pred_rect, rect_gt, shape=(1024, 1024), angle_threshold=30)

                    if iou > max_iou:
                        max_iou = iou
                        max_idx = r_idx
                    
                if max_iou > 0.25:
                    total_obj_num_count += 1
                    total_obj_success_count += 1

                    # rgb = cv2.imread(os.path.join("/home/puzek/sdb/dataset/JACQUARD/jacquard", f))

                    # gt_rect = [rects[max_idx][0]*scale, rects[max_idx][1]*scale, rects[max_idx][2]*scale, rects[max_idx][3]*scale, rects[max_idx][4], 1]
                    # center_x, center_y, width, height, theta, _ = gt_rect
                    # box = ((center_x, center_y), (width, height), -theta)
                    # box = cv2.boxPoints(box)
                    # box = np.int0(box)
                    # cv2.drawContours(rgb, [box], 0, (0,0,255), 2)


                    # center_x, center_y, width, _, theta, _ = pred_rect
                    # box = ((center_x, center_y), (width, height), -theta)
                    # box = cv2.boxPoints(box)
                    # box = np.int0(box)
                    # cv2.drawContours(rgb, [box], 0, (255,0,0), 2)

                    # pred_rect[3] = gt_rect[3]

                    # cv2.imwrite("./results/jacquard/{}_result.png".format(idx), rgb)
                    

                    # shape = (1024, 1024)
                    # center_x, center_y, w_rect, h_rect, theta, cls_id = gt_rect
                    # gt_r_rect = ((center_x, center_y), (w_rect, h_rect), -theta)
                    # gt_box = cv2.boxPoints(gt_r_rect)
                    # gt_box = np.int0(gt_box)
                    # rr1, cc1 = polygon(gt_box[:, 0], gt_box[:,1], shape)

                    # mask_rr = rr1 < shape[1]
                    # rr1 = rr1[mask_rr]
                    # cc1 = cc1[mask_rr]

                    # mask_cc = cc1 < shape[0]
                    # cc1 = cc1[mask_cc]
                    # rr1 = rr1[mask_cc]

                    # center_x, center_y, w_rect, h_rect, theta, cls_id = pred_rect
                    # p_r_rect = ((center_x, center_y), (w_rect, h_rect), -theta)
                    # p_box = cv2.boxPoints(p_r_rect)
                    # p_box = np.int0(p_box)
                    # rr2, cc2 = polygon(p_box[:, 0], p_box[:,1], shape)

                    # mask_rr = rr2 < shape[1]
                    # rr2 = rr2[mask_rr]
                    # cc2 = cc2[mask_rr]

                    # mask_cc = cc2 < shape[0]
                    # cc2 = cc2[mask_cc]
                    # rr2 = rr2[mask_cc]

                    # area = np.zeros(shape)
                    # area[cc1, rr1] += 1
                    # area[cc2, rr2] += 1

                    # cv2.imwrite("iou.png", area*177)

    

                    # union = np.sum(area > 0)
                    # intersection = np.sum(area == 2)

                else:
                    total_obj_num_count += 1
                    total_obj_success_count += 0

                # Draw rects in rgb image

                # dataset._show_data(img, depth, box_p, ids_p, instance_masks, grasps=tmp, pos_masks=pos_masks, ang_masks=ang_masks, wid_masks=wid_masks, tgt_file="./results/jacquard/{:05d}_results.png".format(idx))
                
                # dataset._show_data(img, depth, box_p, ids_p, instance_masks, grasps=tmp, pos_masks=pos_masks, ang_masks=ang_masks, wid_masks=wid_masks, tgt_file="./results/jacquard/{:05d}_results.png".format(idx))
            print(total_obj_num_count, total_obj_success_count)
            print(total_obj_success_count / total_obj_num_count)
            # 0.9047893825735719
            # 99.61.pth 0.9180611656087709
                
                # dataset._show_data(img, depth, box_p, ids_p, instance_masks, grasps=tmp, pos_masks=pos_masks, ang_masks=ang_masks, wid_masks=wid_masks, tgt_file="./results/jacquard/{:05d}_results.png".format(idx))
                # dataset._show_data(img, depth, instance_masks, box_p, tmp, pos_masks, pos_masks, ang_masks, wid_masks, tgt_file="./results/jacquard/{:05d}_results.png".format(idx))
                # data.extend(["{}\n".format(f), "{};{};{};{};{}\n".format(tmp[0][0], tmp[0][1], tmp[0][4], tmp[0][2], tmp[0][3])])

                # # obj_num_count, obj_success_count = calculate_grasp_iou_match(grasps, rects)
                # total_obj_num_count += np.array(obj_num_count)
                # total_obj_success_count += np.array(obj_success_count)
            # txt.writelines(data)
            

# @TODO
# Existing problem
# 1. How to normalize gripper width(Use the width of bbox? 
#    or the mean of (height & width) of bbox) Ongoing... Testing the distribution of width of grasp rectangles
# 2. How to represent grasp quality better(Try the smooth l1 loss on grasp quality) 
#    Finished, Calculate grasp pos loss with smooth L1 Loss + 0.01 * quality loss
# Still need to figure out why the loss of grasp position is so weird
