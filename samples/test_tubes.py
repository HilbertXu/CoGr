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
parser.add_argument('--weight', type=str, default='weightsv2/old/latest_CoGrv2_340000.pth')
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

    args.cfg = 'res101_coco'
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

    dataset = OCIDGraspDataset(
        "/home/puzek/sdb/dataset/OCID_grasp",
        "validation_0",
        mode='test',
        transform=partial(gr_val_aug, 544)
    )

    with torch.no_grad():
        total_obj_num_count = np.array([0 for i in range(31)])
        total_obj_success_count = np.array([0 for i in range(31)])

        pbar = tqdm(range(len(dataset)))
        # pbar = tqdm(range(191))

        for i in pbar:
            seq_path, img_f = dataset._data[i]
            if not os.path.exists(os.path.join(root_dir, seq_path, "masks")):
                print("Making target directory: {}".format(os.path.join(root_dir, seq_path, "masks")))
                os.makedirs(os.path.join(root_dir, seq_path, "masks"))

            tgt_f = os.path.join(root_dir, seq_path, "masks", "{}.npz".format(img_f[:-4]))

            rgbd_tensor = torch.tensor(rgbd).unsqueeze(0).cuda()

            img = rgbd.transpose((1,2,0))[:, :, :3]
            depth = rgbd.transpose((1,2,0))[:, :, 3]
            
            # print(rgbd_tensor.shape)

            class_pred, box_pred, coef_pred, gr_pos_coef_pred, gr_ang_coef_pred, gr_wid_coef_pred, proto_out = net(rgbd_tensor)

            ids_p, class_p, box_p, coef_p, pos_coef_p, ang_coef_p, wid_coef_p, proto_p = gr_nms_v2(
                class_pred, box_pred, coef_pred, proto_out,
                gr_pos_coef_pred, gr_ang_coef_pred, gr_wid_coef_pred,
                net.anchors, cfg
            )

            img, depth, box_p, instance_masks, grasps, pos_masks, ang_masks, wid_masks, ids_p = gr_post_processing(
                img, depth, ids_p, class_p, box_p, coef_p, pos_coef_p, ang_coef_p, wid_coef_p, proto_p, ori_h=480, ori_w=640, visualize_lincomb=True, visualize_results=True
            )

            # print(ids_p)
            # print(grasps)
            # tmp = []
            # for obj_rects in grasps:
            #     tmp.extend(obj_rects)

            # np.savez(
            #     tgt_f,
            #     ins_masks=ins_masks,
            #     pos_masks=pos_masks,
            #     qua_masks=qua_masks,
            #     ang_masks=ang_masks,
            #     wid_masks=wid_masks
            # )

            # dataset.show_data(img, depth, box_p, ids_p, instance_masks, grasps=tmp, pos_masks=pos_masks, ang_masks=ang_masks, wid_masks=wid_masks, tgt_file="results/grasps/cogrv2_test_results_{}.png".format(i))

            obj_num_count, obj_success_count = calculate_grasp_iou_match(grasps, rects)
            total_obj_num_count += np.array(obj_num_count)
            total_obj_success_count += np.array(obj_success_count)

        print(np.array(total_obj_num_count))
        print(np.array(total_obj_success_count))
        print(np.array(total_obj_success_count) / np.array(total_obj_num_count))
        class_rate = np.array(total_obj_success_count) / np.array(total_obj_num_count)
        overrall_rate = class_rate.mean()
        print(overrall_rate)



        

        # keep = (class_p >= 0.3)
        # if not keep.any():
        #     print("No valid instance")
        # else:
        #     ids_p = ids_p[keep]
        #     class_p = class_p[keep]
        #     box_p = box_p[keep]
        #     coef_p = coef_p[keep]   
        #     pos_coef_p = pos_coef_p[keep]
        #     ang_coef_p = ang_coef_p[keep]
        #     wid_coef_p = wid_coef_p[keep]

        # print(ids_p.shape)
        # print(class_p.shape)
        # print(box_p.shape)
        # print(coef_p.shape)
        # print(pos_coef_p.shape)
        # print(ang_coef_p.shape)
        # print(wid_coef_p.shape)
        # print(proto_p.shape)

        # ids_p = (ids_p + 1)
        # ids_p = ids_p.cpu().numpy()
        # box_p = box_p

        # draw_lincomb(proto_p, coef_p, "cogr-v2-sem.png")
        # draw_lincomb(proto_p, pos_coef_p, "cogr-v2-gr-pos.png")
        # draw_lincomb(proto_p, ang_coef_p, "cogr-v2-gr-ang.png")
        # draw_lincomb(proto_p, wid_coef_p, "cogr-v2-gr-wid.png")

        # instance_masks = torch.sigmoid(torch.matmul(proto_p, coef_p.t())).contiguous()
        # print(instance_masks.shape)
        # instance_masks = crop(instance_masks, box_p).permute(2,0,1)
        # print("After crop: ", instance_masks.shape)

        # instance_masks_np = instance_masks.sum(dim=0).cpu().numpy()
        # cv2.imwrite("results/images/sem_mask.png", instance_masks_np*255)



        # pos_masks = torch.sigmoid(torch.matmul(proto_p, pos_coef_p.t())).contiguous()
        # pos_masks = crop(pos_masks, box_p).permute(2,0,1)

        # ang_masks = torch.matmul(proto_p, ang_coef_p.t()).contiguous()
        # ang_masks = crop(ang_masks, box_p).permute(2,0,1)

        # wid_masks = torch.sigmoid(torch.matmul(proto_p, wid_coef_p.t())).permute(2,0,1).contiguous()
        # # wid_masks = crop(wid_masks, box_p).permute(2,0,1)
        # wid_masks = wid_masks * pos_masks

        # box_p = box_p.cpu().numpy()


        # img = rgbd.transpose((1,2,0))[:, :, :3]
        # depth = rgbd.transpose((1,2,0))[:, :, 3]
        # print(img.shape)

        # # Convert processed image to original size
        # img_h, img_w = img.shape[:2]
        # ori_h, ori_w = 480, 640

        # img = cv2.resize(img, (ori_w, ori_w))
        # depth = cv2.resize(depth, (ori_w, ori_w))
        # ori_img = img[0:ori_h, 0:ori_w, :]
        # ori_depth = depth[0:ori_h, 0:ori_w]


        # instance_masks = F.interpolate(instance_masks.unsqueeze(0), (ori_w, ori_w), mode='bilinear', align_corners=False).squeeze(0)
        # instance_masks.gt_(0.5)

        # pos_masks = F.interpolate(pos_masks.unsqueeze(0), (ori_w, ori_w), mode='bilinear', align_corners=False).squeeze(0)
        # ang_masks = F.interpolate(ang_masks.unsqueeze(0), (ori_w, ori_w), mode='bilinear', align_corners=False).squeeze(0)
        # wid_masks = F.interpolate(wid_masks.unsqueeze(0), (ori_w, ori_w), mode='bilinear', align_corners=False).squeeze(0)


        # instance_masks = instance_masks[:, 0:ori_h, 0:ori_w]
        # pos_masks = pos_masks[:, 0:ori_h, 0:ori_w]
        # ang_masks = ang_masks[:, 0:ori_h, 0:ori_w]
        # wid_masks = wid_masks[:, 0:ori_h, 0:ori_w]

        # # ang_masks = ang_masks * pos_masks

        # print("===========================")
        # print(instance_masks.shape)
        # print(pos_masks.shape)
        # print(ang_masks.shape)
        # print(wid_masks.shape)

        # instance_masks = instance_masks.cpu().numpy()
        # pos_masks = pos_masks.cpu().numpy()
        # ang_masks = ang_masks.cpu().numpy()
        # ang_masks = ang_masks * np.pi
        # wid_masks = wid_masks.cpu().numpy()

        # for i in range(pos_masks.shape[0]):
        #     pos_masks[i] = gaussian(pos_masks[i], 2.0, preserve_range=True)
        #     ang_masks[i] = gaussian(ang_masks[i], 2.0, preserve_range=True)
        #     wid_masks[i] = gaussian(wid_masks[i], 1.0, preserve_range=True)

        # scale = np.array([ori_w, ori_w, ori_w, ori_w])
        # box_p *= scale

        # box_p = np.concatenate([box_p, ids_p.reshape(-1,1)], axis=-1)
        # print(box_p.shape)

        # ori_img = ori_img * norm_std + norm_mean
        # # dataset.show_data(ori_img, depth, box_p, ids_p, instance_masks, pos_masks=pos_masks, ang_masks=ang_masks, wid_masks=wid_masks, tgt_file="test_results.png")
        # # # idx = 4
        # # # dataset.show_data(ori_img, depth, np.array([box_p[idx]]), np.array([ids_p[idx]]), np.array([instance_masks[idx]]), pos_masks=np.array([pos_masks[idx]]), ang_masks=np.array([ang_masks[idx]]), wid_masks=np.array([wid_masks[idx]]), tgt_file="test_results.png")


        # from config import PER_CLASS_MAX_GRASP_WIDTH

        # grasps = []
        # for i in range(pos_masks.shape[0]):
        #     cls_id = int(ids_p[i])
        #     max_width = PER_CLASS_MAX_GRASP_WIDTH[cls_id-1]
        #     pos_mask = np.array(pos_masks[i], dtype='float')

        #     loacl_max = peak_local_max(pos_mask, min_distance=2, threshold_abs=0.2, num_peaks=10)

        #     for p_array in loacl_max:
        #         grasp_point = tuple(p_array)
        #         grasp_angle = ang_masks[i][grasp_point]
        #         grasp_width = wid_masks[i][grasp_point]
        #         grasps.append([float(grasp_point[1]), float(grasp_point[0]), grasp_width*max_width, 20, grasp_angle / np.pi * 180, int(ids_p[i])])


        # dataset.show_data(ori_img, ori_depth, box_p, ids_p, instance_masks, grasps=grasps, pos_masks=pos_masks, ang_masks=ang_masks, wid_masks=wid_masks, tgt_file="results/images/cogrv2-test_results.png")
            



    #     cudnn.fastest = True
    #     net = net.cuda()
    # print("Moving to CUDA...Done!")

    # evaluate(net, cfg)
