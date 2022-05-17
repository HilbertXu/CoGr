import os
import cv2
import numpy as np
from functools import partial

from skimage.measure import regionprops
from skimage.draw import polygon

import torch
import torch.utils.data as data

from utils.gr_augmentation import gr_train_aug

from graspnetAPI.grasp import RectGraspGroup
from detectron2.layers import nms_rotated


class GraspNetDataset(data.Dataset):
    def __init__(self, root_path, transform=None, mode="train"):
        self.root = root_path
        self.images = []
        self.annos  = []
        self.grasps = []
        self.depths = []

        self.transform = transform
        self.mode = mode

        for scene in os.listdir(self.root):
            kinect_scene_rgb   = os.path.join(self.root, scene, "kinect/rgb")
            kinect_scene_annos = os.path.join(self.root, scene, "kinect/label")
            kinect_scene_label = os.path.join(self.root, scene, "kinect/rects")
            kinect_scene_depth = os.path.join(self.root, scene, "kinect/depth")
            # realsense_scene_rgb = os.path.join(self.root, scene, "realsense/rgb")

            for f in os.listdir(kinect_scene_rgb):
                self.images.append(os.path.join(kinect_scene_rgb, f))
                self.annos.append(os.path.join(kinect_scene_annos, f))
                self.depths.append(os.path.join(kinect_scene_depth, f))
                self.grasps.append(os.path.join(kinect_scene_label, f[:-4]+".npy"))


    def __len__(self):
        assert len(self.images) == len(self.annos) == len(self.grasps)

        return len(self.images)
    

    def show_data(self, img, depth, bboxes, labels, ins_masks=None, grasps=None, pos_masks=None, ang_masks=None, wid_masks=None, display=False, tgt_file=None):
        from config import GRASPNET_DATASET as cls_list
        from config import COLORS as colors_list
        import matplotlib.pyplot as plt

        print(f'\nimg shape: {img.shape}')
        print('----------------boxes----------------')
        print(bboxes)
        print('----------------labels---------------')
        print([cls_list[int(i)] for i in labels], '\n')

        masks_semantic = ins_masks * (labels[:, None, None]+1)
        masks_semantic = masks_semantic.astype('int').sum(axis=0) % 88

        colors_list = np.array(colors_list)
        color_masks = colors_list[np.array(masks_semantic)].astype('uint8')
        img_u8 = img.astype('uint8')

        print(img_u8.shape)
        print(color_masks.shape)

        img_fused = (color_masks * 0.8 + img_u8 * 0.2)

        fig = plt.figure(figsize=(10, 10))

        for i in range(bboxes.shape[0]):
            name = cls_list[int(bboxes[i, -1])]
            color = colors_list[int(bboxes[i, -1])]
            cv2.rectangle(img_fused, (int(bboxes[i, 0]), int(bboxes[i, 1])),
                        (int(bboxes[i, 2]), int(bboxes[i, 3])), color.tolist(), 1)
            cv2.putText(img_fused, "{}:{}".format(name, int(bboxes[i, -1])), (int(bboxes[i, 0]), int(bboxes[i, 1])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        if grasps is not None:
            for rect in grasps:
                cls_id = rect[-1]
                name = cls_list[int(cls_id)]
                color = colors_list[int(cls_id)]
                if len(rect) == 7: 
                    center_x, center_y, width, height, theta, _, cls_id = rect
                elif len(rect) == 6:
                    center_x, center_y, width, height, theta, cls_id = rect
                box = ((center_x, center_y), (width, height), -(theta+180))
                box = cv2.boxPoints(box)
                box = np.int0(box)
                cv2.drawContours(img_fused, [box], 0, color.tolist(), 2)

        # for obj_rects in grasps:
        #     for rect in obj_rects:
        #         cls_id = rect[-1]
        #         name = cls_list[int(cls_id)]
        #         center_x, center_y, width, height, theta, cls_id = rect
        #         box = ((center_x, center_y), (width, height), theta)
        #         box = cv2.boxPoints(box)
        #         box = np.int0(box)
        #         cv2.drawContours(img_fused, [box], 0, (255, 0, 0), 2)
                

        cv2.imwrite("test.png", img_fused)


        ax = fig.add_subplot(2, 3, 1)
        ax.imshow(img_u8/255.)
        ax.set_title('RGB')
        ax.axis('off')

        if depth is not None:
            ax = fig.add_subplot(2, 3, 2)
            ax.imshow(depth, cmap='gray')
            ax.set_title('Depth')
            ax.axis('off')

        ax = fig.add_subplot(2, 3, 3)
        ax.imshow(img_fused/255.)
        ax.set_title('Masks & Bboxes')
        ax.axis('off')


        if (pos_masks is not None) and (ang_masks is not None) and (wid_masks is not None):
            all_pos_mask = np.zeros((pos_masks.shape[1], pos_masks.shape[2]))
            all_ang_mask = np.zeros((pos_masks.shape[1], pos_masks.shape[2]))
            all_wid_mask = np.zeros((pos_masks.shape[1], pos_masks.shape[2]))

            for pos_mask, ang_mask, wid_mask in zip(pos_masks, ang_masks, wid_masks):
                all_pos_mask += pos_mask
                all_ang_mask += ang_mask
                all_wid_mask += wid_mask

            # all_pos_mask = np.clip(all_pos_mask, 0.0, 1.0)
            # all_ang_mask = np.clip(all_ang_mask, 0.0, 1.0)
            # all_wid_mask = np.clip(all_wid_mask, 0.0, 1.0)
        
            ax = fig.add_subplot(2, 3, 4)
            plot = ax.imshow(all_pos_mask, cmap='jet', vmin=0, vmax=1)
            ax.set_title('Quality')
            ax.axis('off')
            plt.colorbar(plot)

            ax = fig.add_subplot(2, 3, 5)
            plot = ax.imshow(all_ang_mask, cmap='rainbow', vmin=-np.pi / 2, vmax=np.pi / 2)
            ax.set_title('Angle')
            ax.axis('off')
            plt.colorbar(plot)

            ax = fig.add_subplot(2, 3, 6)
            plot = ax.imshow(all_wid_mask, cmap='jet', vmin=0, vmax=1)
            ax.set_title('Width')
            ax.axis('off')
            plt.colorbar(plot)
        if display:
            plt.show()
        elif tgt_file is not None:
            print("Save visualizaitons")
            plt.savefig(tgt_file)
        else:
            print("Please specify the name of output file")
    

    def _load_rgb_img(self, index):
        img = cv2.imread(self.images[index], cv2.COLOR_BGR2RGB)
        
        return img

    def _load_depth_img(self, index):
        depth = cv2.imread(self.depths[index], cv2.IMREAD_UNCHANGED)

        # Preprocess depth image
        depth = 1 - (depth / np.max(depth))
        depth = np.expand_dims(depth, -1)

        return depth


    def _load_sem_masks(self, index):
        annos = cv2.imread(self.annos[index], cv2.IMREAD_UNCHANGED)
        # Here 0: background
        props = regionprops(annos)
        cls_ids = []
        bboxes  = []
        masks   = []
        for prop in props:
            # Get bbox and cls_id for each region
            cls_id = prop.label
            # prop.bbox: [min_y, min_x, max_y, max_x] -> [min_x, min_y, max_x, max_y]
            bboxes.append([prop.bbox[1], prop.bbox[0], prop.bbox[3], prop.bbox[2], cls_id])

            cls_ids.append(cls_id)

            # Get object mask
            mask = (annos == cls_id).astype(np.int8).astype(np.float32)
            masks.append(mask)

            # # Get corresponding bounding box
            # final_box = list(np.array([bbox[0], bbox[1], bbox[2], bbox[3]])/scale)

        bboxes = np.array(bboxes)
        labels = np.array(cls_ids)
        masks  = np.array(masks)
        
        return masks, bboxes, labels

    
    def _load_grasp_rects(self, index, score_threshold=0.7):
        rects = RectGraspGroup(self.grasps[index]).rect_grasp_group_array
        
        # Filter low-threshold rects
        mask = rects[:, 5] > score_threshold
        rects = rects[mask]

        center_x = rects[:, 0] # shape: (*, )
        center_y = rects[:, 1] # shape: (*, )
        open_point_x = rects[:, 2] # shape: (*, )
        open_point_y = rects[:, 3] # shape: (*, )
        height = rects[:, 4] # height of the rectangle, shape: (*, )
        score = rects[:, 5] # shape: (*, )
        cls_ids = rects[:, 6]

        width = 2 * np.sqrt(np.square(open_point_x - center_x) + np.square(open_point_y - center_y)) # width of the rectangle, shape: (*, )
        theta = np.zeros(width.shape) # rotation angle of the rectangle

        # @TODO
        # Be careful of the theta
        for i in range(theta.shape[0]):
            if center_x[i] > open_point_x[i]:
                theta[i] = -np.arctan((open_point_y[i] - center_y[i]) / (center_x[i] - open_point_x[i])) * 180 / np.pi
            elif center_x[i] < open_point_x[i]:
                theta[i] = -np.arctan((center_y[i] - open_point_y[i]) / (open_point_x[i] - center_x[i])) * 180 / np.pi
            else:
                theta[i] = 90.0
                
        rects = np.concatenate([center_x.reshape(-1,1), center_y.reshape(-1,1), width.reshape(-1,1), height.reshape(-1,1), theta.reshape(-1,1), score.reshape(-1,1), cls_ids.reshape(-1,1)], axis=-1)

        return rects
    


    def _match_rects_and_objects(self, rects, bboxes, masks, labels):
        def _apply_nms(rects, iou_threshold=0.2):
            r_boxs = rects[:, :5]
            scores = rects[:, 5]
            keep = nms_rotated(torch.from_numpy(r_boxs), torch.from_numpy(scores), iou_threshold).cpu().numpy()

            return keep

        object_rects = []
        object_bboxes = []
        object_masks = []
        object_labels = []

        _, height, width = masks.shape

        for i in range(bboxes.shape[0]):
            box = bboxes[i]
            mask = masks[i]
            label = labels[i]
            tmp = []
            for rect in rects:
               center_x, center_y, w, h = rect[:4]
               rect_obj_id = rect[-1] + 1
               if int(rect_obj_id) == int(box[4]):
                    # Center of grasp rect in bbox
                    if center_y <= height and center_x <= width:
                        if mask[int(center_y), int(center_x)]:
                            tmp.append(rect)

            if len(tmp) > 0:
                object_rects.append(tmp)
                object_bboxes.append(box)
                object_masks.append(mask)
                object_labels.append(label)
        
        for idx, rects in enumerate(object_rects):
            rects = np.array(rects)
            keep = _apply_nms(rects)
            object_rects[idx] = rects[keep]
            
        
        return object_rects, np.array(object_bboxes).astype("float"), np.array(object_masks), np.array(object_labels)


    def _draw_grasp_rects(self, rects, ins_masks, width, height):
        pos_masks = []
        qua_masks = []
        ang_masks = []
        wid_masks = []

        for idx, obj_rects in enumerate(rects):
            pos_out = np.zeros((height, width))
            qua_out = np.zeros((height, width))
            ang_out = np.zeros((height, width))
            wid_out = np.zeros((height, width))
            ins_mask = ins_masks[idx]
            for rect in obj_rects:
                center_x, center_y, w_rect, h_rect, theta, _, cls_id = rect
                width_factor = 360

                # Get 4 corners of rotated rect
                # Convert from our angle represent to opencv's
                r_rect = ((center_x, center_y), (w_rect/2, h_rect), -(theta+180))
                box = cv2.boxPoints(r_rect)
                box = np.int0(box)

                rr, cc = polygon(box[:, 0], box[:,1])

                mask_rr = rr < width
                rr = rr[mask_rr]
                cc = cc[mask_rr]

                mask_cc = cc < height
                cc = cc[mask_cc]
                rr = rr[mask_cc]


                pos_out[cc, rr] = 1.0
                qua_out[cc, rr] += 1.0
                ang_out[cc, rr] = theta * np.pi / 180
                # Adopt width normalize accoding to class 
                wid_out[cc, rr] = np.clip(w_rect, 0.0, width_factor) / width_factor

            # Preprocessing quality mask
            qua_out = 1 / (1 + np.exp(-qua_out))
            qua_out = qua_out * pos_out
            smooth_factor = 1e-7

            qua_out = np.clip(qua_out, smooth_factor, 1-smooth_factor)

            pos_out = pos_out * ins_mask
            qua_out = qua_out * ins_mask
            ang_out = ang_out * ins_mask
            wid_out = wid_out * ins_mask

            pos_masks.append(pos_out)
            qua_masks.append(qua_out)
            ang_masks.append(ang_out)
            wid_masks.append(wid_out)
        
        return np.array(pos_masks), np.array(qua_masks), np.array(ang_masks), np.array(wid_masks) 



    def __getitem__(self, index):
        rgb   = self._load_rgb_img(index)
        depth = self._load_depth_img(index)
        rects = self._load_grasp_rects(index)

        height, width, _ = rgb.shape

        # return rects

        ins_masks, bboxes, labels = self._load_sem_masks(index)

        # Match grasp rects with objects
        ins_rects, bboxes, ins_masks, labels = self._match_rects_and_objects(rects, bboxes, ins_masks, labels)


        # return ins_rects

        assert len(ins_rects) == len(bboxes) == len(ins_masks) == len(labels), "Data mismatch"

        # self.show_data(rgb, depth, np.array([bboxes[0]]), np.array(labels), np.array([ins_masks[0]]), ins_rects[0], tgt_file="graspnet_0.png")
        

        pos_masks, qua_masks, ang_masks, wid_masks = self._draw_grasp_rects(ins_rects, ins_masks, width, height)
        
        self.show_data(rgb, depth, np.array(bboxes), np.array(labels), np.array(ins_masks), rects, pos_masks, ang_masks, wid_masks, tgt_file="graspnet_0.png")


        if self.mode == "train":
            img, depth, ins_masks, pos_masks, qua_masks, ang_masks, wid_masks, bboxes, labels = self.transform(
                rgb, depth, ins_masks, pos_masks, qua_masks, ang_masks, wid_masks, bboxes[:, :4], labels
            )

            if len(depth.shape) == 2:
                depth = np.expand_dims(depth, -1)
            # self.show_data(img, depth, np.array([bboxes[2]]), np.array([labels[2]]), ins_masks=np.array([ins_masks[2]]), pos_masks=np.array([pos_masks[2]]), ang_masks=np.array([ang_masks[2]]), wid_masks=np.array([wid_masks[2]]), tgt_file="graspnet_0.png")

            rgbd = np.concatenate([img, depth], axis=-1).transpose((2,0,1))
            bboxes = np.concatenate([bboxes, labels.reshape(-1,1)], axis=-1)

            # Test using 0 - pi / pi
            # ang_sin_masks = np.sin(2 * ang_masks)
            # ang_cos_masks = 1 - (np.cos(2 * ang_masks) + 1.) / 2.

            sin_masks = np.sin(2 * ang_masks)
            cos_masks = np.cos(2 * ang_masks)


            return rgbd, bboxes, ins_masks, pos_masks, qua_masks, sin_masks, cos_masks, wid_masks

        elif self.mode == "test":
            img, depth, ins_masks, pos_masks, qua_masks, ang_masks, wid_masks, bboxes, labels = self.transform(
                rgb, depth, ins_masks, pos_masks, qua_masks, ang_masks, wid_masks, bboxes[:, :4], labels
            )
            

            if len(depth.shape) == 2:
                depth = np.expand_dims(depth, -1)
            # self.show_data(img, depth, np.array([bboxes[2]]), np.array([labels[2]]), ins_masks=np.array([ins_masks[2]]), pos_masks=np.array([pos_masks[2]]), ang_masks=np.array([ang_masks[2]]), wid_masks=np.array([wid_masks[2]]))

            rgbd = np.concatenate([img, depth], axis=-1).transpose((2,0,1))
            bboxes = np.concatenate([bboxes, labels.reshape(-1,1)], axis=-1)

            # Test using 0 - pi / pi
            # ang_sin_masks = np.sin(2 * ang_masks)
            # ang_cos_masks = 1 - (np.cos(2 * ang_masks) + 1.) / 2.

            # Normalize angle mask to [-1,1]
            sin_masks = np.sin(2 * ang_masks)
            cos_masks = np.cos(2 * ang_masks)

            # ang_sin_masks: [0, 1]
            # ang_cos_masks: [-1,1]
            # @NOTE
            # Should we normalize ang_cos_masks to [0, 1]?

            # self.show_data(img, depth, np.array([bboxes]), np.array([labels]), ins_masks=np.array([ins_masks]), pos_masks=np.array([pos_masks]), ang_masks=np.array([ang_sin_masks]), wid_masks=np.array([wid_masks]), tgt_file="results/grasps/data_with_annos-1.py")

            return rgbd, bboxes, ins_rects, ins_masks, pos_masks, qua_masks, ang_masks, wid_masks
        

if __name__ == "__main__":
    from tqdm import tqdm

    dataset = GraspNetDataset(
        "/home/puzek/sdb/dataset/graspnet",
        transform=partial(gr_train_aug, 544),
        mode="train"
    )

    dataset[0]

    # length = len(dataset)

    # max_grasp_width = [0 for i in range(88)]
    # min_grasp_angle = 9999
    # max_grasp_angle = -9999

    # pbar = tqdm(range(length))

    # for i in pbar:
    #     rects = dataset[i]

    #     for obj_rects in rects:
    #         for rect in obj_rects:
    #             cls_id = rect[-1]
    #             width = rect[2]
    #             theta = rect[4]
    #             if width > max_grasp_width[int(cls_id)-1]:
    #                 max_grasp_width[int(cls_id)-1] = int(width)
    #             if theta > max_grasp_angle:
    #                 max_grasp_angle = theta
    #             elif theta < min_grasp_angle:
    #                 min_grasp_angle = theta
    

    # print(max_grasp_width)
    # print(min_grasp_angle)
    # print(max_grasp_angle)
