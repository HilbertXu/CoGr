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


class GraspNetDetection(data.Dataset):
    def __init__(self, dataset_path, transform=None, grasp_transform=None) -> None:
        self.root = dataset_path
        self.images = []
        self.annos  = []
        self.grasps = []
        self.depths = []

        self.transform = transform
        self.grasp_transform = grasp_transform

        self.max_grasp_width = 360

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
        masks_semantic = masks_semantic.astype('int').sum(axis=0) % 31

        cls_list = np.array(cls_list)
        colors_list = np.array(colors_list)
        color_masks = colors_list[np.array(masks_semantic)].astype('uint8')
        img_u8 = img.astype('uint8')
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

                

    def _check_data(self, rects, bboxes, masks, labels):
        print("Data checking...")
        for i in range(len(bboxes)):
            c_rects = rects[i]
            box = bboxes[i]
            mask = masks[i]
            object_label = labels[i]

            cls_id = box[4]

            assert int(object_label) == int(cls_id)

            flag = True
            print("Current class: {}, Num rects: {}".format(cls_id, len(c_rects)))

            for rect in c_rects:
                if int(rect[-1]) != int(cls_id):
                    flag = False
                    print("Oooops, class mismatch...")
            if flag:
                print("Class match...") 


    def _apply_nms(self, rects, iou_threshold=0.2):
        x_min, y_min, x_max, y_max, angle, scores, cls_ids = rects[:,0], rects[:,1], rects[:,2], rects[:,3], rects[:,4], rects[:,5], rects[:,6]
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        width = (x_max - x_min)
        height = (y_max - y_min)


        r_box = np.concatenate([
            center_x.reshape(-1,1), center_y.reshape(-1,1), 
            width.reshape(-1,1), height.reshape(-1,1),
            angle.reshape(-1,1)
        ], axis=1)

        keep = nms_rotated(torch.from_numpy(r_box), torch.from_numpy(scores), iou_threshold).cpu().numpy()
        
        return keep
    

    def _match_rects_and_objects(self, rects, bboxes, masks, labels, min_corners=1):
        object_rects = []
        object_bboxes = []
        object_masks = []
        object_labels = []

        for i in range(bboxes.shape[0]):
            box = bboxes[i]
            mask = masks[i]
            label = labels[i]
            tmp = []
            for rect in rects:
               center_x, center_y, w, h = rect[:4]
               rect_obj_id = rect[-1]

               if int(rect_obj_id) == int(box[4]):
                    # Center of grasp rect in bbox
                    if mask[int(center_y), int(center_x)]:
                        tmp.append(rect)

            if len(tmp) > 0:
                object_rects.append(tmp)
                object_bboxes.append(box)
                object_masks.append(mask)
                object_labels.append(label)
        
        return object_rects, object_bboxes, object_masks, object_labels
        

    def _depth_inpaint(self, img, missing_value=0):
        """
        Inpaint missing values in depth image.
        :param missing_value: Value to fill in the depth image.
        """
        # cv2 inpainting doesn't handle the border properly
        # https://stackoverflow.com/questions/25974033/inpainting-depth-map-still-a-black-image-border
        img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
        mask = (img == missing_value).astype(np.uint8)

        # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
        scale = np.abs(img).max()
        img = img.astype(np.float32) / scale  # Has to be float32, 64 not supported.
        img = cv2.inpaint(img, mask, 1, cv2.INPAINT_NS)

        # Back to original size and value range.
        img = img[1:-1, 1:-1]
        img = img * scale

        return img
    

    def _normalise_bboxes(self, bboxes, width, height):
        scale = np.array([width, height, width, height])
        res = []
        for box in bboxes:
            label = box[4]
            final_box = list(np.array([box[0], box[1], box[2], box[3]]) / scale)
            final_box.append(float(label))
            res += [final_box]
        
        return res


    def preprocess_rect_grasps(self, rects, score_threshold=0.7):
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


    def preprocess_semantic_annos(self, img, annos):
        # Get object regions
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

        
        return img, bboxes, masks, labels

    
    def draw_grasp_rects(self, rects, width, height):

        pos_masks = []
        qua_masks = []
        ang_masks = []
        wid_masks = []

        shape = (height, width)

        for obj_rects in rects:
            pos_out = np.zeros((height, width))
            qua_out = np.zeros((height, width))
            ang_out = np.zeros((height, width))
            wid_out = np.zeros((height, width))

            for rect in obj_rects:
                center_x, center_y, w_rect, h_rect, theta = rect[:5]

                # Get 4 corners of rotated rect
                r_rect = ((center_x, center_y), (w_rect/2, h_rect), theta)
                box = cv2.boxPoints(r_rect)
                box = np.int0(box)

                rr, cc = polygon(box[:, 0], box[:,1])

                mask_rr = rr < width
                rr = rr[mask_rr]
                cc = cc[mask_rr]

                mask_cc = cc < height
                cc = cc[mask_cc]
                rr = rr[mask_cc]

                # @TODO
                # Width out should be normalized
                pos_out[cc, rr] = 1.0
                qua_out[cc, rr] += 1.0
                ang_out[cc, rr] = theta * np.pi / 180
                wid_out[cc, rr] = w_rect
            
            qua_out = 1 / (1 + np.exp(-qua_out))
            qua_out = qua_out * pos_out
            smooth_factor = 1e-7

            qua_out = np.clip(qua_out, smooth_factor, 1-smooth_factor)

            pos_masks.append(pos_out)
            qua_masks.append(qua_out)
            ang_masks.append(ang_out)
            wid_masks.append(wid_out)
        
        return np.array(pos_masks), np.array(qua_masks), np.array(ang_masks), np.array(wid_masks)



    def __len__(self):
        assert len(self.images) == len(self.annos) == len(self.grasps)

        return len(self.images)

    
    def __getitem__(self, index):
        ori_img = cv2.imread(self.images[index])
        annos   = cv2.imread(self.annos[index], cv2.IMREAD_UNCHANGED)
        depth   = cv2.imread(self.depths[index], cv2.IMREAD_UNCHANGED) / 1000
        rects   = RectGraspGroup(self.grasps[index]).rect_grasp_group_array

        # Normalize depth image
        depth = 1 - np.clip(depth / (np.max(depth)), 0, 1)
        depth = np.expand_dims(depth, axis=-1)

        # Preprocessing RGB image and semantic annotations
        img, bboxes, masks, labels = self.preprocess_semantic_annos(ori_img, annos)
        
        # Preprocessing rectangle grasps to [x_min, x_max, y_min, y_max, angle]
        rects = self.preprocess_rect_grasps(rects) 

        print(rects.shape)
        print(rects)
        
        rects, bboxes, masks, labels = self._match_rects_and_objects(rects, bboxes, masks, labels)
        # for obj_rects, abs_bbox, bbox in zip(obj_rect_group, abs_bboxes, bboxes):
        #     print(obj_rects[0][-1], abs_bbox[-1], bbox[-1])

        # Check data
        assert len(rects) == len(bboxes) == len(masks) == len(labels), "Data mismatch"

        # @TODO
        # Do we need to remove some extremely small object? 
        # Strawberry in scene_0006/0121.png
        # Apply object-wise NMS
        for idx, obj_rects in enumerate(rects):
            obj_rects = np.array(obj_rects)
            keep = self._apply_nms(obj_rects, iou_threshold=0.3)
            rects[idx] = obj_rects[keep]


        height, width, _ = img.shape
        num_crowds = 0

        # Draw grasp rects to generate grasp maps
        pos_masks, qua_masks, ang_masks, wid_masks = self.draw_grasp_rects(rects, width, height)

        bboxes = np.array(bboxes, dtype='float32')
        labels = np.array(labels)
        masks  = np.array(masks)

        self.show_data(img, depth, bboxes, labels, masks, rects, pos_masks,  ang_masks, wid_masks, tgt_file="data_with_annos_1.png")

        # if self.transform is not None:
        #     if len(bboxes) > 0:
        #         img, depth, pos_masks, ang_masks, wid_masks, masks, bboxes, labels = self.transform(
        #             img, depth, masks, pos_masks, ang_masks, wid_masks, bboxes[:, :4], labels
        #         )
        #         if img is None:
        #             return None, None, None, None, None, None, None, None

        #     else:
        #         return None, None, None, None, None, None, None, None
        # rgbd = np.concatenate([img, depth], axis=-1).transpose((2,0,1))
        # bboxes = np.concatenate([bboxes, labels.reshape(-1,1)], axis=-1)

        # ang_sin_masks = np.sin(2 * ang_masks)
        # ang_cos_masks = np.cos(2 * ang_masks)


        return rgbd, bboxes, masks, pos_masks, ang_sin_masks, ang_cos_masks, wid_masks



if __name__ == "__main__":
    dataset = GraspNetDetection(
        "/home/puzek/sdb/dataset/graspnet",
        transform=partial(gr_train_aug, 768)
    )

    rgbd, bboxes, masks, pos_masks, ang_sin_masks, ang_cos_masks, wid_masks = dataset[0]


    print(pos_masks.shape)

    # img = np.ones((720, 1280, 3))
    # # cv2.imshow("test", img)
    # # cv2.waitKey(0)


    # rect = [435.8349, 266.76755, 600.89734, 257.68747, 30.779282, 0.90000004, 8.]


    # # original way    
    # center_x, center_y, open_x, open_y, height, score, object_id = rect
    # center = np.array([center_x, center_y])
    # left = np.array([open_x, open_y])
    # axis = left - center
    # normal = np.array([-axis[1], axis[0]])
    # normal = normal / np.linalg.norm(normal) * height / 2
    # p1 = center + normal + axis
    # p2 = center + normal - axis
    # p3 = center - normal - axis
    # p4 = center - normal + axis
    # print(p1, p2, p3, p4)
    # cv2.line(img, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), (0,0,255), 1, 8)
    # cv2.line(img, (int(p2[0]),int(p2[1])), (int(p3[0]),int(p3[1])), (255,0,0), 3, 8)
    # cv2.line(img, (int(p3[0]),int(p3[1])), (int(p4[0]),int(p4[1])), (0,0,255), 1, 8)
    # cv2.line(img, (int(p4[0]),int(p4[1])), (int(p1[0]),int(p1[1])), (255,0,0), 3, 8)

    # # cv2.imshow("test", img)
    # # cv2.waitKey(0)

    # # corners + rotation way
    # center_x = rect[0] # shape: (*, )
    # center_y = rect[1] # shape: (*, )
    # open_point_x = rect[2] # shape: (*, )
    # open_point_y = rect[3] # shape: (*, )
    # height = rect[4] # height of the rectangle, shape: (*, )
    # score = rect[5] # shape: (*, )
    # print(center_x, center_y)
    # print(open_point_x, open_point_y)
    # print(height)
    # print(score)
    # width = 2 * np.sqrt(np.square(open_point_x - center_x) + np.square(open_point_y - center_y))
    # theta = np.zeros(width.shape)

    # print(theta.shape)
    # rotation_class = np.zeros(theta.shape, dtype='int32') # rotation class of the rectangle
    # x_min = center_x - width / 2
    # y_min = center_y - height / 2
    # x_max = center_x + width / 2
    # y_max = center_y + height / 2

    # if center_x > open_point_x:
    #     theta = -np.arctan((open_point_y - center_y) / (center_x - open_point_x)) * 180 / np.pi
    # elif center_x < open_point_x:
    #     theta = -np.arctan((center_y - open_point_y) / (open_point_x - center_x)) * 180 / np.pi
    # else:
    #     theta = 90.0
    # rotation_class = int(round((theta + 90) / 10) + 1)
    # print(rotation_class, theta)

    # center_x = (x_min + x_max) / 2
    # center_y = (y_min + y_max) / 2
    # w = x_max - x_min
    # h = y_max - y_min



    # r_rect = ((center_x, center_y), (w, h), theta)
    # box = cv2.boxPoints(r_rect)
    # box = np.int0(box)
    # cv2.drawContours(img,[box],0,(0,0,255),2)

    # cv2.imshow("test", img)
    # cv2.waitKey(0)

