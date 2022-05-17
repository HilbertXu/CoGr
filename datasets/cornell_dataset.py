from functools import partial
import os
import cv2
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as data
from pycocotools.coco import COCO
from skimage.draw import polygon
from config import CORNELL_LABEL_MAP
from utils.gr_augmentation import gr_train_aug_cornell, gr_val_aug

# dataset = json.load(open("/home/hilbertxu/dataset/CornellGrasping/CornellGrasping.json"))
# print(len(dataset["images"]))
# print(len(dataset["annotations"]))
# print(dataset["images"][0])
# print(dataset.keys())
# print(dataset["categories"])
# print(len(dataset["categories"]))

# for i in range(len(dataset)):
#     dataset["images"][i]["path"] = dataset["images"][i]["path"][25:]

# print(dataset["images"][0])


# tmp = {}
# for idx, anno in enumerate(dataset["categories"]):
#     tmp[int(anno["id"])] = idx + 1


# print(tmp)


class CornellGraspDetection(data.Dataset):
    def __init__(self, root_path, transform=None, mode='train'):
        self.root_path  = root_path
        self.mode = mode

        annos_path = os.path.join(self.root_path, "CornellGrasping.json")
        self.coco = COCO(annos_path)

        self.ids = list(self.coco.imgToAnns.keys())


        self.max_grasp_width = [74, 69, 112, 67, 92, 102, 82, 62, 108, 64, 82, 
                                89, 54, 57, 33, 92, 95, 84, 84, 82, 56, 94, 82,
                                149, 63, 67, 81, 102, 80, 73, 71, 59, 47, 70, 65,
                                133, 111, 67, 61, 115, 79, 88, 106, 88, 91, 71, 102,
                                115, 115, 125, 82, 129, 95, 91, 117, 108, 98, 113, 96,
                                60, 89, 87, 62]
        
        self.transform = transform
    

    def __len__(self):
        return len(self.ids)

    
    def show_data(self, img, depth, bboxes, labels, ins_masks=None, grasps=None, pos_masks=None, ang_masks=None, wid_masks=None, display=False, tgt_file=None):
        from OCID_class_dict import colors_list
        from config import CORNELL_CLASSES
        
        print(f'\nimg shape: {img.shape}')
        print('----------------boxes----------------')
        print(bboxes)
        print('----------------labels---------------')
        print([CORNELL_CLASSES[int(i)] for i in labels], '\n')

        masks_semantic = ins_masks * (labels[:, None, None]+1)
        masks_semantic = masks_semantic.astype('int').sum(axis=0) % 31

        colors_list = np.array(colors_list)
        color_masks = colors_list[np.array(masks_semantic)].astype('uint8')
        img_u8 = img.astype('uint8')
        img_fused = (color_masks * 0.8 + img_u8 * 0.2)

        fig = plt.figure(figsize=(10, 10))

        for i in range(bboxes.shape[0]):
            name = CORNELL_CLASSES[int(bboxes[i, -1])]
            color = colors_list[int(bboxes[i, -1])]
            cv2.rectangle(img_fused, (int(bboxes[i, 0]), int(bboxes[i, 1])),
                        (int(bboxes[i, 2]), int(bboxes[i, 3])), color.tolist(), 1)
            cv2.putText(img_fused, "{}:{}".format(name, int(bboxes[i, -1])), (int(bboxes[i, 0]), int(bboxes[i, 1])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        if grasps is not None:
            for rect in grasps:
                cls_id = rect[-1]
                name = CORNELL_CLASSES[int(cls_id)]
                color = colors_list[int(cls_id)]
                print(rect)
                center_x, center_y, width, height, theta, cls_id = rect
                box = ((center_x, center_y), (width, height), -theta)
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

            all_pos_mask = np.clip(all_pos_mask, 0.0, 1.0)
            all_ang_mask = np.clip(all_ang_mask, 0.0, 1.0)
            all_wid_mask = np.clip(all_wid_mask, 0.0, 1.0)
        
            ax = fig.add_subplot(2, 3, 4)
            plot = ax.imshow(all_pos_mask, cmap='jet', vmin=0, vmax=1)
            ax.set_title('Quality')
            ax.axis('off')
            plt.colorbar(plot)

            ax = fig.add_subplot(2, 3, 5)
            plot = ax.imshow(all_ang_mask, cmap='hsv', vmin=-np.pi / 2, vmax=np.pi / 2)
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
        

    
    
    def _gr_text_to_no(self, l, offset=(0, 0)):
        """
        Transform a single point from a Cornell file line to a pair of ints.
        :param l: Line from Cornell grasp file (str)
        :param offset: Offset to apply to point positions
        :return: Point [y, x]
        """
        x, y = l.split()
        return [int(round(float(y))) - offset[0], int(round(float(x))) - offset[1]]

    

    def _load_grasp(self, fname, label):
        grs = []
        with open(fname) as f:
            while True:
                # Load 4 lines at a time, corners of bounding box.
                p1 = f.readline()
                if not p1:
                    break  # EOF
                p2, p3, p4 = f.readline(), f.readline(), f.readline()

                try:
                    p1 = self._gr_text_to_no(p1)
                    p2 = self._gr_text_to_no(p2)
                    p3 = self._gr_text_to_no(p3)
                    p4 = self._gr_text_to_no(p4)
                    

                    center_x = (p1[0] + p3[0]) / 2
                    center_y = (p1[1] + p3[1]) / 2

                    width  = np.sqrt((p1[0] - p4[0]) * (p1[0] - p4[0]) + (p1[1] - p4[1]) * (p1[1] - p4[1]))
                    height = np.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]))

                    theta = 90 + np.arctan2(p4[0] - p1[0], p4[1] - p1[1]) * 180 / np.pi

                    if theta > 180:
                        theta = theta % 180
                    elif theta < 0:
                        theta = theta + 180

                    grs.append((center_y, center_x, height, width, theta, label))
                except ValueError:
                    # Some files contain weird values.
                        continue
        return grs


    def _draw_grasp_rects(self, rects, width, height):
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
                center_x, center_y, w_rect, h_rect, theta, cls_id = rect
                width_factor = float(self.max_grasp_width[int(cls_id)-1])

                # Get 4 corners of rotated rect
                r_rect = ((center_x, center_y), (w_rect/2, h_rect), theta)
                box = cv2.boxPoints(r_rect)
                box = np.int0(box)

                rr, cc = polygon(box[:, 0], box[:,1], shape)

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

            pos_masks.append(pos_out)
            qua_masks.append(qua_out)
            ang_masks.append(ang_out)
            wid_masks.append(wid_out)
        
        return np.array(pos_masks), np.array(qua_masks), np.array(ang_masks), np.array(wid_masks) 




    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)

        # Only one target in each image of Cornell dataset
        target = self.coco.loadAnns(ann_ids)[0]
        img_path = os.path.join(self.root_path, self.coco.loadImgs(img_id)[0]['path'][25:])
        depth_path = img_path[:-5] + "d.tiff"
        grasp_path = img_path[:-5] + "cpos.txt"

        img = cv2.imread(img_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth = 1 - depth / np.max(depth)

        depth = np.expand_dims(depth, -1)

        height, width, _ = img.shape

        bbox = target["bbox"]
        bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
        ins_mask = self.coco.annToMask(target)
        label = CORNELL_LABEL_MAP[int(target["category_id"])]

        bbox.append(label)

        rects = self._load_grasp(grasp_path, label)

        ins_masks = np.array([ins_mask])
        bboxes = np.array([bbox], dtype='float')
        labels = np.array([label])

        # self.show_data(img, depth, bboxes=bboxes, labels=labels, ins_masks=ins_masks, tgt_file="cornell_test.png")

        pos_masks, qua_masks, ang_masks, wid_masks = self._draw_grasp_rects([rects], 640, 480)

        # self.show_data(img, depth, bboxes=np.array([box]), labels=np.array([label]), ins_masks=np.array([ins_mask]), grasps=np.array(rects), pos_masks=pos_masks, ang_masks=ang_masks, wid_masks=wid_masks, tgt_file="cornell_test.png")

        if self.mode == "train":
            img, depth, ins_masks, pos_masks, qua_masks, ang_masks, wid_masks, bboxes, labels = self.transform(
                img, depth, ins_masks, pos_masks, qua_masks, ang_masks, wid_masks, bboxes[:, :4], labels
            )

            if img is None:
                return None, None, None, None, None, None, None, None, None

            if len(depth.shape) == 2:
                depth = np.expand_dims(depth, -1)
            
            # self.show_data(img, depth, bboxes, labels, ins_masks=ins_masks, pos_masks=pos_masks, ang_masks=ang_masks, wid_masks=wid_masks, tgt_file="cornell_test.png")

            rgbd = np.concatenate([img, depth], axis=-1).transpose((2,0,1))
            bboxes = np.concatenate([bboxes, labels.reshape(-1,1)], axis=-1)

            # Test using 0 - pi / pi
            # ang_sin_masks = np.sin(2 * ang_masks)
            # ang_cos_masks = 1 - (np.cos(2 * ang_masks) + 1.) / 2.

            ang_masks = ang_masks / np.pi


            return rgbd, bboxes, ins_masks, pos_masks, qua_masks, ang_masks, wid_masks

        elif self.mode == "test":
            img, depth, ins_masks, pos_masks, qua_masks, ang_masks, wid_masks, bboxes, labels = self.transform(
                img, depth, ins_masks, pos_masks, qua_masks, ang_masks, wid_masks, bboxes[:, :4], labels
            )

            if len(depth.shape) == 2:
                depth = np.expand_dims(depth, -1)
            # self.show_data(img, depth, np.array([bboxes[2]]), np.array([labels[2]]), ins_masks=np.array([ins_masks[2]]), pos_masks=np.array([pos_masks[2]]), ang_masks=np.array([ang_masks[2]]), wid_masks=np.array([wid_masks[2]]))


            rgbd = np.concatenate([img, depth], axis=-1).transpose((2,0,1))
            bboxes = np.concatenate([bboxes, labels.reshape(-1,1)], axis=-1)

            # Test using 0 - pi / pi
            # ang_sin_masks = np.sin(2 * ang_masks)
            # ang_cos_masks = 1 - (np.cos(2 * ang_masks) + 1.) / 2.

            ang_masks = ang_masks / np.pi

            # ang_sin_masks: [0, 1]
            # ang_cos_masks: [-1,1]
            # @NOTE
            # Should we normalize ang_cos_masks to [0, 1]?

            # self.show_data(img, depth, np.array([bboxes[2]]), np.array([labels[2]]), ins_masks=np.array([ins_masks[2]]), pos_masks=np.array([pos_masks[2]]), ang_masks=np.array([ang_sin_masks[2]]), wid_masks=np.array([wid_masks[2]]), tgt_file="data_with_annos-1.py")

            return rgbd, bboxes, rects, ins_masks, pos_masks, qua_masks, ang_masks, wid_masks
        


if __name__ == "__main__":
    from tqdm import tqdm

    dataset = CornellGraspDetection("/home/puzek/sdb/dataset/CornellGrasping", transform=partial(gr_train_aug_cornell, 544))

    dataset[0]

    # max_grasp_width = [0 for i in range(63)]
    # min_grasp_angle = 9999
    # max_grasp_angle = -9999

    # pbar = tqdm(range(len(dataset)))
    # for i in pbar:
    #     rects = dataset[i]

    #     for rect in rects:
    #         cls_id = rect[-1]
    #         width = rect[2]
    #         theta = rect[4]
    #         print(cls_id)
    #         if width > max_grasp_width[int(cls_id)-1]:
    #             max_grasp_width[int(cls_id)-1] = int(width)
    #         if theta > max_grasp_angle:
    #             max_grasp_angle = theta
    #         elif theta < min_grasp_angle:
    #             min_grasp_angle = theta

    # print(max_grasp_width)
