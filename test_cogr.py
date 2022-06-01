# Basic modules
import os
import cv2
import time
import math
import torch
import argparse
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

# ROS modules
import rospy
import tf
from cv_bridge import CvBridge
from sensor_msgs.msg import PointCloud2, Image
from sensor_msgs import point_cloud2
from race_basic_motion_control.srv import *
from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory
from message_filters import ApproximateTimeSynchronizer, Subscriber
from geometry_msgs.msg import PointStamped

# Gazebo modules
from gazebo_msgs.msg import ModelState
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import DeleteModel, SpawnModel, SetModelState, GetWorldProperties
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Point, Pose

# CoGr modules
from CoGr.modules.gr_yolact import GraspYolact
from CoGr.utils.gr_augmentation import normalize_and_toRGB
from CoGr.utils.output_utils import gr_nms_v2, gr_post_processing
from CoGr.config import get_config
from CoGr.utils import timer
from CoGr.datasets.OCID_class_dict import cls_list


parser = argparse.ArgumentParser(description='YOLACT COCO Evaluation')
parser.add_argument('--img_size', type=int, default=544, help='The image size for validation.')
parser.add_argument('--weight', type=str, default='weights/CoGr/latest_CoGrv2_26000.pth')
parser.add_argument('--traditional_nms', default=False, action='store_true', help='Whether to use traditional nms.')
parser.add_argument('--val_num', default=-1, type=int, help='The number of images for test, set to -1 for all.')
parser.add_argument('--coco_api', action='store_true', help='Whether to use cocoapi to evaluate results.')

norm_mean = np.array([103.94, 116.78, 123.68], dtype=np.float32)
norm_std = np.array([57.38, 57.12, 58.40], dtype=np.float32)

name_mapping = {
    "coffee_mug":"mug",
    "banana":"banana",
    "apple":"apple",
    "cereal_box":"cereal_box1"
}



class CoGrDemo(object):
    def __init__(
        self, 
        cfg="res101_race", 
        input_size=544, 
        weights="/home/hilbertxu/air_ws/race_project/dual_arm_ws/src/CoGr/weights/OCID-weights/latest_CoGrv2_34000.pth",
        max_num_obj=4) -> None:

        """
        area: 
            x: [-0.47, 0.32]
            y: [0.47, 0.9]
        """
        
        # initialize CoGr network
        print("========== Initialize CoGr Network ==========")
        self.args = parser.parse_args()
        self.args.cfg = cfg

        self.cfg = get_config(self.args, mode='val')

        self.net = GraspYolact(self.cfg)
        self.net.eval()

        state_dict = torch.load(weights, map_location='cpu')

        correct_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        with torch.no_grad():
            import torch.nn as nn
            self.net.backbone.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.net.load_weights(correct_state_dict, self.cfg.cuda)

        self.net = self.net.cuda()

        self.input_size = input_size
        self.norm_mean = np.array([103.94, 116.78, 123.68], dtype=np.float32)
        self.norm_std = np.array([57.38, 57.12, 58.40], dtype=np.float32)

        print("=============================================")

        print("============= Initialize ROS Node ===========")
        # initialize ros node
        rospy.init_node('cogr_demo')
        print("Waiting for behaviors service...")
        rospy.wait_for_service('/behaviors_service')
        print("Done!")

        self.behaviors_service = rospy.ServiceProxy('/behaviors_service', behaviors)
        self.delete_model_service = rospy.ServiceProxy

        self.bridge = CvBridge()

        rgb_image_sub = Subscriber("/head_mount_kinect/rgb/image_raw", Image)
        depth_image_sub = Subscriber("/head_mount_kinect/depth_registered/image_raw", Image)
        pcl_sub = Subscriber("/head_mount_kinect/depth_registered/points", PointCloud2)

        # self.ats = ApproximateTimeSynchronizer([rgb_image_sub, depth_image_sub, pcl_sub], queue_size=5, slop=0.2)
        # self.ats.registerCallback(self.image_callback)

        self.listener = tf.TransformListener()
        self.listener.waitForTransform("world", "head_mount_kinect_rgb_optical_frame", rospy.Time(0), rospy.Duration(0.1))

        print("=============================================")

        print("============= Initialize Requests ===========")
        self.init_behavior = behaviorsRequest()
        self.init_behavior.robot = "dual"
        self.init_behavior.behavior = "go_to_initial_pose"
        

        self.rest_behavior = behaviorsRequest()
        self.rest_behavior.robot = "dual"
        self.rest_behavior.behavior = "IK_move_to_pose"
        self.rest_behavior.left_target_pose.pose.position.x = -0.6316975673894907
        self.rest_behavior.left_target_pose.pose.position.y = 0.2640369396674833
        self.rest_behavior.left_target_pose.pose.position.z = 0.4172036088024915
        self.rest_behavior.left_target_pose.pose.orientation.x = 0.0
        self.rest_behavior.left_target_pose.pose.orientation.y = 0.707
        self.rest_behavior.left_target_pose.pose.orientation.z = 0.0
        self.rest_behavior.left_target_pose.pose.orientation.w = 0.707

        self.rest_behavior.right_target_pose.pose.position.x = 0.6385976023959739
        self.rest_behavior.right_target_pose.pose.position.y = 0.2839896001735137
        self.rest_behavior.right_target_pose.pose.position.z = 0.41695947545892564
        self.rest_behavior.right_target_pose.pose.orientation.x = 0.0
        self.rest_behavior.right_target_pose.pose.orientation.y = 0.707
        self.rest_behavior.right_target_pose.pose.orientation.z = 0.0
        self.rest_behavior.right_target_pose.pose.orientation.w = 0.707

        self.load_model("wooden_box", "wooden_box_left", [-0.6316975673894907, 0.2640369396674833, 0.01])
        self.load_model("wooden_box", "wooden_box_right", [0.6385976023959739, 0.2839896001735137, 0.01])

        print("=============================================")


        print("============== Create Objects ===============")
        self.valid_obj_list = ["apple", "banana", "cereal_box1", "mug", "cereal_box2", "pepsi_can", "sprite_can", "bowl_blue", "marker_blue", "lemon"]
        # "mug_blue", "mug_red", "mug_yellow", "marker_black", "sponge_yellow"
        self.max_num_obj = max_num_obj
        last_x, last_y = 0, 0

        for i in range(self.max_num_obj):
            x = np.random.uniform(low=-0.47, high=0.32)
            y = np.random.uniform(low=0.47, high=0.90)

            if i == 0:
                last_x, last_y = x, y
            else:
                dist = math.sqrt(math.pow((x-last_x),2)+math.pow((y-last_y),2))
                while (dist <= 0.1):
                    x = np.random.uniform(low=-0.47, high=0.32)
                    y = np.random.uniform(low=0.47, high=0.90)
                    dist = math.sqrt(math.pow((x-last_x),2)+math.pow((y-last_y),2))
                last_x, last_y = x, y
            self.load_model(self.valid_obj_list[i], self.valid_obj_list[i], [x, y, 0.01])


        print("=============================================")

        self.timer = timer
        self.timer.reset()

    def visualize_results(self, img, depth, bboxes, masks, grasps, labels):
        from CoGr.datasets.OCID_class_dict import colors_list, cls_list

        masks_semantic = masks * (labels[:, None, None]+1)
        masks_semantic = masks_semantic.astype('int').sum(axis=0) % 31

        colors_list = np.array(colors_list)
        color_masks = colors_list[np.array(masks_semantic)].astype('uint8')
        img_u8 = img.astype('uint8')
        img_fused = (color_masks * 0.6 + img_u8 * 0.8)

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
                color = colors_list[int(cls_id)].tolist()
                center_x, center_y, width, height, theta, cls_id = rect
                box = ((center_x, center_y), (width, height), -(theta))
                box = cv2.boxPoints(box)
                box = np.int0(box)
                # cv2.drawContours(img_fused, [box], 0, color, 2)

                inv_color = (255, 255-color[1], 255-color[2])

                p1, p2, p3, p4 = box
                length = width
                p5 = (p1+p2)/2
                p6 = (p3+p4)/2
                p7 = (p5+p6)/2

                rad = theta / 180 * np.pi
                p8 = (p7[0]-length*np.sin(rad), p7[1]+length*np.cos(rad))
                cv2.circle(img_fused, (int(p7[0]), int(p7[1])), 2, (0,0,255), 2)
                cv2.line(img_fused, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), (0,0,255), 3, 8)
                cv2.line(img_fused, (int(p3[0]),int(p3[1])), (int(p4[0]),int(p4[1])), (0,0,255), 3, 8)
                cv2.line(img_fused, (int(p5[0]),int(p5[1])), (int(p6[0]),int(p6[1])), (255,0,0), 2, 8)


        ax = fig.add_subplot(1, 3, 1)
        ax.imshow((img_u8/255.)[...,::-1])
        ax.set_title('RGB')
        ax.axis('off')

        if depth is not None:
            ax = fig.add_subplot(1, 3, 2)
            ax.imshow(depth, cmap='gray')
            ax.set_title('Depth')
            ax.axis('off')
        
        ax = fig.add_subplot(2, 3, 3)
        ax.imshow(img_fused/255.)
        ax.set_title('Results')
        ax.axis('off')

        cv2.imwrite("results.png", img_fused)

        plt.savefig("overall_cogr_result.png")

    
    def pad_to_square(self, rgb, depth):
        h, w, c = rgb.shape
        pad_size = max(h, w)
        pad_img = np.zeros((pad_size, pad_size, c), dtype='float32')
        pad_img[:, :, :] = self.norm_mean
        pad_depth = np.zeros((pad_size, pad_size), dtype='float32')

        pad_img[0: h, 0: w, :] = rgb
        pad_depth[0: h, 0: w] = depth

        pad_img = cv2.resize(pad_img, (self.input_size, self.input_size))
        pad_depth = cv2.resize(pad_depth, (self.input_size, self.input_size))

        return pad_img, pad_depth



    def delete_model(self, model_name):
        rospy.wait_for_service("/gazebo/delete_model")
        srv = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)
        feedback = srv(model_name)
        success = feedback.success
        msg = feedback.status_message

        return success, msg
        
    

    def load_model(self, model_name, name, pos, orn=None):
        rospy.wait_for_service("/gazebo/spawn_sdf_model")
        srv = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
        with open("/home/hilbertxu/.gazebo/models/{}/model.sdf".format(model_name), 'r') as xml_f:
            model_xml = xml_f.read().replace('\n', '')
        
        if orn:
            o = tf.transformations.quaternion_from_euler(orn[0],orn[1],orn[2])
            orn = Quaternion()
            orn.x = o[0]
            orn.y = o[1]
            orn.z = o[2]
            orn.w = o[3]
        
            obj_pos = Pose(Point(pos[0],pos[1],pos[2]), orn)
        
        else:
            orn = Quaternion()
            orn.w = 1.0
            obj_pos = Pose(Point(pos[0],pos[1],pos[2]), orn)
        
        feedback = srv(name, model_xml, "", obj_pos, "world")

        success = feedback.success
        msg = feedback.status_message

        return success, msg




    def get_world_properties(self):
        rospy.wait_for_service("/gazebo/get_world_properties")
        srv = rospy.ServiceProxy("/gazebo/get_world_properties", GetWorldProperties)
        status = srv().model_names

        obj_list = [obj for obj in status if obj not in ['ground_plane', 'asus_xtion', 'table', 'robot']]

        return obj_list

    
    def image_callback(self, rgb_data, depth_data, pcl_data):
        print(pcl_data.header.frame_id)
        depth_image = self.bridge.imgmsg_to_cv2(depth_data, "passthrough")
        depth_image = depth_image.astype(np.uint8)
        depth_image = 1 - depth_image

        rgb_image = self.bridge.imgmsg_to_cv2(rgb_data, "bgr8")
        h, w, c = rgb_image.shape
        
        rgb, depth = self.pad_to_square(rgb_image, depth_image)
        rgb = normalize_and_toRGB(rgb)

        pc = np.asarray(list(point_cloud2.read_points(pcl_data))).reshape(h, w, 4)

        rgbd = np.concatenate([rgb, np.expand_dims(depth, -1)], axis=-1)

        obj_list = self.get_world_properties()


        with torch.no_grad():
            rgbd_tensor = torch.tensor(rgbd).float().permute(2,0,1).unsqueeze(0).cuda()
            
            tic = time.perf_counter()
            class_pred, box_pred, coef_pred, gr_pos_coef_pred, gr_sin_coef_pred, gr_cos_coef_pred, gr_wid_coef_pred, proto_out = self.net(rgbd_tensor)
            toc = time.perf_counter()
            t_f = toc - tic
            

            tic = time.perf_counter()
            ids_p, class_p, box_p, coef_p, pos_coef_p, sin_coef_p, cos_coef_p, wid_coef_p, proto_p = gr_nms_v2(
                    class_pred, box_pred, coef_pred, proto_out,
                    gr_pos_coef_pred, gr_sin_coef_pred, gr_cos_coef_pred, gr_wid_coef_pred,
                    self.net.anchors, self.cfg
                )

            img, depth, box_p, instance_masks, grasps, pos_masks, ang_masks, wid_masks, ids_p = gr_post_processing(
                rgb, depth, ids_p, class_p, box_p, coef_p, pos_coef_p, sin_coef_p, cos_coef_p, wid_coef_p, proto_p, 
                ori_h=480, ori_w=640, visualize_lincomb=False, visualize_results=False,
                num_grasp_per_object=1
            )
            toc = time.perf_counter()
            t_p = toc - tic
            
            # t_f, t_p = self.timer.get_times(['forward', 'postprocess'])
            print("Detected {} in {:.3f} sec forward and {:.3f} sec postprocessing".format(ids_p, t_f, t_p))

            all_grasps = []
            for obj_rects in grasps:
                all_grasps.extend(obj_rects)
            
            self.visualize_results(img, depth, box_p, instance_masks, all_grasps, ids_p)
            
            # Move dual arm to init pose
            status = self.behaviors_service(self.init_behavior)
            rospy.sleep(1)

            # Calculate valid area for generating objects
            # [100, 200] -> [540, 479]

            # print(pc[200, 100], pc[479, 540])
            # lt_point = PointStamped()
            # lt_point.header.frame_id = "head_mount_kinect_rgb_optical_frame"
            # lt_point.header.stamp = rospy.Time(0)
            # lt_point.point.x = pc[200, 100][0]
            # lt_point.point.y = pc[200, 100][1]
            # lt_point.point.z = pc[200, 100][2]

            # rb_point = PointStamped()
            # rb_point.header.frame_id = "head_mount_kinect_rgb_optical_frame"
            # rb_point.header.stamp = rospy.Time(0)
            # rb_point.point.x = pc[479, 540][0]
            # rb_point.point.y = pc[479, 540][1]
            # rb_point.point.z = pc[479, 540][2]

            # lt_point_trans = self.listener.transformPoint("world", lt_point)
            # rb_point_trans = self.listener.transformPoint("world", rb_point)


            for grasp in all_grasps:
                # Get 3D grasp point in camera frame
                center_x, center_y, width, height, theta, cls_id = grasp
                if cls_list[int(cls_id)] not in name_mapping.keys():
                    continue
                coord = pc[int(center_y), int(center_x)][:3]
                grasp_point = PointStamped()
                grasp_point.header.frame_id = "head_mount_kinect_rgb_optical_frame"
                grasp_point.header.stamp = rospy.Time(0)
                grasp_point.point.x = coord[0]
                grasp_point.point.y = coord[1]
                grasp_point.point.z = coord[2]

                # Transform from camera frame to world frame
                world_point = self.listener.transformPoint("world", grasp_point)
                pose = p.getQuaternionFromEuler([0, np.pi/2, -theta-np.pi/2]) 

                # Go to initial pose
                status = self.behaviors_service(self.init_behavior)
                rospy.sleep(2)

                if int(center_x) < w/2:
                    robot_arm = "left"
                    grasp_behavior = behaviorsRequest()
                    grasp_behavior.robot = robot_arm
                    grasp_behavior.behavior = "visualize_target_ee_pose"
                    grasp_behavior.left_target_pose.pose.position.x = world_point.point.x
                    grasp_behavior.left_target_pose.pose.position.y = world_point.point.y
                    grasp_behavior.left_target_pose.pose.position.z = world_point.point.z + 0.11

                    grasp_behavior.left_target_pose.pose.orientation.x = pose[0]
                    grasp_behavior.left_target_pose.pose.orientation.y = pose[1]
                    grasp_behavior.left_target_pose.pose.orientation.z = pose[2]
                    grasp_behavior.left_target_pose.pose.orientation.w = pose[3]

                    status = self.behaviors_service(grasp_behavior)
                    print("Visualizing grasp target")

                    if status.result is True:
                        rospy.sleep(0.5)
                        grasp_behavior.behavior = "IK_move_to_pose"
                        grasp_behavior.left_target_pose.pose.position.z = world_point.point.z + 0.4
                        print("Moving to pre-grasp pose")
                        status = self.behaviors_service(grasp_behavior)

                        if status.result is True:
                            rospy.sleep(0.5)
                            grasp_behavior.behavior = "IK_move_to_pose"
                            grasp_behavior.left_target_pose.pose.position.z = world_point.point.z + 0.11
                            print("Moving to target object")
                            status = self.behaviors_service(grasp_behavior)
                            rospy.sleep(1)

                            if status.result is True:
                                rospy.sleep(1)
                                print("Grasping target object")
                                grasp_behavior.behavior = "grasp_object"
                                grasp_behavior.object_name = name_mapping[cls_list[int(cls_id)]]
                                status = self.behaviors_service(grasp_behavior)

                                if status.result is True:  
                                    rospy.sleep(1)
                                    print("Lifting up...")
                                    grasp_behavior.behavior = "IK_move_to_pose"
                                    grasp_behavior.left_target_pose.pose.position.z = world_point.point.z + 0.4
                                    status = self.behaviors_service(grasp_behavior)
                                    rospy.sleep(1)
                                    if status.result is True:
                                        self.rest_behavior.robot = "left"
                                        status = self.behaviors_service(self.rest_behavior)
                                        if status.result is True:
                                            rospy.sleep(0.5)
                                            print("Releasing object...")
                                            grasp_behavior.behavior = "release_object"
                                            status = self.behaviors_service(grasp_behavior)

                                            if status.result is True:
                                                rospy.sleep(2)
                                                self.delete_model(grasp_behavior.object_name)
                                                status = self.behaviors_service(self.init_behavior)
                                                rospy.sleep(1)
                                
                                else:
                                    print("Failed to grasp {}".format(grasp_behavior.object_name))
                else:
                    robot_arm = "right"
                    grasp_behavior = behaviorsRequest()
                    grasp_behavior.robot = robot_arm
                    grasp_behavior.behavior = "visualize_target_ee_pose"
                    grasp_behavior.right_target_pose.pose.position.x = world_point.point.x
                    grasp_behavior.right_target_pose.pose.position.y = world_point.point.y
                    grasp_behavior.right_target_pose.pose.position.z = world_point.point.z + 0.11

                    grasp_behavior.right_target_pose.pose.orientation.x = pose[0]
                    grasp_behavior.right_target_pose.pose.orientation.y = pose[1]
                    grasp_behavior.right_target_pose.pose.orientation.z = pose[2]
                    grasp_behavior.right_target_pose.pose.orientation.w = pose[3]

                    status = self.behaviors_service(grasp_behavior)
                    print("Visualizing grasp target")

                    if status.result is True:
                        rospy.sleep(0.5)
                        grasp_behavior.behavior = "IK_move_to_pose"
                        grasp_behavior.right_target_pose.pose.position.z = world_point.point.z + 0.4
                        print("Moving to pre-grasp pose")
                        status = self.behaviors_service(grasp_behavior)

                        if status.result is True:
                            rospy.sleep(0.5)
                            grasp_behavior.behavior = "IK_move_to_pose"
                            grasp_behavior.right_target_pose.pose.position.z = world_point.point.z + 0.11
                            print("Moving to target object")
                            status = self.behaviors_service(grasp_behavior)
                            rospy.sleep(1)

                            if status.result is True:
                                rospy.sleep(0.5)
                                print("Grasping target object")
                                grasp_behavior.behavior = "grasp_object"
                                grasp_behavior.object_name = name_mapping[cls_list[int(cls_id)]]
                                status = self.behaviors_service(grasp_behavior)

                                if status.result is True:  
                                    rospy.sleep(1)
                                    print("Lifting up...")
                                    grasp_behavior.behavior = "IK_move_to_pose"
                                    grasp_behavior.left_target_pose.pose.position.z = world_point.point.z + 0.4
                                    status = self.behaviors_service(grasp_behavior)
                                    rospy.sleep(1)
                                    if status.result is True:
                                        self.rest_behavior.robot = "right"
                                        status = self.behaviors_service(self.rest_behavior)
                                        if status.result is True:
                                            rospy.sleep(0.5)
                                            print("Releasing object...")
                                            grasp_behavior.behavior = "release_object"
                                            status = self.behaviors_service(grasp_behavior)

                                            if status.result is True:
                                                rospy.sleep(2)
                                                self.delete_model(grasp_behavior.object_name)
                                                status = self.behaviors_service(self.init_behavior)
                                                rospy.sleep(1)
                                
                                else:
                                    print("Failed to grasp {}".format(grasp_behavior.object_name))


                
            




if __name__ == "__main__":

    demo = CoGrDemo()

    rospy.spin()