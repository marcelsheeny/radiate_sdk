import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
import math
import yaml
from utils.calibration import Calibration


class Sequence:
    """
    This class loads the sequence of RADIATE dataset

    | Example:
    | >>> import radiate
    | >>> root_path = 'path/to/radiate/city_3_7/'
    | >>> seq = radiate.Sequence(root_path)
    | >>> output = seq.get_from_timestamp(seq.init_timestamp)
    | >>> seq.vis_all(output)
    """

    def __init__(self, sequence_path, config_file='config/config.yaml'):
        """
        Initialise the class Sequence. This class contains the methods related to
        access the sensor and annotation information at certain timestamp

        :type sequence_path: string
        :param sequence_path: path/to/sequence_root

        :type config_file: string
        :param config_file: the path to the configuration files
        """
        self.sequence_path = sequence_path

        # load annotations
        self.annotations_path = os.path.join(
            self.sequence_path, 'annotations', 'annotations.json')
        self.__load_annotations()

        # load parameters and calibration file
        with open(config_file, 'r') as file:
            self.config = yaml.full_load(file)
        with open(self.config['calib_file'], 'r') as file:
            self.calib = yaml.full_load(file)
        self.config.update(self.calib)

        # generate calibration matrices from calib file
        self.calib = Calibration(self.config)

        # output folder
        self.output_folder = os.path.join(
            self.config['output_folder'], os.path.basename(self.sequence_path))

        # colors used for display
        self.colors = {'car': (1, 0, 0),
                       'bus': (0, 1, 0),
                       'truck': (0, 0, 1),
                       'pedestrian': (1.0, 1.0, 0.0),
                       'van': (1.0, 0.3, 0.0),
                       'group_of_pedestrians': (1.0, 1.0, 0.3),
                       'motorbike': (0.0, 1.0, 1.0),
                       'bicycle': (0.3, 1.0, 1.0),
                       'vehicle': (1.0, 0.0, 0.0)
                       }

        # average object height
        self.heights = {'car': 1.5,
                        'bus': 3,
                        'truck': 2.5,
                        'pedestrian': 1.8,
                        'van': 2,
                        'group_of_pedestrians': 1.8,
                        'motorbike': 1.5,
                        'bicycle': 1.5,
                        'vehicle': 1.5
                        }

        # load timestamps
        self.timestamp_camera = self.load_timestamp(os.path.join(
            self.sequence_path, self.config['camera_timestamp_file']))
        self.timestamp_radar = self.load_timestamp(os.path.join(
            self.sequence_path, self.config['radar_timestamp_file']))
        self.timestamp_lidar = self.load_timestamp(os.path.join(
            self.sequence_path, self.config['lidar_timestamp_file']))

        # get minimum timestamp
        self.init_timestamp = np.min([self.timestamp_camera['time'][0],
                                      self.timestamp_lidar['time'][0],
                                      self.timestamp_radar['time'][0]])

        # get end timestamp
        self.end_timestamp = np.max([self.timestamp_camera['time'][-1],
                                     self.timestamp_lidar['time'][-1],
                                     self.timestamp_radar['time'][-1]])

    def __load_annotations(self):
        if (os.path.exists(self.annotations_path)):
            f = open(self.annotations_path)
            self.annotations = json.load(f)
        else:
            self.annotations = None

    def overlay_camera_lidar(self, camera, lidar):
        """
        Method that joins camera and projected lidar in one image for visualisation

        :type camera: np.array
        :param camera: camera image 

        :type lidar: np.array
        :param lidar: lidar image with the same size as camera

        :return: overlayed image
        :rtype: np.array
        """
        overlay = np.copy(camera)
        overlay[np.nonzero(lidar)] = lidar[np.nonzero(lidar)]
        return overlay

    def project_lidar(self, lidar, lidar_extrinsics, cam_intrinsic, color_mode='same'):
        """
        Method to project the lidar into the camera

        :type lidar: np.array
        :param lidar: lidar point cloud with shape Nx5 (x,y,z,intensity,ring)

        :type lidar_extrinsics: np.array
        :param lidar_extrinsics: 4x4 matrix with lidar extrinsic parameters (Rotation
            and translations)

        :type cam_intrinsic: np.array
        :param cam_intrinsic: 3x3 matrix with camera intrinsic parameters in the form
            [[fx 0 cx],
            [0 fx cy],
            [0 0 1]]

        :type color_mode: string
        :param color_mode: what type of information is going to be representend in the lidar image
        options: 'same' always constant color. 'pseudo_distance': uses a color map to create a psedo
        color which refers to the distance. 'distance' creates an image with the actual distance as float

        :rtype: np.array
        :return: returns the projected lidar into the respective camera with the same size as the camera
        """
        fx = cam_intrinsic[0, 0]
        fy = cam_intrinsic[1, 1]
        cx = cam_intrinsic[0, 2]
        cy = cam_intrinsic[1, 2]
        if color_mode == 'same' or color_mode == 'pseudo_distance':
            im_lidar = np.zeros((self.config['left_cam_calib']['res'][1],
                                 self.config['left_cam_calib']['res'][0], 3))
        else:
            im_lidar = np.zeros((self.config['left_cam_calib']['res'][1],
                                 self.config['left_cam_calib']['res'][0]))
        lidar_points = lidar[:, :3].T
        R = lidar_extrinsics[:3, :3]
        lidar_points = np.matmul(R, lidar_points).T
        lidar_points += lidar_extrinsics[:3, 3]
        for i in range(lidar.shape[0]):
            if (lidar_points[i, 2] > 0 and lidar_points[i, 2] < self.config['lidar_proj']['max_dist']):
                xx = int(((lidar_points[i, 0] * fx) / lidar_points[i, 2]) + cx)
                yy = int(((lidar_points[i, 1] * fy) / lidar_points[i, 2]) + cy)
                if (xx > 0 and xx < self.config['left_cam_calib']['res'][0] and
                        yy > 0 and yy < self.config['left_cam_calib']['res'][1]):
                    if color_mode == 'same':
                        im_lidar = cv2.circle(
                            im_lidar, (xx, yy), 1, color=(0, 255, 0))
                    elif color_mode == 'pseudo_distance':
                        dist = np.sqrt(lidar_points[i, 0]*lidar_points[i, 0] +
                                       lidar_points[i, 1]*lidar_points[i, 1] +
                                       lidar_points[i, 2]*lidar_points[i, 2])
                        norm_dist = np.array(
                            [(dist/self.config['lidar_proj']['max_dist'])*255]).astype(np.uint8)
                        cc = np.array(plt.get_cmap('viridis')(norm_dist))*255
                        im_lidar = cv2.circle(
                            im_lidar, (xx, yy), 1, color=cc.tolist()[0][:3][::-1])
                    elif color_mode == 'distance':
                        dist = np.sqrt(lidar_points[i, 0]*lidar_points[i, 0] +
                                       lidar_points[i, 1]*lidar_points[i, 1] +
                                       lidar_points[i, 2]*lidar_points[i, 2])
                        im_lidar[yy, xx] = dist

        return im_lidar

    def get_from_timestamp(self, t, get_sensors=True, get_annotations=True):
        """method to get sensor and annotation information from some timestamp

        :param t: This is the timestamp which access the sensors/annotations
        :type t: float
        :param get_sensors: whether to retrieve sensor information, defaults to True
        :type get_sensors: bool, optional
        :param get_annotations: whether to retrieve annotation info, defaults to True
        :type get_annotations: bool, optional
        :return: returns a single variable as a dictionary with 'sensors' and 'annotations' as key
        :rtype: dict
        """
        output = {}
        self.current_time = t
        id_camera, ts_camera = self.get_id(
            t, self.timestamp_camera, self.config['sync']['camera'])
        id_lidar, ts_lidar = self.get_id(
            t, self.timestamp_lidar, self.config['sync']['lidar'])
        id_radar, ts_radar = self.get_id(
            t, self.timestamp_radar, self.config['sync']['radar'])
        if (len(self.timestamp_radar['time']) > id_radar + 1):
            t2 = self.timestamp_radar['time'][id_radar + 1]
        else:
            return output

        if get_sensors:
            str_format = '{:06d}'

            # generata paths from frames
            im_left_path = os.path.join(
                self.sequence_path, 'zed_left', str_format.format(id_camera) + '.png')

            im_right_path = os.path.join(
                self.sequence_path, 'zed_right', str_format.format(id_camera) + '.png')

            radar_cartesian_path = os.path.join(
                self.sequence_path, 'Navtech_Cartesian', str_format.format(id_radar) + '.png')

            lidar_path = os.path.join(
                self.sequence_path, 'velo_lidar', str_format.format(id_lidar) + '.csv')

            sensors = {}
            if (self.config['use_camera_left_raw'] or
                self.config['use_camera_right_raw'] or
                self.config['use_camera_left_rect'] or
                    self.config['use_camera_right_rect']):
                im_left = cv2.imread(im_left_path)
                im_right = cv2.imread(im_right_path)

            if (self.config['use_camera_left_rect'] or self.config['use_camera_right_rect']):
                im_left_rect, im_right_rect, disp_to_depth = self.get_rectfied(
                    im_left, im_right)

            if (self.config['use_lidar_bev_image'] or
                self.config['use_proj_lidar_left'] or
                    self.config['use_proj_lidar_right']):
                lidar = self.read_lidar(lidar_path)

            if (self.config['use_camera_left_raw']):
                sensors['camera_left_raw'] = im_left

            if (self.config['use_camera_right_raw']):
                sensors['camera_right_raw'] = im_right

            if (self.config['use_camera_left_rect']):
                sensors['camera_left_rect'] = im_left_rect

            if (self.config['use_camera_right_rect']):
                sensors['camera_right_rect'] = im_right_rect

            if (self.config['use_radar_cartesian']):
                radar_cartesian = cv2.imread(radar_cartesian_path)
                sensors['radar_cartesian'] = radar_cartesian

            if (self.config['use_lidar_bev_image']):
                sensors['lidar_bev_image'] = self.lidar_to_image(lidar)

            if (self.config['use_proj_lidar_left']):
                proj_lidar_left = self.project_lidar(lidar, self.calib.LidarToLeft, self.calib.left_cam_mat,
                                                     color_mode=self.config['lidar_proj']['color_mode'])
                sensors['proj_lidar_left'] = proj_lidar_left

            if (self.config['use_proj_lidar_right']):
                proj_lidar_right = self.project_lidar(lidar, self.calib.LidarToRight, self.calib.right_cam_mat,
                                                      color_mode=self.config['lidar_proj']['color_mode'])
                sensors['proj_lidar_right'] = proj_lidar_right

            output['sensors'] = sensors

        if (get_annotations):
            annotations = {}
            if (self.annotations != None):

                if self.config['use_radar_cartesian']:
                    radar_annotation_id = self.__get_correct_radar_id_from_raw_ind(
                        id_radar)
                    radar_annotations = self.get_annotation_from_id(
                        radar_annotation_id)
                    annotations['radar_cartesian'] = radar_annotations

                if self.config['use_lidar_bev_image']:
                    annotations['lidar_bev_image'] = self.get_lidar_annotations(
                        id_radar, self.config['interpolate_bboxes'], t, ts_radar, t2)

                if self.config['use_camera_left_rect']:
                    annotations['lidar_bev_image'] = self.get_lidar_annotations(
                        id_radar, self.config['interpolate_bboxes'], t, ts_radar, t2)
                    bboxes_3d = self.project_bboxes_to_camera(annotations['lidar_bev_image'],
                                                              self.calib.left_cam_mat,
                                                              self.calib.RadarToLeft)
                    annotations['camera_left_rect'] = bboxes_3d

                if self.config['use_camera_right_rect']:
                    annotations['lidar_bev_image'] = self.get_lidar_annotations(
                        id_radar, self.config['interpolate_bboxes'], t, ts_radar, t2)
                    bboxes_3d = self.project_bboxes_to_camera(annotations['lidar_bev_image'],
                                                              self.calib.right_cam_mat,
                                                              self.calib.RadarToRight)
                    annotations['camera_right_rect'] = bboxes_3d

            output['annotations'] = annotations

        return output

    def vis_all(self, output, wait_time=1):
        """method to diplay all the sensors/annotations

        :param output: gets the output from self.get_from_timestamp(t)
        :type output: dict
        :param wait_time: how to long to wait until display next frame. 0 means it will wait for any key, defaults to 1
        :type wait_time: int, optional
        """
        if (output != {}):
            if self.config['save_images']:
                os.makedirs(os.path.join(self.output_folder,
                                         str(self.current_time)), exist_ok=True)
            if self.config['use_camera_left_raw']:
                cv2.imshow('camera left raw',
                           output['sensors']['camera_left_raw'])
                if self.config['save_images']:
                    cv2.imwrite(os.path.join(self.output_folder,  str(
                        self.current_time), 'camera_left_raw.png'), output['sensors']['camera_left_raw'])

            if self.config['use_camera_right_raw']:
                cv2.imshow('camera right raw',
                           output['sensors']['camera_right_raw'])
                if self.config['save_images']:
                    cv2.imwrite(os.path.join(self.output_folder,  str(
                        self.current_time), 'camera_right_raw.png'), output['sensors']['camera_right_raw'])

            if self.config['use_camera_left_rect']:
                left_bb = self.vis_3d_bbox_cam(
                    output['sensors']['camera_left_rect'], output['annotations']['camera_left_rect'])
                cv2.imshow('camera left', left_bb)
                if self.config['save_images']:
                    cv2.imwrite(os.path.join(self.output_folder,  str(
                        self.current_time), 'left_bb.png'), left_bb)

            if self.config['use_camera_right_rect']:
                right_bb = self.vis_3d_bbox_cam(
                    output['sensors']['camera_right_rect'], output['annotations']['camera_right_rect'])
                cv2.imshow('camera right', right_bb)
                if self.config['save_images']:
                    cv2.imwrite(os.path.join(self.output_folder,  str(
                        self.current_time), 'right_bb.png'), right_bb.astype(np.uint8))

            if self.config['use_radar_cartesian']:
                radar_cart_vis = self.vis(
                    output['sensors']['radar_cartesian'], output['annotations']['radar_cartesian'])
                cv2.imshow('radar', radar_cart_vis)
                if self.config['save_images']:
                    cv2.imwrite(os.path.join(self.output_folder,  str(
                        self.current_time), 'radar_cart_vis.png'), radar_cart_vis)

            if self.config['use_radar_polar']:
                cv2.imshow('radar', output['sensors']['radar_polar'])
                if self.config['save_images']:
                    cv2.imwrite(os.path.join(self.output_folder,  str(
                        self.current_time), 'radar_polar.png'), output['sensors']['radar_polar'])

            if (self.config['use_lidar_bev_image']):
                lidar_vis = self.vis(
                    output['sensors']['lidar_bev_image'], output['annotations']['lidar_bev_image'])
                cv2.imshow('lidar image', lidar_vis)
                if self.config['save_images']:
                    cv2.imwrite(os.path.join(self.output_folder,  str(
                        self.current_time), 'lidar_vis.png'), lidar_vis)

            if self.config['use_proj_lidar_left']:
                overlay_left = self.overlay_camera_lidar(output['sensors']['camera_left_rect'],
                                                         output['sensors']['proj_lidar_left'])
                overlay_left_bb = self.vis_3d_bbox_cam(
                    overlay_left, output['annotations']['camera_left_rect'])
                cv2.imshow('projected lidar to left camera', overlay_left_bb)
                if self.config['save_images']:
                    cv2.imwrite(os.path.join(self.output_folder,  str(
                        self.current_time), 'overlay_left_bb.png'), overlay_left_bb)
            if self.config['use_proj_lidar_right']:
                overlay_right = self.overlay_camera_lidar(output['sensors']['camera_right_rect'],
                                                          output['sensors']['proj_lidar_right'])
                overlay_right_bb = self.vis_3d_bbox_cam(
                    overlay_right, output['annotations']['camera_right_rect'])
                cv2.imshow('projected lidar to right camera', overlay_right_bb)
                if self.config['save_images']:
                    cv2.imwrite(os.path.join(self.output_folder, str(
                        self.current_time), 'overlay_right_bb.png'), overlay_right_bb)

        cv2.waitKey(wait_time)

    def project_bboxes_to_camera(self, annotations, intrinsict, extrinsic):
        """method to project the bounding boxes to the camera

        :param annotations: the annotations for the current frame
        :type annotations: list
        :param intrinsict: intrisic camera parameters
        :type intrinsict: np.array
        :param extrinsic: extrinsic parameters
        :type extrinsic: np.array
        :return: dictionary with the list of bbounding boxes with camera coordinate frames
        :rtype: dict
        """
        bboxes_3d = []
        for object in annotations:
            obj = {}
            class_name = object['class_name']
            obj['class_name'] = class_name
            obj['id'] = (object['id'] if 'id' in object.keys() else 0)
            height = self.heights[class_name]
            bb = object['bbox']['position']
            rotation = object['bbox']['rotation']
            bbox_3d = self.__get_projected_bbox(
                bb, rotation, intrinsict, extrinsic, height)
            obj['bbox_3d'] = bbox_3d
            bboxes_3d.append(obj)

        return bboxes_3d

    def vis_3d_bbox_cam(self, image, bboxes_3d, pc_size=0.7):
        """diplay pseudo 3d bounding box from camera

        :param image: camera which the bounding box is going to be projected
        :type image: np.array
        :param bboxes_3d: list of bounding box information with pseudo-3d image coordinate frame
        :type bboxes_3d: dict
        :param pc_size: percentage of the size of the bounding box [0.0 1.0]
        :type pc_size: float
        :return: camera image with the correspondent bounding boxes
        :rtype: np.array
        """
        vis_im = np.copy(image)
        for obj in bboxes_3d:
            bbox_3d = obj['bbox_3d']
            for ii in range(len(bbox_3d)):
                color = self.colors[obj['class_name']]
                vis_im = cv2.line(vis_im, (bbox_3d[ii - 1][0], bbox_3d[ii - 1][1]),
                                  (bbox_3d[ii][0], bbox_3d[ii][1]), (np.array(color) * 255).astype(np.int).tolist(), 1)

        return vis_im

    def vis_bbox_cam(self, image, bboxes_3d, pc_size=0.7):
        """diplay pseudo 2d bounding box from camera

        :param image: camera which the bounding box is going to be projected
        :type image: np.array
        :param bboxes_3d: list of bounding box information with pseudo-3d image coordinate frame
        :type bboxes_3d: dict
        :param pc_size: percentage of the size of the bounding box [0.0 1.0]
        :type pc_size: float
        :return: camera image with the correspondent bounding boxes
        :rtype: np.array
        """
        vis_im = np.copy(image)
        for obj in bboxes_3d:
            color = self.colors[obj['class_name']]
            bb = np.zeros((4))
            if obj['bbox_3d'].shape[0] > 0:
                bb[0] = np.min(obj['bbox_3d'][:, 0])
                bb[1] = np.min(obj['bbox_3d'][:, 1])
                bb[2] = np.max(obj['bbox_3d'][:, 0])
                bb[3] = np.max(obj['bbox_3d'][:, 1])
                wid = bb[2] - bb[0]
                # hei = bb[3] - bb[1]
                bb[0] += wid*(1.0 - pc_size)
                bb[2] -= wid*(1.0 - pc_size)
                bb = bb.astype(np.int)
                vis_im = cv2.rectangle(
                    vis_im, (bb[0], bb[1]), (bb[2], bb[3]), (np.array(color) * 255))

        return vis_im

    def get_lidar_annotations(self, id_radar, interp=False, t_c=None, t_r1=None, t_r2=None):
        """get the annotations in lidar image coordinate frame

        :param id_radar: the annotation radar id
        :type id_radar: int
        :param interp: whether to use interpolation or not
        :type interp: bool
        :param t: timestamp
        :type t: float
        :return: the annotations in lidar image coordinate frame
        :rtype: dict
        """
        lidar_annotation_id = self.__get_correct_lidar_id_from_raw_ind(
            id_radar)
        lidar_annotations = self.get_annotation_from_id(lidar_annotation_id)
        if interp and len(self.get_annotation_from_id(lidar_annotation_id+1)) > 0:
            lidar_annotations_next = self.get_annotation_from_id(
                lidar_annotation_id+1)

            for ii in range(len(lidar_annotations)):
                try:
                    p1x = lidar_annotations[ii]['bbox']['position'][0]
                    p1y = lidar_annotations[ii]['bbox']['position'][1]
                    p2x = lidar_annotations_next[ii]['bbox']['position'][0]
                    p2y = lidar_annotations_next[ii]['bbox']['position'][1]
                    lidar_annotations[ii]['bbox']['position'][0] = self.__linear_interpolation(
                        p1x, t_c, t_r1, t_r2, p2x)
                    lidar_annotations[ii]['bbox']['position'][1] = self.__linear_interpolation(
                        p1y, t_c, t_r1, t_r2, p2y)
                    # __linear_interpolation(self, p1, t_c, t_r1, t_r2, p2)
                except:
                    pass
        M = self.calib.RadarToLidar

        h_width = self.config['lidar_bev_image']['res'][0]/2.0
        h_height = self.config['lidar_bev_image']['res'][1]/2.0
        cell_res_x = 100.0/h_width
        cell_res_y = 100.0/h_height

        # convert meters to pixel
        M[3, 0] = M[3, 0] / cell_res_x
        M[3, 1] = M[3, 1] / cell_res_y
        lidar_annotations = self.transform_annotations(lidar_annotations, M)
        return lidar_annotations

    def get_rectfied(self, left_im, right_im):
        """get the left and right image rectfied

        :param left_im: raw left image
        :type left_im: np.array
        :param right_im: raw right image
        :type right_im: np.array
        :return: tuple (left_rect, right_rect, disp_to_depth)
            WHERE
            np.array left_rect is the rectfied left image
            np.array right_rect is the rectfied right image
            np.array disp_to_depth is a matrix that converts the disparity values to distance in meters
        :rtype: tuple
        """
        (leftRectification, rightRectification, leftProjection,
         rightProjection, dispartityToDepthMap, leftROI, rightROI) = cv2.stereoRectify(
            cameraMatrix1=self.calib.left_cam_mat,
            distCoeffs1=self.calib.left_cam_dist,
            cameraMatrix2=self.calib.right_cam_mat,
            distCoeffs2=self.calib.right_cam_dist,
            imageSize=tuple(self.calib.left_cam_res),
            R=self.calib.stereoR,
            T=self.calib.stereoT,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0
        )

        leftMapX, leftMapY = cv2.initUndistortRectifyMap(
            self.calib.left_cam_mat,
            self.calib.left_cam_dist,
            leftRectification,
            leftProjection, tuple(self.calib.left_cam_res), cv2.CV_32FC1)

        rightMapX, rightMapY = cv2.initUndistortRectifyMap(
            self.calib.right_cam_mat,
            self.calib.left_cam_dist,
            rightRectification,
            rightProjection, tuple(self.calib.left_cam_res), cv2.CV_32FC1)

        fixedLeft = cv2.remap(left_im, leftMapX,
                              leftMapY, cv2.INTER_LINEAR)
        fixedRight = cv2.remap(right_im, rightMapX,
                               rightMapY, cv2.INTER_LINEAR)

        return fixedLeft, fixedRight, dispartityToDepthMap

    def transform_annotations(self, annotations, M):
        """method to transform the annotations to annother coordinate

        :param annotations: the list of annotations
        :type annotations: list
        :param M: transformation matrix
        :type M: np.array
        :return: the list of annotations in another coodinate frame
        :rtype: list
        """
        new_annotations = []
        for object in annotations:
            new_object = object
            xx = object['bbox']['position'][0]
            yy = object['bbox']['position'][1]
            zz = self.config['sensors_height']
            pos = np.array([xx, yy, zz, 1])
            new_pos = np.matmul(M, pos)
            new_pos = new_pos/new_pos[3]
            new_object['bbox']['position'][0] = pos[0]
            new_object['bbox']['position'][1] = pos[1]
            new_annotations.append(new_object)
        return new_annotations

    def transform_point_cloud(self, pc, M):
        """transform a 3d point cloud to another coordinate frame

        :param pc: point cloud in the form Nx% (x,y,z,intensity, ring)
        :type pc: np.array
        :param M: transformation matrix
        :type M: np.array
        :return: transformed point cloud
        :rtype: np.array
        """
        new_pc = []
        for point in pc:
            # new_object = object
            xx = point[0]
            yy = point[1]
            zz = point[2]
            pos = np.array([xx, yy, zz, 1])
            new_pos = np.matmul(M, pos)
            new_pos = new_pos/new_pos[3]
            new_pc.append([pos[0], pos[1], pos[2], point[3], point[4]])
        new_pc = np.array(new_pc)
        return new_pc

    def get_annotation_from_id(self, annotation_id):
        """ get the annotation from an id

        :param annotation_id: frame id
        :type annotation_id: int
        :return: list of annotations for the id given as parameter
        :rtype: list
        """
        raw_annotations = []
        for object in self.annotations:
            if (object['bboxes'][annotation_id]):
                obj = {}
                obj['id'] = object['id']
                obj['class_name'] = object['class_name']
                obj['bbox'] = object['bboxes'][annotation_id]
                raw_annotations.append(obj)
        return raw_annotations

    def __inner_lidar_bev_image(self, lidar,
                                image,
                                i,
                                cell_res_x,
                                cell_res_y,
                                h_width,
                                h_height):
        xyzi = lidar[i]
        x = xyzi[0]/cell_res_x + h_width
        y = h_height - xyzi[1]/cell_res_y
        if self.config['lidar_bev_image']['use_ring']:
            c = int(xyzi[4]) * 8
        else:
            c = int(xyzi[3])
        image = cv2.circle(image, (int(x), int(y)), 1, (c, c, c))
        return image

    def lidar_to_image(self, lidar):
        """Convert an lidar point cloud to an 2d bird's eye view image

        :param lidar: lidar point cloud Nx5 (x,y,z, intensity, ring)
        :type lidar: np.array
        :return: 2d bird's eye image with the lidar information
        :rtype: np.array
        """
        image = np.zeros((self.config['lidar_bev_image']['res']
                          [0], self.config['lidar_bev_image']['res'][1], 3))
        h_width = self.config['lidar_bev_image']['res'][0]/2.0
        h_height = self.config['lidar_bev_image']['res'][1]/2.0
        cell_res_x = 100.0/h_width
        cell_res_y = 100.0/h_height
        for i in range(lidar.shape[0]):
            if self.config['lidar_bev_image']['remove_ground']:
                if lidar[i, 2] > -self.config['lidar_bev_image']['ground_thresh']:
                    image = self.__inner_lidar_bev_image(
                        lidar, image, i, cell_res_x, cell_res_y, h_width, h_height)
            else:
                image = self.__inner_lidar_bev_image(
                    lidar, image, i, cell_res_x, cell_res_y, h_width, h_height)
        return image.astype(np.uint8)

    def __get_correct_radar_id_from_raw_ind(self, id):
        return id-1

    def __get_correct_lidar_id_from_raw_ind(self, id):
        return id-1

    def vis(self, sensor, objects, color=None, mode='rot'):
        """ visualise the sensor and its annotation

        :param sensor: 
        :type sensor: the given sensor
        :param objects: np.array
        :type objects: list of objects
        :return: image with the objects overlayed
        :rtype: np.array
        """
        sensor_vis = np.copy(sensor)
        for object in objects:
            bbox = object['bbox']['position']
            angle = object['bbox']['rotation']

            class_name = object['class_name']
            color = self.colors[class_name]
            if mode == 'rot':
                sensor_vis = self.draw_boundingbox_rot(
                    sensor_vis, bbox, angle, color)

        return sensor_vis

    def read_lidar(self, lidar_path):
        """given a lidar raw path returns it lidar point cloud

        :param lidar_path: path to lidar raw point
        :type lidar_path: string
        :return: lidar point cloud Nx5 (x,y,z,intensity,ring)
        :rtype: np.array
        """
        return pd.read_csv(lidar_path, delimiter=',').values

    def get_id(self, t, all_timestamps, time_offset=0.0):
        """get the closest id given the timestamp

        :param t: timestamp in seconds
        :type t: float
        :param all_timestamps: a list with all timestamps
        :type all_timestamps: np.array
        :param time_offset: offset in case there is some unsynchronoised sensor, defaults to 0.0
        :type time_offset: float, optional
        :return: the closest id
        :rtype: int
        """
        ind = np.argmin(np.abs(all_timestamps['time'] - t + time_offset))
        return all_timestamps['frame'][ind], all_timestamps['time'][ind]

    def __timestamp_format(self, raw_timestamp):
        """
        function to fix the timestamp
        """
        raw_decimal_place_len = len(raw_timestamp.split('.')[-1])
        if(raw_decimal_place_len < 9):
            place_diff = 9 - raw_decimal_place_len
            zero_str = ''
            for _ in range(place_diff):
                zero_str = zero_str + '0'
            formatted_timestamp = raw_timestamp.split(
                '.')[0] + '.' + zero_str + raw_timestamp.split('.')[1]
            return float(formatted_timestamp)
        else:
            return float(raw_timestamp)

    def load_timestamp(self, timestamp_path):
        """load all timestamps from a sensor

        :param timestamp_path: path to text file with all timestamps
        :type timestamp_path: string
        :return: list of all timestamps
        :rtype: dict
        """
        with open(timestamp_path, "r") as file:
            lines = file.readlines()
            timestamps = {'frame': [], 'time': []}
            for line in lines:
                words = line.split()
                timestamps['frame'].append(int(words[1]))
                timestamps['time'].append(self.__timestamp_format(words[3]))
        return timestamps

    def __get_projected_bbox(self, bb, rotation, cameraMatrix, extrinsic, obj_height=2):
        """get the projected boundinb box to some camera sensor
        """
        rotation = np.deg2rad(-rotation)
        res = self.config['radar_calib']['range_res']
        cx = bb[0] + bb[2] / 2
        cy = bb[1] + bb[3] / 2
        T = np.array([[cx], [cy]])
        pc = 0.2
        bb = [bb[0]+bb[2]*pc, bb[1]+bb[3]*pc, bb[2]-bb[2]*pc, bb[3]-bb[3]*pc]

        R = np.array([[np.cos(rotation), -np.sin(rotation)],
                      [np.sin(rotation), np.cos(rotation)]])

        points = np.array([[bb[0], bb[1]],
                           [bb[0] + bb[2], bb[1]],
                           [bb[0] + bb[2], bb[1] + bb[3]],
                           [bb[0], bb[1] + bb[3]],
                           [bb[0], bb[1]],
                           [bb[0] + bb[2], bb[1] + bb[3]]]).T

        points = points - T
        points = np.matmul(R, points) + T
        points = points.T

        points[:, 0] = points[:, 0] - self.config['radar_calib']['range_cells']
        points[:, 1] = self.config['radar_calib']['range_cells'] - points[:, 1]
        points = points * res

        points = np.append(points, np.ones(
            (points.shape[0], 1)) * -1.7, axis=1)
        p1 = points[0, :]
        p2 = points[1, :]
        p3 = points[2, :]
        p4 = points[3, :]

        p5 = np.array([p1[0], p1[1], p1[2] + obj_height])
        p6 = np.array([p2[0], p2[1], p2[2] + obj_height])
        p7 = np.array([p3[0], p3[1], p3[2] + obj_height])
        p8 = np.array([p4[0], p4[1], p4[2] + obj_height])
        points = np.array([p1, p2, p3, p4, p1, p5, p6, p2, p6,
                           p7, p3, p7, p8, p4, p8, p5, p4, p3, p2, p6, p3, p1])

        points = np.matmul(np.append(points, np.ones(
            (points.shape[0], 1)), axis=1), extrinsic.T)

        points = (points / points[:, 3, None])[:, 0:3]

        filtered_indices = []
        for i in range(points.shape[0]):
            if (points[i, 2] > 0 and points[i, 2] < self.config['max_range_bbox_camera']):
                filtered_indices.append(i)

        points = points[filtered_indices]

        fx = cameraMatrix[0, 0]
        fy = cameraMatrix[1, 1]
        cx = cameraMatrix[0, 2]
        cy = cameraMatrix[1, 2]

        xIm = np.round((fx * points[:, 0] / points[:, 2]) + cx).astype(np.int)
        yIm = np.round((fy * points[:, 1] / points[:, 2]) + cy).astype(np.int)

        proj_bbox_3d = []
        for ii in range(1, xIm.shape[0]):
            proj_bbox_3d.append([xIm[ii], yIm[ii]])
        proj_bbox_3d = np.array(proj_bbox_3d)
        return proj_bbox_3d

    def draw_boundingbox_rot(self, im, bbox, angle, color):
        points = self.gen_boundingbox_rot(bbox, angle)

        color = (np.array(color) * 255).tolist()

        cv2.line(im, tuple(points[:, 0]), tuple(points[:, 1]), color, 3)
        cv2.line(im, tuple(points[:, 1]), tuple(points[:, 2]), color, 3)
        cv2.line(im, tuple(points[:, 2]), tuple(points[:, 3]), color, 3)
        cv2.line(im, tuple(points[:, 3]), tuple(points[:, 0]), color, 3)

        return im

    def __linear_interpolation(self, p1, t_c, t_r1, t_r2, p2):
        return p1 + (t_c - t_r1)*((p2 - p1)/(t_r2 - t_r1))

    def gen_boundingbox_rot(self, bbox, angle):
        """
        generate a list of 2D points from bbox and angle 
        """
        theta = np.deg2rad(-angle)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        points = np.array([[bbox[0], bbox[1]],
                           [bbox[0] + bbox[2], bbox[1]],
                           [bbox[0] + bbox[2], bbox[1] + bbox[3]],
                           [bbox[0], bbox[1] + bbox[3]]]).T

        cx = bbox[0] + bbox[2] / 2
        cy = bbox[1] + bbox[3] / 2
        T = np.array([[cx], [cy]])

        points = points - T
        points = np.matmul(R, points) + T
        points = points.astype(int)

        return points
