import numpy as np


class Calibration:

    def __init__(self, cfg):

        # defining camera matrices
        fxl = cfg['left_cam_calib']['fx']
        fyl = cfg['left_cam_calib']['fy']
        cxl = cfg['left_cam_calib']['cx']
        cyl = cfg['left_cam_calib']['cy']
        k1l = cfg['left_cam_calib']['k1']
        k2l = cfg['left_cam_calib']['k2']
        p1l = cfg['left_cam_calib']['p1']
        p2l = cfg['left_cam_calib']['p2']
        self.left_cam_res = cfg['left_cam_calib']['res']

        fxr = cfg['right_cam_calib']['fx']
        fyr = cfg['right_cam_calib']['fy']
        cxr = cfg['right_cam_calib']['cx']
        cyr = cfg['right_cam_calib']['cy']
        p2r = cfg['right_cam_calib']['p2']
        k1r = cfg['right_cam_calib']['k1']
        k2r = cfg['right_cam_calib']['k2']
        p1r = cfg['right_cam_calib']['p1']
        self.right_cam_res = cfg['right_cam_calib']['res']

        self.left_cam_mat = np.array([[fxl, 0, cxl],
                                    [0, fyl, cyl],
                                    [0,  0,  1]])
        self.left_cam_dist = np.array([k1l, k2l, p1l, p2l])

        self.right_cam_mat = np.array([[fxr, 0, cxr],
                                    [0, fyr, cyr],
                                    [0,  0,  1]])
        self.right_cam_dist = np.array([k1r, k2r, p1r, p2r])

        # extrinsic
        self.RadarT = np.array(cfg['radar_calib']['T'])
        self.RadarR = np.array(cfg['radar_calib']['R'])
        self.LidarT = np.array(cfg['lidar_calib']['T'])
        self.LidarR = np.array(cfg['lidar_calib']['R'])
        self.LeftT = np.array(cfg['left_cam_calib']['T'])
        self.LeftR = np.array(cfg['left_cam_calib']['R'])
        self.RightT = np.array(cfg['right_cam_calib']['T'])
        self.RightR = np.array(cfg['right_cam_calib']['R'])

        # stereo
        self.stereoR = np.array(cfg['stereo_calib']['R'])
        self.stereoT = np.array([cfg['stereo_calib']['TX'],
                                 cfg['stereo_calib']['TY'],
                                 cfg['stereo_calib']['TZ']])


        self.LidarToLeftT = self.LidarT - self.LeftT
        self.LidarToRightT = self.LidarT - self.RightT

        self.LidarToLeftR = self.LidarR - self.LeftR
        self.LidarToRightR = self.LidarR - self.RightR

        self.LidarToLeft = self.transform(
            self.LidarToLeftR, self.LidarToLeftT)

        self.LidarToRight = self.transform(
            self.LidarToRightR, self.LidarToRightT)

        self.RadarToLeftT = self.RadarT - self.LeftT
        self.RadarToLeftR = self.RadarR - self.LeftR
        self.RadarToLeft = self.transform(
            self.RadarToLeftR, self.RadarToLeftT)

        self.RadarToRightT = self.RadarT - self.RightT
        self.RadarToRightR = self.RadarR - self.RightR
        self.RadarToRight = self.transform(
            self.RadarToRightR, self.RadarToRightT)

        self.RadarToLidarT = self.RadarT - self.LidarT
        self.RadarToLidarR = self.RadarR - self.LidarR
        self.RadarToLidar = self.transform(
            self.RadarToLidarR, self.RadarToLidarT)

    def RX(self, LidarToCamR):
        thetaX = np.deg2rad(LidarToCamR[0])
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(thetaX), -np.sin(thetaX)],
                       [0, np.sin(thetaX), np.cos(thetaX)]]).astype(np.float)
        return Rx

    def RY(self, LidarToCamR):
        thetaY = np.deg2rad(LidarToCamR[1])
        Ry = np.array([[np.cos(thetaY), 0, np.sin(thetaY)],
                       [0, 1, 0],
                       [-np.sin(thetaY), 0, np.cos(thetaY)]])
        return Ry

    def RZ(self, LidarToCamR):
        thetaZ = np.deg2rad(LidarToCamR[2])
        Rz = np.array([[np.cos(thetaZ), -np.sin(thetaZ), 0],
                       [np.sin(thetaZ), np.cos(thetaZ), 0],
                       [0, 0, 1]]).astype(np.float)
        return Rz

    def transform(self, LidarToCamR, LidarToCamT):
        Rx = self.RX(LidarToCamR)
        Ry = self.RY(LidarToCamR)
        Rz = self.RZ(LidarToCamR)

        R = np.array([[1, 0, 0],
                      [0, 0, 1],
                      [0, -1, 0]]).astype(np.float)
        R = np.matmul(R, np.matmul(Rx, np.matmul(Ry, Rz)))

        LidarToCam = np.array([[R[0, 0], R[0, 1], R[0, 2], 0.0],
                               [R[1, 0], R[1, 1], R[1, 2], 0.0],
                               [R[2, 0], R[2, 1], R[2, 2], 0.0],
                               [LidarToCamT[0], LidarToCamT[1], LidarToCamT[2], 1.0]]).T
        return LidarToCam
