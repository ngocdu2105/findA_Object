import cv2
import numpy as np
import random
import os
from cfg import config_grasp

class Calibrator(config_grasp):
    def __init__(self):
        super().__init__()
        self.path_img=cv2.imread(os.path.join(os.path.dirname(__file__),self.path_calibration))
        self.call()
        # self.CHECKERBOARD=CHECKERBOARD
    def call(self):
        color = [(0,0,155),(0,255,0),(0,244,155)]
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.img = cv2.imread(self.path_img) if self.path_img is str else self.path_img
        gray = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(gray, self.CHECKERBOARD, 
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
            self.imgpoints=corners2.reshape(-1,2)
        corners2=corners2.reshape(-1,2)
        
        pointOxy=corners2[3]
        # self.img=cv2.circle(self.img,pointOxy.astype('int'),3,(255,0,0),2)
        self.ori_oxy= np.argmin(abs(self.imgpoints-pointOxy),axis=0)[0]
        self.pointOxy=pointOxy
        # print(self.imgpoints.shape)
        self.dis_oxy=np.array([self.imgpoints[self.ori_oxy+1],self.imgpoints[self.ori_oxy+self.CHECKERBOARD[0]*2]])
        self.oxy=[pointOxy,corners2[self.ori_oxy+1],corners2[self.ori_oxy+self.CHECKERBOARD[0]*2]]
        self.appr_edge_length=self.Dis2Squeeze(corners2)
        
        # self.img
        # cv2.imshow('asdf',self.img)
        # cv2.waitKey(0)
        # print(corners2)
        print(self.appr_edge_length)
        print('Loading Success Calibation')
    def CalOxyReal(self,point3D):
        self.disReal=np.array([self.dis_pointLine(p2=self.pointOxy,p1=self.dis_oxy[0],p3=point3D)/self.appr_edge_length,
        self.dis_pointLine(p2=self.pointOxy,p1=self.dis_oxy[1],p3=point3D)/self.appr_edge_length])
        self.disReal= np.array([self.disReal[1],self.disReal[0]]) if self.CHECKERBOARD[0] > self.CHECKERBOARD[1] else self.disReal
        self.point3D=point3D
        self.coverOxy()
        print(self.disReal)
    
    def Caculation_Angle(self,a,b):    
        self.angleoxy=np.array([np.arccos(sum(a*b)/(np.linalg.norm(a)*np.linalg.norm(b))) for a in a])*180/np.pi < [90,90]
    def coverOxy(self):
        # print(self.dis_oxy)
        # print(self.pointOxy)
        self.Caculation_Angle(np.array(self.dis_oxy-self.pointOxy),np.array(self.point3D)-self.pointOxy)
        self.disReal*=[1 if i else -1 for i in self.angleoxy]

    def Dis2Squeeze(self,corners):
        idx=np.random.randint(len(corners),size=(len(corners),1))
        return np.mean(np.sort(np.sqrt(np.sum((np.array(corners[idx])-np.array(corners))**2,axis=-1)))[:,1])
        
    @classmethod
    def dis_pointLine(cls,p1,p2,p3):
        return np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)
    @classmethod
    def distance(cls,p1,p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))
