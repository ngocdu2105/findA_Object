
import cv2
import time
import os
import numpy as np
from run_yolo import Yolov5
import glob
# from threading import Thread
# from arduino_cmd import write_data

# cv2.imwrite('ChessboardCalibration_master/camera.jpg',frame)

class Run(Yolov5):
    def __init__(self):
        super().__init__()
        # self.frame = cv2.imread(r'C:\Users\Admin\Desktop\grasp\grasp\frameokie\frame36.jpg')

    def findA(self):
        kc = self.distance(self.disReal,self.pointC)/(1+self.anpha)
        self.oxyC=[self.pointC+[1,0],self.pointC+[0,1]]
        print(self.pointC)
        self.Caculation_Angle(self.oxyC ,self.disReal)
        print(self.angleoxy)
        self.A=self.disReal - np.array([kc if i else -kc for i in self.angleoxy])
    
    def image(self):
        # print(os.path.join(os.path.dirname(__file__),'anh'))
        for index,k in enumerate(glob.glob('dataset/*.jpg')):
            self.start_time
            self.frame=cv2.imread(k)
            print(index)
            self.get_frame(self.frame)

            # print(self.frame)
            
            self.save_result='resulterr2'
            self.save_resultYolo='save_reYolo'
            for (classid, confidence, box,results_seg) in zip(self.result_class_ids, self.result_confidences, self.result_boxes,self.result_seg):

                    color = self.colors[int(classid) % len(self.colors)]
                    # print(results_seg)
                    center_cx,cx,poin1,point2=results_seg
                    # np.save(f'{self.save_resultYolo}/{os.path.basename(k)[:-4]}npy',(box[0]+int(box[2]/2),box[1]+int(box[3]/2)))
                    self.point3d=np.array([box[0]+int(box[2]/2)-self.offex,box[1]+int(box[3]/2)-self.offex])
                    self.CalOxyReal(self.point3d)
                    np.save(f'{self.save_resultYolo}/{os.path.basename(k)[:-4]}npy',(self.disReal[0],self.disReal[1]))
                    
                    if center_cx[0]==0:
                        cv2.circle(self.frame,(box[0]+int(cx[0])-self.offex,box[1]+int(cx[1])-self.offex),4,color,2)
                        # self.point3d=np.array([box[0]+int(box[2]/2)-self.offex,box[1]+int(box[3]/2)-self.offex])
                        self.point3d=np.array([box[0]+int(cx[0])-self.offex,box[1]+int(cx[1])-self.offex])
                        # np.save(f'{self.save_result}/{os.path.basename(k)[:-4]}npy',self.point3d)
                        self.CalOxyReal(self.point3d)
                        cv2.putText(self.frame, f'Toa do P({self.disReal[0]:0.2f},{self.disReal[1]:0.2f})', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
                        # np.save(f'{self.save_result}/{os.path.basename(k)[:-4]}npy',(self.disReal[0],self.disReal[1]))
                        # cv2.imwrite(f'{self.save_result}/{os.path.basename(k)[:-4]}.jpg',self.frame)              
                        continue
                    self.point3d=np.array([box[0]+int(center_cx[0])-self.offex,box[1]+int(center_cx[1])-self.offex])
                    # np.save(f'{self.save_result}/{os.path.basename(k)[:-4]}npy',self.point3d)
                    
                    self.CalOxyReal(self.point3d)
                    # np.save(f'{self.save_result}/{os.path.basename(k)[:-4]}npy',(self.disReal[0],self.disReal[1]))
                    # cv2.rectangle(self.frame, box, color, 2){len(os.listdir(self.save_result))}_
                    # cv2.circle(self.frame,(box[0]+int(box[2]/2),box[1]+int(box[3]/2)),3,(255,0,0),2)
                    cv2.circle(self.frame,(box[0]+int(center_cx[0])-self.offex,box[1]-self.offex+int(center_cx[1])),3,color,-1)
                    cv2.circle(self.frame,(box[0]+int(poin1[0])-self.offex,box[1]-self.offex+int(poin1[1])),3,(255,0,0),-1)
                    cv2.circle(self.frame,(box[0]+int(point2[0])-self.offex,box[1]-self.offex+int(point2[1])),3,(255,0,0),-1)
                    cv2.circle(self.frame,(box[0]+int(cx[0])-self.offex,box[1]-self.offex+int(cx[1])),3,color,2)
                    # cv2.circle(self.frame,(box[0]+box[2],box[1]),10,(0,0,255)){len(os.listdir(self.save_result))}_
                    cv2.putText(self.frame, f'Toa do P({self.disReal[0]:0.2f},{self.disReal[1]:0.2f})', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)

                    # cv2.imwrite(f'{self.save_result}/{os.path.basename(k)[:-4]}.jpg',self.frame) 
                    # print(box)
                    # cv2.imshow('sdf',self.frame[box[1]:box[3]+box[1],box[0]:box[0]+box[2]])
                    # self.findA()
                    # print(self.A)
                    [cv2.circle(self.frame,i.astype('int'),3,k1,10) for i,k1 in zip(self.oxy,[(0,0,255),(0,0,255),(0,0,255)])]
                    
            self.stop
            
            cv2.imshow("output", self.frame)
            cv2.waitKey(0)
        print(np.mean(self.value_time))
    # def save_npy(self,i):
    #     np.save(f'{self.save_result}/{os.path.basename(i)[:-4]}npy',self.point3d)
    def video(self):
        # from arduino_cmd import write_data
        while True:
            self.loop_frame()
            
            
            for (classid, confidence, box,result_seg) in zip(self.result_class_ids, self.result_confidences, self.result_boxes,self.result_seg):
                center_cx,_,poin1,point2=result_seg

                # cv2.circle(self.imgr,tuple(np.int32(center_t)), 2, (0,255,0), thickness=5)
                # cv2.circle(self.imgr,tuple(np.int32(self.cx)), 2, (255,255,0), thickness=5)
                # cv2.circle(self.imgr,tuple(np.int32(self.b1[self.point_contact][self.index_maxX])), 2, (255,0,0), thickness=2)
                # cv2.circle(self.imgr,tuple(np.int32(self.b1[self.point_Ncontact][self.index_maxY])), 2, (255,0,0), thickness=2)
                color = self.colors[int(classid) % len(self.colors)]
                self.point3d=np.array([box[0]+box[2]/2,box[1]+box[3]/2])
                self.CalOxyReal(self.point3d)
                cv2.rectangle(self.frame, box, color, 2)
                cv2.circle(self.frame,(box[0]+int(box[2]/2),box[1]+int(box[3]/2)),3,color,2)
            
            # if self.frame_count >= 30:
            #     end = time.time()
            #     fps =  self.frame_count / (end - self.start)
            #     self.frame_count = 0
            #     start = time.time()
            # self.disReal
            _x,_y=self.disReal
            _x,_y=_x*19,_y*19+150+10
            command=f'up {_x} {_y}'
            print(command)
            # write_data(command)
            # time.sleep(60)
            # time.sleep(10)
            # if fps > 0:
            #     fps_label = "FPS: %.2f" % fps
            #     cv2.putText(self.frame, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow("output", self.frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # self.frame
# class findA(Run):
#     def __init__(self) -> None:
#         super().__init__()
#         # self.point
#         self.image()
#     def cal(self):
#         self.distance()
a=Run()
# # # a.video()
a.image()
