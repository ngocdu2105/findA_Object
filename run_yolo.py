
import cv2
import time
import sys
import numpy as np

from processMask import predict_PointA
# from cfg import config_grasp
from predict import Calibrator
class Yolov5(Calibrator):
    def __init__(self):
        super().__init__()
        # self.capture = cv2.VideoCapture(r"1.mp4")
        # self.capture=cv2.VideoCapture(0)
        # self.colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]
        # # self.capture=cv2.VideoCapture(1)cv2.imread(r'C:\Users\Admin\Desktop\grasp\grasp\frameokie\frame8.jpg')
        # # self.capture=path_img
        # self.INPUT_WIDTH = 640
        # self.INPUT_HEIGHT = 640
        # self.SCORE_THRESHOLD = 0.2
        # self.NMS_THRESHOLD = 0.4
        # self.CONFIDENCE_THRESHOLD = 0.4
        # self.class_list = ['obj_1','obj_2']
        self.build_model()
        self.value_time=[]
        
    @property
    def start_time(self) :
        # colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]
        
        # capture = self.load_capture()

        self.start = time.time()
        # self.frame_count = 0
        # self.total_frames = 0
        # self.fps = -1

        # _, self.frame = self.capture.read()
        # self.__iter__()
    @property
    def stop(self):
        self.value_time.append(time.time()-self.start)
    def get_frame(self,img):

        self.inputImage = self.format_yolov5(img)
        self.outs = self.detect(self.inputImage)
        self.wrap_detection(self.inputImage, self.outs[0])
        self.mask_object()
    def loop_frame(self):
        _, self.frame = self.capture.read()
        self.inputImage = self.format_yolov5(self.frame)
        self.outs = self.detect(self.inputImage)
        self.wrap_detection(self.inputImage, self.outs[0])
    def build_model(self):
        self.net = cv2.dnn.readNet(self.modelYolov5)
        is_cuda = len(sys.argv) > 1 and sys.argv[1] == "cuda"
        if is_cuda:
            print("Attempty to use CUDA")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        else:
            print("Running on CPU")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.preSeg=predict_PointA()



    def detect(self,image):
        self.blob = cv2.dnn.blobFromImage(image, 1/255.0, (self.INPUT_WIDTH, self.INPUT_HEIGHT), swapRB=True, crop=False)
        self.net.setInput(self.blob)
        preds = self.net.forward()
        return preds
    def wrap_detection(self,input_image, output_data):
        self.class_ids = []
        self.confidences = []
        self.boxes = []

        self.rows = output_data.shape[0]
        self.image_width, self.image_height, _ = input_image.shape

        self.x_factor = self.image_width / self.INPUT_WIDTH
        self.y_factor =  self.image_height / self.INPUT_HEIGHT
        for r in range(self.rows):
            row = output_data[r]
            confidence = row[4]
            # print(confidence)
            if confidence >= 0.4:

                classes_scores = row[5:]
                _, _, _, max_indx = cv2.minMaxLoc(classes_scores)

                class_id = max_indx[1]
                # print(class_id)
                if ( classes_scores[class_id] > .25):

                    self.confidences.append(confidence)

                    self.class_ids.append(class_id)

                    x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
                    left = int((x - 0.5 * w) * self.x_factor)
                    top = int((y - 0.5 * h) * self.y_factor)
                    width = int(w * self.x_factor)
                    height = int(h * self.y_factor)
                    box = np.array([left, top, width, height])
                    # result_seg.append(self.preSeg.call(self.frame[box[1]-20:box[3]+box[1]+20,box[0]-20:box[0]+box[2]+20]))
                    self.boxes.append(box)

        self.indexes = cv2.dnn.NMSBoxes(self.boxes, self.confidences, 0.25, 0.45) 

        self.result_class_ids = []
        self.result_confidences = []
        self.result_boxes = []
        # self.result_seg=[]
        
        # for box in self.result_boxes:
            # cv2.imshow('sdff',self.frame[box[1]-20:box[3]+box[1]+20,box[0]-20:box[0]+box[2]+20])
            # cv2.waitKey(0)
            
            
        for i in self.indexes:
            self.result_confidences.append(self.confidences[i])
            self.result_class_ids.append(self.class_ids[i])
            self.result_boxes.append(self.boxes[i])
            # self.result_seg.append(self.result_seg[i])



    def mask_object(self):
        self.result_seg=[]
        
        for box in self.result_boxes:
            # cv2.imshow('sdff',self.frame[box[1]-20:box[3]+box[1]+20,box[0]-20:box[0]+box[2]+20])
            # cv2.waitKey(0)
            
            try:
                self.result_seg.append(self.preSeg.call(self.frame[box[1]-20:box[3]+box[1]+20,box[0]-20:box[0]+box[2]+20]))
            except:continue

    def format_yolov5(self,frame):

        self.row, self.col, _ = frame.shape
        self._max = max(self.col, self.row)
        self.result = np.zeros((self._max, self._max, 3), np.uint8)
        self.result[0:self.row, 0:self.col] = frame
        return self.result



