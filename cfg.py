
class config_grasp:
    def __init__(self):
        # self.path_calibration='anh\z3639039285061_358e7b4479ccd02d09e3e83c8ebc17fb.jpg' # frame dau
        self.path_calibration='dataset\z3785564845852_002762006e7993079fa5383cc5b4020c.jpg'
        self.path_saveImageResultsPointA='save_seg'
        self.modelYolov5='model/best_dataset.onnx'
        self.modelCascadeMaskRCNN='model/mask_rcnn_dataset_fix.onnx'
        self.infer_cfg='model\infer_cfg_dataset.yml'
        # self.modelCascadeMaskRCNN='model\casdemask_rcnn224.onnx'
        # self.infer_cfg='model\infer_cfg_cascade_mask_crnn.yml'
        self.colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]
        self.INPUT_WIDTH = 640
        self.INPUT_HEIGHT = 640
        self.SCORE_THRESHOLD = 0.2
        self.NMS_THRESHOLD = 0.4
        self.CONFIDENCE_THRESHOLD = 0.4
        self.class_list = ['obj_1','obj_2']
        self.CHECKERBOARD=(13,9)
        self.offex=20
        self.center_tt=(0,0)
        
        
        self.point_contact=0
        self.index_maxX=0
        
        self.point_Ncontact=0
        self.indexVthcY=0
        self.index_maxY=0
        self.index_nbPcontact=(0,0)
