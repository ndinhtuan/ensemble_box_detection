import numpy as np 
from ensemble_boxes import *

class WBF_VTX(object):

    def __init__(self, model_weighted, iou_thr=0.6):
        
        self.model_weighted = model_weighted

        self._check_condition_input()
    
    def _check_condition_input(self):

        keys = self.model_weighted.keys()
        num = len(glob.glob("{}/*".format(self.model_weighted[keys[0]])))

        for i in range(1, len(keys)):
            num_ = len(glob.glob("{}/*".format(self.model_weighted[keys[i]])))
            assert num == num_, "Length of frame between {} and {} is different".format(keys[0]. key[i])

    def _process_data_frame(self, list_prediction_file):
        
        boxes_list = []
        score_list = []
        labels_list = []

        for file_ in list_prediction_file:

            f = open(file_, "r")
            data = f.read().splitlines()

            for d in data:
                tmp = d.split(" ")
                score_list.append(float(tmp[1]))
                box = [float(_) for _ in tmp[2:]]
                boxes_list.append(box)
        
        return boxes_list, score_list, labels_list
        
    def run_wbf(self, path_output):
            
        keys = self.model_weighted.keys()
        first_path = self.model_weighted[keys[0]]

        

if __name__=="__main__":
    
    model_weighted = {
        "mask_score_rcnn": {
            "weight": 0.7434
            "path_prediction": "/home/hangd/hainp/data/predicts/mask_scoring_rcnn"
        }, 
        "hybrid_task_cascade": {
            "weight": 0.7776
            "path_prediction": "/home/hangd/hainp/data/predicts/htc_x100"
        }
    
