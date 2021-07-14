import numpy as np 
from ensemble_boxes import *
import glob
import os

SAVE_FOLDER = "ensemble_results/"

class WBF_VTX(object):
    name2idx = {"person": 0}
    idx2name = {0: "person"}
    def __init__(self, model_weighted, img_size=(1080, 1920), iou_thr=0.6, skip_box_thr=0.0001):
        
        self.model_weighted = model_weighted
        self.iou_thr = iou_thr
        self.skip_box_thr = skip_box_thr
        self.img_size = img_size # (h, w)

        self._check_condition_input()
    
    def _check_condition_input(self):

        keys = list(self.model_weighted.keys())
        print(keys)
        num = len(glob.glob("{}/*".format(self.model_weighted[keys[0]])))

        for i in range(1, len(keys)):
            num_ = len(glob.glob("{}/*".format(self.model_weighted[keys[i]])))
            assert num == num_, "Length of frame between {} and {} is different".format(keys[0]. key[i])
    
    def _normalize_coords(self, box):
        xmin, ymin, xmax, ymax = box
        nxmin, nymin, nxmax, nymax = xmin / self.img_size[1], ymin / self.img_size[0], xmax / self.img_size[1], ymax / self.img_size[0]

        return nxmin, nymin, nxmax, nymax
    
    def _scale_coords(self, box):
        nxmin, nymin, nxmax, nymax = box
        xmin, ymin, xmax, ymax = nxmin * self.img_size[1], nymin * self.img_size[0], nxmax * self.img_size[1], nymax * self.img_size[0]

        return xmin, ymin, xmax, ymax   

    def _process_data_frame(self, list_prediction_file):
        
        boxes_list = []
        score_list = []
        labels_list = []

        for file_ in list_prediction_file:

            f = open(file_, "r")
            data = f.read().splitlines()
            
            boxes = []
            scores = []
            labels = []
            for d in data:
                tmp = d.split(" ")
                scores.append(float(tmp[1]))
                box = [float(_) for _ in tmp[2:]]
                boxes.append(self._normalize_coords(box))
                labels.append(WBF_VTX.name2idx[str(tmp[0])])
            
            boxes_list.append(boxes)
            score_list.append(scores)
            labels_list.append(labels)
        return boxes_list, score_list, labels_list
    
    def _get_frame_predict_file(self, frame_name, models):
        list_prediction_file = []
        for model in models:
            #print(model)
            model_path = self.model_weighted[model]["path_prediction"]
            list_prediction_file.append(os.path.join(model_path, frame_name))
        return list_prediction_file
    
    def _write_results(self, frame_name, results):
        boxes, scores, labels = results
        with open(os.path.join(SAVE_FOLDER, frame_name), "wt") as f:
            for idx in range(len(boxes)):
                boxes[idx] = self._scale_coords(boxes[idx])
                line = WBF_VTX.idx2name[labels[idx]] + " " + str(scores[idx]) + " " + " ".join(list(map(str, boxes[idx])))
                f.write(line + "\n")

    def _get_frame_wbf(self, frame_name, model_name, weights):
        list_pred = self._get_frame_predict_file(frame_name, model_name)
        boxes_list, scores_list, labels_list = self._process_data_frame(list_pred)
              
        assert len(boxes_list) == len(scores_list), "Not matched"

        boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=self.iou_thr, skip_box_thr=self.skip_box_thr)
        return boxes, scores, labels

    def run_wbf(self, path_output):
        
        keys = list(self.model_weighted.keys())
        weights = list(map(lambda x: self.model_weighted[x]["weight"], keys))
        
        frames = os.listdir(self.model_weighted[keys[0]]["path_prediction"])
        for frame_name in frames:
            print("Process: ", frame_name)
            boxes, scores, labels = self._get_frame_wbf(frame_name, keys, weights)
            self._write_results(frame_name, [boxes, scores, labels])

        #print("Boxes: ", len(boxes_list))
        #print("Score: ", len(score_list))

if __name__=="__main__":
    
    model_weighted = {
        "yolof": {
            "weight": 0.7848,
            "path_prediction": "outputs/mmdet/yolof"
        }, 
        "hybrid_task_cascade": {
            "weight": 0.7776,
            "path_prediction": "outputs/mmdet/htc_x100"
        }
    }
    wbf = WBF_VTX(model_weighted)
    wbf.run_wbf("")

