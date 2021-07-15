import numpy as np 
from ensemble_boxes import *
import glob
import os
import argparse
SAVE_FOLDER = ""

class WBF_VTX(object):
    name2idx = {"person": 0}
    idx2name = {0: "person"}

    def __init__(self, opts, model_weighted, iou_thr=0.6, skip_box_thr=0.0001):
        self.opt = opts
        self.model_weighted = model_weighted
        self.iou_thr = iou_thr
        self.skip_box_thr = skip_box_thr
        self.img_size = self.opt.img_size # w, h

        self._check_condition_input()
    
    def _check_condition_input(self):

        keys = list(self.model_weighted.keys())
        #print(keys)

        for i in range(0, len(keys)):
            pred_paths = glob.glob("{}/*".format(self.model_weighted[keys[i]]["path_prediction"]))
            num = len(pred_paths)
            if(num != self.opt.frame_length):
                tmp_path = "/".join(pred_paths[0].split("/")[:-1])
                self._fill_result_files(tmp_path)

    def _fill_result_files(self, folder):
        max_num = 6 # frame_{max_num}
        exist_frame = os.listdir(folder)
        for idx in range(self.opt.frame_length):
            frame_name = "frame_" + "0" * (6 - len(str(idx))) + str(idx) + ".txt"
            if(frame_name not in exist_frame):
                print(frame_name)
                with open(os.path.join(folder, frame_name), "wt") as f:
                    f.write("")

    def _normalize_coords(self, box):
        xmin, ymin, xmax, ymax = box
        nxmin, nymin, nxmax, nymax = xmin / self.img_size[0], ymin / self.img_size[1], xmax / self.img_size[0], ymax / self.img_size[1]

        return nxmin, nymin, nxmax, nymax
    
    def _scale_coords(self, box):
        nxmin, nymin, nxmax, nymax = box
        xmin, ymin, xmax, ymax = nxmin * self.img_size[0], nymin * self.img_size[1], nxmax * self.img_size[0], nymax * self.img_size[1]

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
        save_path = os.path.join(SAVE_FOLDER, self.opt.save_name)
        # check folder exist
        if(not os.path.exists(save_path)):
            os.mkdir(save_path)

        with open(os.path.join(save_path, frame_name), "wt") as f:
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
            #print("Process: ", frame_name)
            boxes, scores, labels = self._get_frame_wbf(frame_name, keys, weights)
            self._write_results(frame_name, [boxes, scores, labels])

        #print("Boxes: ", len(boxes_list))
        #print("Score: ", len(score_list))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Change config for ensembles')
    parser.add_argument('--save_name', type=str, required=True,
                                        help='name of result files')
    parser.add_argument('--frame_length', type=str, default=19200, help="num frame of videos")
    parser.add_argument('-n', '--path_list', nargs='+', default=[])
    parser.add_argument('--img_size', nargs='+', type=str, help="Image resolution (w, h)")
    
    args = parser.parse_args()
    
    args.frame_length = int(args.frame_length)
    args.img_size = list(map(int, args.img_size))
    print(args.path_list)
    print(args.img_size)
    model_weighted = {
        "crowd": {
            "weight": 2,
            "path_prediction": args.path_list[0], 
        }, 
        "baseline":{
            "weight": 3,
            "path_prediction": args.path_list[1],
        }
    }
    wbf = WBF_VTX(args, model_weighted)
    wbf.run_wbf("")

