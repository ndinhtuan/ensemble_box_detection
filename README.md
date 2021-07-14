# ensemble_box_detection

# Guide
- Vào file wbs.py, add các model muốn ensemble vào. 
- path_prediction là path đến thư mục mà chứa kết quả predict của model này.
- weight tương ứng với mAP của model ở file này 
https://docs.google.com/spreadsheets/d/1o3iFfN9I3ediKU_zcIU0IF20__WJtoBkfzTJsAr_LqA/edit?fbclid=IwAR0uYXEn4YUauFI7-I39TFXttfGkormb-MgA5CV_4iCJnHBJmWm2orKLD0w#gid=0
- Run: python wbs.py --save_name {name} --frame_length {length}

  name -> tên folder để save kết quả
  
  length -> số frame của video chạy 
- Evaluation: Theo hướng dẫn của repo https://github.com/Cartucho/mAP

# References:
- https://github.com/ZFTurbo/Weighted-Boxes-Fusion
- https://github.com/Cartucho/mAP
