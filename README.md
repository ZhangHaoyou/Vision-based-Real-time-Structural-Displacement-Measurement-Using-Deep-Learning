# Vision-based-Real-time-Structural-Displacement-Measurement-Using-Deep-Learning
The labelled dataset and trained deep learning model will be uploaded here once the journal article is online.

Displacements serve as a key factor in structural tests for assessing structural safety and conducting maintenance. This study presents a framework for real-time vision-based structural displacement measurement using deep learning and object tracking algorithms. The You Only Look Once (YOLO) object detection deep learning algorithm and the deep SORT (Simple Online Real-Time Tracking) algorithm were used to track a chessboard target in input video frames for estimating structural displacement in real-time. To train the deep learning models, a dataset comprising 2132 high-resolution experimental photos was annotated. Additionally, a novel Black-White Attention (BWA) module was proposed to enhance the original YOLO model, surpassing the performance of four existing attention modules in the computer vision field with an accuracy of 93.33%. Moreover, an attention map was generated using the eigen class activation mapping (CAM) method to interpret the proposed attention module. The issue of jitter or instability of bounding boxes in the original YOLO model was mitigated using weighted non-maximum suppression (WNMS). Furthermore, the proposed method achieved real-time processing with a practical frame detection rate of 35 frames per second. Importantly, this study also proposed a solution to occlusion in structural displacement measurement by using the deep SORT method. Finally, the proposed framework was validated using a dynamic shaking table test and a static test, yielding a root mean square error of 2.05 mm and 0.11 mm, respectively. The errors were less than the one-pixel error. Hence, the proposed model achieved sub-pixel accuracy, as the YOLO-based model is a regression model used for predicting bounding boxes of targets.
