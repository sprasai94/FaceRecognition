# faceRec

### This repo contains the implementation of face recognition using two different neural networks: 
SSD Inception v2 for face detection, and FaceNet for feature extraction. Furthermore, an SVM is trained
on the extracted features for face recognition. The FaceNet model was obtained from the OpenFace project, a python
and torch implementation of face with deep learning. 

#### Requirements: 
The SSD Inception v2 model may not load on a CPU only system. Use of a powerful enough GPU is suggested. 
 - Tensorflow-gpu 1.15
 - OpenCV 4.1.1
 - imutils
 - Scikit learn
 - Matplotlib
 - Numpy

To extract embeddings run:
```bash
$ python tf_extract_embeddings.py
```

To train SVM on the extracted embeddings, run:
```buildoutcfg
$ python train_model.py
```

To recognize am image, run
```buildoutcfg
$ python tf_recognize.py
```

