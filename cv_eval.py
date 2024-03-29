# import the necessary packages
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
from imutils import paths
from confusion_matrix import plotConfusionMatrix, confusionMatrix, classAccuracy, overallAccuracy



dataset_test_path = "dataset/test"

detector_path = "face_detection_model/"
detector_model = "res10_300x300_ssd_iter_140000.caffemodel"

embedding_model = "openface_nn4.small2.v1.t7"
svm_model_path = "output/recognizer.pickle"
label_encoder = "output/le.pickle"

CONFIDENCE_DEFAULT = 0.5

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.join(detector_path, "deploy.prototxt")
modelPath = os.path.join(detector_path, detector_model)
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(embedding_model)

# load the actual face recognition model along with the label encoder
svm_model = pickle.loads(open(svm_model_path, "rb").read())
le = pickle.loads(open("output/le.pickle", "rb").read())

imagePaths = list(paths.list_images(dataset_test_path))

gt_names = []
detected_names = []

for (i, image_path) in enumerate(imagePaths):
    # Read and preprocess an image.
    print("[INFO] processing image {}/{}".format(i + 1,
                                                 len(imagePaths)))
    name_gt = image_path.split(os.path.sep)[-2]

    print(image_path, name_gt)


    # load the image, resize it to have a width of 600 pixels (while
    # maintaining the aspect ratio), and then grab the image dimensions
    image = cv2.imread(image_path)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > CONFIDENCE_DEFAULT:
            # compute the (x, y)-coordinates of the bounding box for the
            # face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

            # construct a blob for the face ROI, then pass the blob
            # through our face embedding model to obtain the 128-d
            # quantification of the face
            faceBlob = cv2.dnn.blobFromImage(face,
                                             1.0 / 255,
                                             (96, 96),
                                             (0, 0, 0),
                                             swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # perform classification to recognize the face
            preds = svm_model.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name_detected = le.classes_[j]

            gt_names.append(name_gt)
            detected_names.append(name_detected)

            print(image_path, name_gt, name_detected)

cm, names = confusionMatrix(gt_names, detected_names)
plotConfusionMatrix(cm, names)
class_acc, _ = classAccuracy(cm, names)

overall_acc = overallAccuracy(gt_names, detected_names)
print("People: {}", names)
print("Person Accuracy: {}", class_acc)
print("Overall Accuracy {}", overall_acc)