# import the necessary packages
import numpy as np
import argparse
import imutils
import tensorflow as tf
import pickle
import cv2
import os


image_path = "images/adele-3.jpg"

detector_path = "face_detection_model/"
detector_model = "frozen_inference_graph.pb"

embedding_model = "openface_nn4.small2.v1.t7"
svm_model_path = "output/recognizer.pickle"
label_encoder = "output/le.pickle"

CONFIDENCE_DEFAULT = 0.5

# Read the graph.
print("[INFO] loading face detector...")
modelPath = os.path.join(detector_path, detector_model)
with tf.gfile.FastGFile(modelPath, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())


# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(embedding_model)

# load the actual face recognition model along with the label encoder
svm_model = pickle.loads(open(svm_model_path, "rb").read())
le = pickle.loads(open("output/le.pickle", "rb").read())




# construct a blob from the image
# imageBlob = cv2.dnn.blobFromImage(
#     cv2.resize(image, (300, 300)), 1.0, (300, 300),
#     (104.0, 177.0, 123.0), swapRB=False, crop=False)

# apply OpenCV's deep learning-based face detector to localize
# faces in the input image
with tf.Session() as sess:
    # Restore session
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    # Read and preprocess an image.
    img = cv2.imread(image_path)
    rows = img.shape[0]
    cols = img.shape[1]
    inp = cv2.resize(img, (300, 300))
    inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

    # Run the model
    out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                    sess.graph.get_tensor_by_name('detection_scores:0'),
                    sess.graph.get_tensor_by_name('detection_boxes:0'),
                    sess.graph.get_tensor_by_name('detection_classes:0')],
                   feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

    # Visualize detected bounding boxes.
    num_detections = int(out[0][0])
    for i in range(num_detections):
        classId = int(out[3][0][i])
        score = float(out[1][0][i])
        bbox = [float(v) for v in out[2][0][i]]
        if score > CONFIDENCE_DEFAULT:
            x = bbox[1] * cols
            y = bbox[0] * rows
            right = bbox[3] * cols
            bottom = bbox[2] * rows
            # cv2.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)
            face = img[int(y):int(bottom), int(x):int(right)]

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
            name = le.classes_[j]

            # draw the bounding box of the face along with the associated
            # probability
            text = "{}: {:.2f}%".format(name, proba * 100)
            y = y - 10 if y - 10 > 10 else y + 10
            cv2.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)

            cv2.putText(img, text, (int(x), int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    # show the output image
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
