import numpy as np
from imutils import paths
import tensorflow as tf
import pickle
import cv2
import os

dataset = "dataset/train/"
embeddings = "output/embeddings.pickle"
detector_path = "face_detection_model"
detector_model = 'frozen_inference_graph.pb'
embedding_model = "openface_nn4.small2.v1.t7"
CONFIDENCE_DEFAULT= 0.5


# Read the graph.
print("[INFO] loading face detector...")
modelPath = os.path.join(detector_path, detector_model)
with tf.gfile.FastGFile(modelPath, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())


# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(embedding_model)

# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(dataset))

# initialize our lists of extracted facial embeddings and
# corresponding people names
knownEmbeddings = []
knownNames = []

# initialize the total number of faces processed
total = 0


with tf.Session() as sess:
    # Restore session
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    for (i, imagePath) in enumerate(imagePaths):
        # Read and preprocess an image.
        print("[INFO] processing image {}/{}".format(i + 1,
                                                     len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]

        print(imagePath)
        img = cv2.imread(imagePath)
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
                cv2.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)
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

                # add the name of the person + corresponding face
                # embedding to their respective lists
                knownNames.append(name)
                knownEmbeddings.append(vec.flatten())
                total += 1


# dump the facial embeddings + names to disk
print("[INFO] serializing {} encodings...".format(total))
data = {"embeddings": knownEmbeddings, "names": knownNames}
with open(embeddings, "wb") as f:
    f.write(pickle.dumps(data))

# cv2.imshow('Image', img)
# cv2.waitKey()
# cv2.destroyAllWindows()