# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle


embeddings = "output/embeddings.pickle"
recognizer = "output/recognizer.pickle"
label_encoder = "output/le.pickle"


# load the face embeddings
print("[INFO] loading face embeddings...")
data = pickle.loads(open(embeddings, "rb").read())

# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

# train the model used to accept the 128-d embeddings of the face and
# then produce the actual face recognition
print("[INFO] training model...")
svm_model = SVC(probability=True)
svm_model.fit(data["embeddings"], labels)

# write the actual face recognition model to disk
f = open(recognizer, "wb")
f.write(pickle.dumps(svm_model))
f.close()

# write the label encoder to disk
f = open(label_encoder, "wb")
f.write(pickle.dumps(le))
f.close()