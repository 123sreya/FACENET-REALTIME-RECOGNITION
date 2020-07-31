# FACENET-REALTIME-RECOGNITION
FaceNet is a face recognition system developed in 2015 by researchers at Google that achieved then state-of-the-art results on a range of face recognition benchmark datasets. The FaceNet system can be used broadly thanks to multiple third-party open source implementations of the model and the availability of pre-trained models.
The FaceNet system can be used to extract high-quality features from faces, called face embeddings, that can then be used to train a face identification system.
It is a system that, given a picture of a face, will extract high-quality features from the face and predict a 128 element vector representation these features, called a face embedding.
The model is a deep convolutional neural network trained via a triplet loss function that encourages vectors for the same identity to become more similar (smaller distance), whereas vectors for different identities are expected to become less similar (larger distance). The focus on training a model to create embeddings directly (rather than extracting them from an intermediate layer of a model) was an important innovation in this work.
These face embeddings were then used as the basis for training classifier systems on standard face recognition benchmark datasets, achieving then-state-of-the-art results.
It is a robust and effective face recognition system, and the general nature of the extracted face embeddings lends the approach to a range of applications.
In system, called Face Net, that directly learns a mapping from face images to a compact Euclidean space where distances directly correspond to a measure of face similarity. Once this space has been produced, tasks such as face recognition, verification and clustering can be easily implemented using standard techniques with Face Net embeddings as feature vectors.
The benefit of our approach is much greater representational efficiency: we achieve state-of-the-art face recognition performance using only 128-bytes per face.
. The network is trained such that the squared L2 distances in the embedding space directly correspond to face similarity : faces of the same person have small distances and faces of distinct people have large distances.

# Advantages of Facenet 
Previous face recognition approaches based on deep networks use a classification layer trained over a set of known face identities and then take an intermediate bottle-neck layer The downsides of this approach are its indirectness and its inefficiency.
By using a bottleneck layer the representation size per face is usually very large but this is a linear transformation that can be easily learnt in one layer of the network.

# Triplet-Loss
The Triplet Loss minimizes the distance between an anchor and a positive, both of which have the same identity, and maximizes the distance between the anchor and a negative of a different identity.
Choosing the correct image pairs is extremely important as there will be a lot of image pairs that will satisfy this condition and hence our model won’t learn much from them and will also converge slowly because of that.
It is infeasible to compute the argmin and argmax across the whole training set. Additionally, it might lead to poor training, as mislabeled and poorly imaged faces would dominate the hard positives and negatives.
To have a meaningful representation of the anchor positive distances, it needs to be ensured that a minimal number of exemplars of any one identity is present in each
A clever workaround here is to compute the hard positives and negatives for a mini-batch. Here, we will choose around 1000–2000 samples (In most experiments the batch size was around 1800).This method is found to give stable results.

# MTCNN
we will also use the Multi-Task Cascaded Convolutional Neural Network, or MTCNN, for face detection, e.g. finding and extracting faces from photos.
 Multi-task Cascaded Convolutional Networks (MTCNN), which is essentially several convolutional networks.This neural network detects individual faces, locates facial landmarks (i.e. two eyes, nose, and endpoints of the mouth), and draws a bounding box around the face
as well as the network’s confidence in classifying that area as a face.

# Libraries
PICKLE

SCIPY

SCIKITLEARN

MTCNN

# PROCEDURE OF EXECUTION
1.Run utils-consists all functions required.

2.Run prepare_date-Encodings are extracted.

3.Run Face_recognizer_camera-Faces are being recognized by comparision of encodings in Vedio capture.

4.Attendance_facenet_project-Faces recognized are updated in an excel sheet with current date and time 

  Note-before running above code create an empty csv file named "Attendance"
