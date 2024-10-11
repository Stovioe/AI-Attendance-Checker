# AI-Attendance-Checker
Face Detection:
The initial step involves detecting the face in an image using the Haar Cascade Classifier.
The Haar Cascade method is a traditional feature-based classifier that uses integral images and AdaBoost for rapid face detection. 

Face Alignment:
Once a face is detected, alignment ensures that the face is normalized for recognition, which is done by using warping and affine transformations based on facial landmarks like eyes, nose, and mouth.
Facial landmark detectors (OpenCV’s DNN-based detectors) provide key points for alignment to ensure that facial features are correctly oriented in the frame.

Feature Extraction:
This step transforms the aligned face image into a vector representation. 
The model extracts high-dimensional facial feature vectors from the face image. These embeddings capture critical facial characteristics in a way that is robust to variations in lighting, pose, and expression.

Face Recognition (Matching):
In face recognition, the embeddings from the query image are compared with embeddings from known faces. This is done using the distance metrics of Euclidean similarity.
OpenCV utilizes the 128-dimensional face embedding vector. These embeddings are crucial as they allow for a compact, yet highly discriminative representation of the face.
