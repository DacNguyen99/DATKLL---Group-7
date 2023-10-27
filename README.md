# DATKLL---Group-7
* 27/10/2023 - Upload the first source, the program can detect and recognize faces through traning and datas in database
Things to improve: Accuracy, Interacting Speed
Things have been used: OpenCV and Python
Tutorial: 
  + The first python file Detector.py is used first to detect the faces appeared on the camera. The program will detect faces and takes 20 photos, then put them in the folder dataset.
  + The second python file Training.py is used next to "train" the program through the images taken in the first step, then output a file called trainingdata.yml, which contains the training datas.
  + The last python file Recognize.py is used to detect faces appeared on  the camera and recognize them through the trainingdata.yml and the datas in the database.
