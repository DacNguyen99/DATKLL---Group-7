# Face_Recognition Tutorial
Nhận diện khuôn mặt khá chuẩn xác bằng MTCNN và Facenet!
Chạy trên Tensorflow 2.x

Tiền xử lý dữ liệu: python src/align_dataset_mtcnn.py  Dataset/FaceData/raw Dataset/FaceData/processed --image_size 160 --margin 32  --random_order --gpu_memory_fraction 0.25

Train model: python src/classifier.py TRAIN Dataset/FaceData/processed Models/20180402-114759.pb Models/facemodel.pkl --batch_size 1000

Real-time face  recognition: python src/face_rec_cam.py