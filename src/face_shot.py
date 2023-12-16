import os.path
from pymongo_get_database import get_database
import cv2

dbName = get_database()
people = dbName["peoples"]
# print(people.find_one({'name': "DAC"}))

new = 1
if new == 0: # Register
    id = input('Please enter  your id: ')
    name = input('Please enter your name: ')
    new_obj = {"id_person": int(id), "name": name, "recog": 0}
elif new == 1: # Update images
    id = input('Please enter your id: ')
    found = people.find_one({'id_person': int(id)})
    name = found['name']
    print(name)

cam = cv2.VideoCapture(0)

cv2.namedWindow("Face Shot", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Face Shot", 500, 300)

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        break
    cv2.imshow("Face Shot", frame)

    k = cv2.waitKey(1)
    if k % 256 == 32:
        # SPACE pressed
        print("Press SPACE to take shots when the cam is perfect!")
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        else:
            folder_path = 'Dataset/FaceData/raw/' + name
            if os.path.exists(folder_path):
                img_name = folder_path + "/image_{}.jpg".format(img_counter)
                cv2.imwrite(img_name, frame)
                print("{} written!".format(img_name))
                img_counter += 1
            else:
                os.mkdir(folder_path)
                img_name = folder_path + "/image_{}.jpg".format(img_counter)
                cv2.imwrite(img_name, frame)
                print("{} written!".format(img_name))
                img_counter += 1

        if img_counter == 100:
            break
    elif k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

cam.release()

cv2.destroyAllWindows()
