
import numpy as np
import cv2

# to capture frames create an object
cap = cv2.VideoCapture(0)

"""
    OpenCV already contains many pre-trained classifiers for face, eyes, smile etc. Those XML files are stored in
    opencv/data/haarcascades/ folder. Let's create face and eye detector with OpenCV.
    openCV provides you a class for face detection called haarascade. It extracts image features.
    face_cas is an object for haarascade to detect images first
"""
face_cas = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

data = []

"""
    In this we run an infinite loop and call camera object cap that will capture/read one frame/image.
    cap.read() will return 2 values first is boolean that tell if camera is working or not
    second is an input frame as numpy matrix. Every image consists of pixels and each pixel
    is a collection of 3 colors i.e. bgr. So each pixel has 3 matrices for blue, green and red.
    Convert this frame(image) into gray image using cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).
    Now I will apply face detection function of haarascade i.e detectMultiScale on this grayscale image.
    This function take a frame with 2 extra parameters scaleFactor and minNeighbors and
    return list of faces that are present in one frame.
    scaleFactor(1-2) : Parameter specifying how much the image size is reduced at each image scale.
    minNeighbors(3-6) : Parameter specifying how many neighbors each candidate rectangle should have to retain it.
    Now every index of list contains location of this face. It returns corner coordinates(x,y) and width and height.
    For every face in faces, we extract this face from original frame and do resizing and then store it in data.
    For visualization we create a rectangle around a face in that frame.
    Now show a frame/image in different window called "Your face".
    Now if user enter escape key or we get more than 20 faces sample loop will break
"""
while True :

    ret, frame = cap.read()

    if ret == True :

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in frame and return list of objects and each object contain coordinates of a face
        # we need this function to get coordinates of faces in one frame/image
        faces = face_cas.detectMultiScale(gray, 1.2, 4)

        for (x, y, w, h) in faces :

            # extract area of face from frame with bgr, : means take all values of bgr
            face_component = frame[y:y+h, x:x+w, : ]

            # resize face image into 50x50x3
            fc = cv2.resize(face_component,  (50, 50))

            # store rescaled face in data
            data.append(fc)

            # for visualization draw a rectangle around a face in frame/image
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 3)

        # display frame
        cv2.imshow("Your face", frame)

        if cv2.waitKey(1) == 27 or len(data) >= 200 :
            break

    else :
        print "error"


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

# convert list of data into numpy array
data = np.asarray(data)

# to check we get 200 images or not
print data.shape

# save data into numpy encoded format
np.save("face_4", data)


# now we will run the code for different people and store data into multiple files
