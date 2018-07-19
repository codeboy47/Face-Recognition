
import numpy as np
import cv2

# to capture frames create an object
cap = cv2.VideoCapture(0)

# create haarcascade object
face_cas = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# we will flatten or reshape 200x50x50x3 into 200x(50*50*3) or 200x7500
# now it will give 200 linear vectors of length 7500 each
face_1 = np.load('face_1.npy').reshape((200,50*50*3))
face_2 = np.load('face_2.npy').reshape((200,50*50*3))
face_3 = np.load('face_3.npy').reshape((200,50*50*3))
print face_1.shape
# so there are 200 faces, each face is represented by a vector of 7500 length. Now each index of vector act as
# a single feature and we apply our Machine Learning algorithm on this.

# create a dictionary
names = {
    0 : "Akshit",
    1 : "Pranav",
    2 : "Reena"
}

# create numpy matrix of zeros of size 600
labels = np.zeros((600,1))
labels[:200, :] = 0.0  # first 200 for person 1
labels[200:400, :] = 1.0  # second 200 for person 2
labels[400:, :] = 2.0  # last 200 for person 3

# combine all information into 1 data array
data = np.concatenate([face_1,face_2,face_3])
print data.shape, labels.shape


#Euclidean Distance
def dist(x1,x2):
    return np.sqrt(((x1-x2)**2).sum())

# K-nearest neighbor
def knn(X_train,y_train,query_point,k=5):

    vals = []

    for ix in range(X_train.shape[0]):
        v = [ dist(query_point,X_train[ix,:]), y_train[ix]]
        vals.append(v)

    # vals is a list containing distances and their labels
    updated_vals = sorted(vals)

    # Lets us pick up top K values
    pred_arr = np.asarray(updated_vals[:k])
    pred_arr = np.unique(pred_arr[:,1],return_counts = True)

    #Largest Occurence
    index = pred_arr[1].argmax() #Index of largest freq

    return pred_arr[0][index]


while True :

    # get a frame image
    ret, frame = cap.read()


    if ret == True :

        # convert to grayscale and get faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cas.detectMultiScale(gray, 1.3, 5)

        # for each face
        for (x, y, w, h) in faces :

            face_component = frame[y:y+h, x:x+w, : ]

            fc = cv2.resize(face_component,  (50, 50))
            fc = fc.reshape((1,50*50*3))
            #print fc.shape

            # apply classifier
            from sklearn.naive_bayes import GaussianNB
            clf = GaussianNB()
            clf.fit(data,labels)
            #predicted_label = clf.predict(fc)
            predicted_label = knn(data,labels,fc)

            # convert this label into int and get the corresponding name
            person = names[int(predicted_label)]

            # declare type of font to be used on output window
            font = FONT_HERSHEY_DUPLEX = 2
            # display the name
            cv2.putText(frame, person, (x, y), font, 2,(255,255,0), 3)

            #draw rectangle around the face
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 3)

        # display frame
        cv2.imshow("Your faces", frame)

        if cv2.waitKey(1) == 27 :
            break

    else :
        print "Error"


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
