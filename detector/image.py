import cv2
import dlib
import numpy as np
from keras.models import load_model
from scipy.spatial import distance as dist
from imutils import face_utils
from playsound import playsound
from gtts import gTTS
import os

num = 1


def assistant_speaks(output):
    global num
    num += 1
    print("PerSon : ", output)
    toSpeak = gTTS(text=output, lang='en-US', slow=False)
    file = str(num) + ".mp3"
    toSpeak.save(file)
    playsound(file, True)
    os.remove(file)


predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')


# detect the face rectangle
def detect(img, cascade=face_cascade, minimumFeatureSize=(20, 20)):
    if cascade.empty():
        raise (Exception("There was a problem loading your Haar Cascade xml file."))
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=1, minSize=minimumFeatureSize)

    # if it doesn't return rectangle return array
    # with zero lenght
    if len(rects) == 0:
        return []

    #  convert last coord from (width,height) to (maxX, maxY)
    rects[:, 2:] += rects[:, :2]

    return rects


def cropEyes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect the face at grayscale image
    te = detect(gray, minimumFeatureSize=(80, 80))

    # if the face detector doesn't detect face
    # return None, else if detects more than one faces
    # keep the bigger and if it is only one keep one dim
    if len(te) == 0:
        return None
    elif len(te) > 1:
        face = te[0]
    elif len(te) == 1:
        [face] = te

    # keep the face region from the whole frame
    face_rect = dlib.rectangle(left=int(face[0]), top=int(face[1]),
                               right=int(face[2]), bottom=int(face[3]))

    # determine the facial landmarks for the face region
    shape = predictor(gray, face_rect)
    shape = face_utils.shape_to_np(shape)

    #  grab the indexes of the facial landmarks for the left and
    #  right eye, respectively
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # extract the left and right eye coordinates
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    # keep the upper and the lower limit of the eye
    # and compute the height
    l_uppery = min(leftEye[1:3, 1])
    l_lowy = max(leftEye[4:, 1])
    l_dify = abs(l_uppery - l_lowy)

    # compute the width of the eye
    lw = (leftEye[3][0] - leftEye[0][0])

    # we want the image for the cnn to be (26,34)
    # so we add the half of the difference at x and y
    # axis from the width at height respectively left-right
    # and up-down
    minxl = (leftEye[0][0] - ((34 - lw) / 2))
    maxxl = (leftEye[3][0] + ((34 - lw) / 2))
    minyl = (l_uppery - ((26 - l_dify) / 2))
    maxyl = (l_lowy + ((26 - l_dify) / 2))

    # crop the eye rectangle from the frame
    left_eye_rect = np.rint([minxl, minyl, maxxl, maxyl])
    left_eye_rect = left_eye_rect.astype(int)
    left_eye_image = gray[(left_eye_rect[1]):left_eye_rect[3], (left_eye_rect[0]):left_eye_rect[2]]

    # same as left eye at right eye
    r_uppery = min(rightEye[1:3, 1])
    r_lowy = max(rightEye[4:, 1])
    r_dify = abs(r_uppery - r_lowy)
    rw = (rightEye[3][0] - rightEye[0][0])
    minxr = (rightEye[0][0] - ((34 - rw) / 2))
    maxxr = (rightEye[3][0] + ((34 - rw) / 2))
    minyr = (r_uppery - ((26 - r_dify) / 2))
    maxyr = (r_lowy + ((26 - r_dify) / 2))
    right_eye_rect = np.rint([minxr, minyr, maxxr, maxyr])
    right_eye_rect = right_eye_rect.astype(int)
    right_eye_image = gray[right_eye_rect[1]:right_eye_rect[3], right_eye_rect[0]:right_eye_rect[2]]

    # if it doesn't detect left or right eye return None
    if 0 in left_eye_image.shape or 0 in right_eye_image.shape:
        return None
    # resize for the conv net
    left_eye_image = cv2.resize(left_eye_image, (34, 26))
    right_eye_image = cv2.resize(right_eye_image, (34, 26))
    right_eye_image = cv2.flip(right_eye_image, 1)
    # return left and right eye
    return left_eye_image, right_eye_image


# make the image to have the same format as at training
def cnnPreprocess(img):
    img = img.astype('float32')
    img /= 255
    img = np.expand_dims(img, axis=2)
    img = np.expand_dims(img, axis=0)
    return img


def mouth_aspect_ratio(mouth):
    # compute the euclidean distances between the two sets of
    # vertical mouth landmarks (x, y)-coordinates
    A = dist.euclidean(mouth[2], mouth[10])  # 51, 59
    B = dist.euclidean(mouth[4], mouth[8])  # 53, 57

    # compute the euclidean distance between the horizontal
    # mouth landmark (x, y)-coordinates
    C = dist.euclidean(mouth[0], mouth[6])  # 49, 55

    # compute the mouth aspect ratio
    mar = (A + B) / (2.0 * C)

    # return the mouth aspect ratio
    return mar


detector = dlib.get_frontal_face_detector()


def mouth(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect the face at grayscale image
    te = detector(gray, 0)
    (mStart, mEnd) = (49, 68)
    # if the face detector doesn't detect face
    # return None, else if detects more than one faces
    # keep the bigger and if it is only one keep one dim
    if len(te) == 0:
        return None
    elif len(te) > 1:
        face = te[0]
    elif len(te) == 1:
        [face] = te

    for rect in te:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extract the mouth coordinates, then use the
        # coordinates to compute the mouth aspect ratio
        mouth = shape[mStart:mEnd]

        mouthMAR = mouth_aspect_ratio(mouth)
    mouthMAR = mouth_aspect_ratio(mouth)

    mar1 = mouthMAR
    return mar1


def main():
    # open the camera,load the cnn model
    camera = cv2.VideoCapture(0)
    frame_width = int(camera.get(3))
    frame_height = int(camera.get(4))
    out = cv2.VideoWriter('test.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
    model = load_model('blinkModel.hdf5')

    # blinks is the number of total blinks ,close_counter
    # the counter for consecutive close predictions
    # and mem_counter the counter of the previous loop
    close_counter = blinks = mem_counter = number = x =0
    MOUTH_AR_THRESH = 0.75
    yawn = yawns = 0
    yawn_status = False
    state = ''
    state1= ''
    state2= ''
    while True:

        ret, frame = camera.read()


        # img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        # img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        # frame = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        mar1 = mouth(frame)

        if mar1 is None:
            continue
        else:
            mar1 = mouth(frame)
        # detect eyes
        eyes = cropEyes(frame)
        if eyes is None:
            continue
        else:
            left_eye, right_eye = eyes

        # average the predictions of the two eyes
        prediction = (model.predict(cnnPreprocess(left_eye)) + model.predict(cnnPreprocess(right_eye))) / 2.0

        # blinks
        # if the eyes are open reset the counter for close eyes
        if prediction > 0.5:
            state = 'open'
            close_counter = 0
            number += 1
        else:
            state = 'close'
            close_counter += 1
            number += 1
        state2 = 'No drowsiness'
        # if the eyes are open and previousle were closed
        # for sufficient number of frames then increcement
        # the total blinks
        if number % 665 == 0:
            x = x + 1
            if blinks <= 10 or yawns > 3:

                #state2 = 'less drowsiness'
                assistant_speaks('''i think you feel sleepy, you may need to stop your car. Would you like me to search for a coffee shop nearby?
		       ''')

                cv2.putText(frame, "Would you like to get a cup of coffee?", (5, 220),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.03, (0, 0, 255), 2)
                blinks = 0
                yawns = 0
                state2 = 'less drowsiness'
                continue

            # blinks = 0
            # yawns = 0
        else:
            x = x
        if x * 665 < number < (x + 1) * 665:

            if state == 'open' and mem_counter > 1:

                blinks += 1
            if mar1 > MOUTH_AR_THRESH:
                state1 = 'open'
                yawn += 1

            else:
                state1 = 'close'
                yawn =0
            print(yawn)
            if state1 == 'open' and yawn % 18==0 and yawn !=0 :
                yawns += 1

            if close_counter > 45:
                state2 = 'drowsiness'
                playsound('C:/Users/tranc/Downloads/sirena_ambulanza.WAV')
                cv2.putText(frame, "Wake up", (200, 220),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

            # keep the counter for the next loop
            mem_counter = close_counter

            # draw the total number of blinks on the frame along with
            # the state for the frame
            cv2.putText(frame, "Blinks: {}".format(blinks), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Eye State: {}".format(state), (430, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Yawns: {}".format(yawns), (10, 90),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 127), 2)
            cv2.putText(frame, "Mouth State: {}".format(state1), (410, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 127), 2)
            cv2.putText(frame, "LoD: {}".format(state2), (180, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 127), 2)
            cv2.imshow('drowsiness detection', frame)

            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord('q'):
                break
    # do a little clean up

    cv2.destroyAllWindows()
    del (camera)


if __name__ == '__main__':
    main()