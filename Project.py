#Loading the required libraries
import sqlite3
import webbrowser
import cv2
import numpy as np
import dlib
from keras.models import load_model
from statistics import mode
from utils.datasets import get_labels
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.preprocessor import preprocess_input
import time


def cam():
    emotion = None
    while cap.isOpened():  # True:

        ret, bgr_image = cap.read()

        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        # cv2.imshow('window', gray_image)

        # Applying the face detection method on the grayscale image
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5,
                                              minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        for face_coordinates in faces:

            x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]
            # cv2.imshow('window', bgr_image)
            try:
                gray_face = cv2.resize(gray_face, (emotion_target_size))
            except:
                continue

            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            # cv2.imshow('window1', bgr_image)
            emotion_prediction = emotion_classifier.predict(gray_face)
            emotion_probability = np.max(emotion_prediction)
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = emotion_labels[emotion_label_arg]
            emotion_window.append(emotion_text)

            if len(emotion_window) > frame_window:
                emotion_window.pop(0)
            try:
                emotion_mode = mode(emotion_window)
            except:
                continue

            if emotion_text == 'angry':
                print(emotion_probability)

                if emotion_probability >= 0.48181498 and emotion_probability <= 0.49352397:
                    emotion="angry"

                    cap.release()
                    cv2.destroyAllWindows()

                color = emotion_probability * np.asarray((255, 0, 0))


            elif emotion_text == 'sad':
                print(emotion_probability)
                if emotion_probability >= 0.32582545 and emotion_probability <= 0.3314238:
                    emotion="sad"
                    cap.release()
                    cv2.destroyAllWindows()

                color = emotion_probability * np.asarray((0, 0, 255))


            elif emotion_text == 'happy':
                print(emotion_probability)
                if emotion_probability >= 0.9015124 and emotion_probability <= 0.9188275:
                    emotion="happy"
                    cap.release()
                    cv2.destroyAllWindows()

                color = emotion_probability * np.asarray((255, 255, 0))

            elif emotion_text == 'surprise':
                print(emotion_probability)

                if emotion_probability >= 0.7189152 and emotion_probability <= 0.73248416:
                    emotion="surprise"


                    cap.release()
                    cv2.destroyAllWindows()



                color = emotion_probability * np.asarray((0, 255, 255))



            else:
                cap.release()
                cv2.destroyAllWindows()
                color = emotion_probability * np.asarray((0, 255, 0))
                emotion="neutral"
                # start_time = time.time()
                # time.sleep(1.0 - time.time() + start_time)

            color = color.astype(int)
            color = color.tolist()

            draw_bounding_box(face_coordinates, rgb_image, color)
            draw_text(face_coordinates, rgb_image, emotion_mode,
                      color, 0, -45, 1, 1)

        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        cv2.imshow('window_frame', bgr_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    return emotion



def eye():
    cap = None
    cap = cv2.VideoCapture(0)
    time.sleep(5)
    eye_list=None

    numerator = 0
    denominator = 0
    while cap.isOpened():
        ret, frame = cap.read()
        # cv2.imshow("image",frame)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        roi = frame
        if ret:
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            # print(faces)
            for (x, y, w, h) in faces:
                print(x, y, w, h)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)

                d = 10920.0 / float(w)

                x1 = int(x + w / 4.2) + 1
                x2 = int(x + w / 2.5)
                y1 = int(y + h / 3) + 1
                y2 = int(y + h / 2.2)
                roi = frame[y1:y2, x1:x2]
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                equ = cv2.equalizeHist(gray)
                thres = cv2.inRange(equ, 0, 20)
                kernel = np.ones((3, 3), np.uint8)

                dilation = cv2.dilate(thres, kernel, iterations=2)

                erosion = cv2.erode(dilation, kernel, iterations=3)

                contours, hierarchy = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

                if len(contours) == 2:
                    numerator += 1
                    M = cv2.moments(contours[1])
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        cv2.line(roi, (cx, cy), (cx, cy), (0, 255, 255), 3)
                        cv2.line(roi, (cx, cy), (cx, cy), (0, 255, 255), 3)


                elif len(contours) == 1:
                    numerator += 1
                    M = cv2.moments(contours[0])
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        # print(cx, cy)
                        # print(cx)
                        cv2.line(roi, (cx, cy), (cx, cy), (0, 0, 255), 3)
                        cv2.line(roi, (cx, cy), (cx, cy), (0, 0, 255), 3)

                        if cx > 20:
                            eye_list="right"
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                            cv2.putText(frame, 'right', (500, 250),
                                        font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
                            cap.release()
                            cv2.destroyAllWindows()

                        if cx < 10:
                            eye_list="left"
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 1)
                            cv2.putText(frame, 'left', (100, 250),
                                        font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
                            cap.release()
                            cv2.destroyAllWindows()
                        # if cx == 10:
                        #     webbrowser.open('https://w3resource.com/python-exercises/')
                        #     camera.release()
                        #     print("accurracy=", (float(numerator) / float(numerator + denominator)) * 100)
                        #     cv2.destroyAllWindows()
                        #
                        #     break
                        if cy > 20:
                            eye_list="sleep"
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 1)
                            cv2.putText(frame, 'sleep', (210, 450),
                                        font, 0.8, (255, 255, 0), 1, cv2.LINE_AA)
                            cap.release()
                            cv2.destroyAllWindows()
                        if cy < 3:
                            eye_list="up"
                            # print("looking top")
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                            cv2.putText(frame, 'up', (300, 50),
                                        font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
                            cap.release()
                            cv2.destroyAllWindows()

                else:
                    denominator += 1

                # ran = x2 - x1
                # mid = ran / 2
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # print("accurracy=", (float(numerator) / float(numerator + denominator)) * 100)
    return eye_list

def head(draw=None):

    head_list=None

    # draw = draw.draw()


    facedetector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

    src = cv2.VideoCapture(0)
    time.sleep(5)

    # src.set(3, 640)  # 横サイズ
    # src.set(4, 480)  # 縦サイズ

    while src.isOpened():
        ret, frame = src.read()
        height, width = frame.shape[:2]
        draw_frame = frame.copy()


        faces, scores, types = facedetector.run(frame, 1)
        for i, face in enumerate(faces):
            top, bottom, left, right = face.top(), face.bottom(), face.left(), face.right()
            print(top)
            if min(top, height - bottom - 1, left, width - right - 1) < 0:
                continue
            cv2.rectangle(draw_frame, (left, top), (right, bottom), (0, 0, 255), 1)
            font = cv2.FONT_HERSHEY_SIMPLEX

            cv2.putText(draw_frame, "Scores : " + str(scores[i]), (10, i + 15), cv2.FONT_HERSHEY_PLAIN, 1.0,
                        (255, 255, 0))
            cv2.putText(draw_frame, "Orientation : " + str(types[i]), (10, i + 30), cv2.FONT_HERSHEY_PLAIN, 1.0,
                        (255, 255, 0))
            if types[i] == 0:
                head_list="front"
                cv2.putText(draw_frame, "Front", (300, 50), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
                src.release()
                cv2.destroyAllWindows()
                # print(head_list)
            elif types[i] == 1:
                head_list="Right"
                cv2.putText(draw_frame, "Right", (300, 50), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
                src.release()
                cv2.destroyAllWindows()
                # print(head_list)
            elif types[i] == 2:
                head_list="Left"
                cv2.putText(draw_frame, "Left", (300, 50), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
                src.release()
                cv2.destroyAllWindows()
                # print(head_list)


            """Facial Features"""
            # draw.drawFacePoint(draw_frame, predictor, face, line=True, point=True)

        cv2.imshow('Image', draw_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    return head_list



#To connect with the facce_eye_head database and initiliaze the cconnection object to CONN
conn = sqlite3.connect('face_eye_head.db')

with conn:
    # Creating a cursor object using the cursor() method
    cursor = conn.cursor()
cursor.execute('CREATE TABLE IF NOT EXISTS concentration (emotion VARCHAR(255),eye VARCHAR(255),head VARCHAR(255))')
# cursor.execute('CREATE TABLE IF NOT EXISTS concentration (emotion VARCHAR(255),eye VARCHAR(255))')


USE_WEBCAM = True  # If false, loads video file source

# parameters for loading data and images
emotion_model_path = './models/emotion_model.hdf5'
emotion_labels = get_labels('fer2013')  #returns a dictionary of emotion values
                        # {0:'angry',1:'disgust',2:'fear',3:'happy',4:'sad',5:'surprise',6:'neutral'}

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# loading models
face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
emotion_classifier = load_model(emotion_model_path)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]
print(emotion_target_size)

# starting lists for calculating modes
emotion_window = []

# starting video streaming
cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)

# Select video or webcam feed
cap = None
if (USE_WEBCAM == True):
    cap = cv2.VideoCapture(0)  # Webcam source
else:
    cap = cv2.VideoCapture('.\\demo\\dinner.mp4')  # Video file source                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            \

emotion =cam()
eye=eye()
head=head()

# cursor.execute('INSERT INTO concentration (emotion,eye) VALUES(?,?)', (str(emotion),str(eye)))
cursor.execute('INSERT INTO concentration (emotion,eye,head) VALUES(?,?,?)', (str(emotion),str(eye),str(head)))
cursor.execute('select * from concentration')

conn.commit()
if (emotion=='neutral' or emotion=='happy') and (eye=='left' or eye=='right')  and head=='front':
    print('concentrated')
else:
    print('distracted')
    webbrowser.open('https://www.learnopencv.com/install-opencv-3-and-dlib-on-windows-python-only/')