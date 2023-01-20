import cv2
import mediapipe as mp
import face_recognition
import imutils
import concurrent.futures
import numpy as np


# Détection des mains via la librairie Mediapipe.
def detect_hands(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_draw = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=4)
    results = hands.process(rgb)
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
    return frame


# Détection des visages par les cascades d'Haar.
def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return frame


# Détermine si l'un des visages détectés est l'un de ceux enregistré dans MTI805_1/people
# via la librairie face_recognition.
def recognize_face(frame):
    small_frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)
    rgb_small_frame = small_frame[:, :, ::-1]

    known_image_1 = face_recognition.load_image_file("./people/enzo_dimaria.jpg")
    known_encoding_1 = face_recognition.face_encodings(known_image_1)[0]

    known_image_2 = face_recognition.load_image_file("./people/luc_duong.png")
    known_encoding_2 = face_recognition.face_encodings(known_image_2)[0]

    known_encodings = [known_encoding_1, known_encoding_2]
    known_names = ["Enzo DM", "Luc Duong"]

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    face_names = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Autre"

        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_names[best_match_index]

        face_names.append(name)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 5
        right *= 5
        bottom *= 5
        left *= 5

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    return frame


def run_webcam(face_reco, face_detect):
    cap = cv2.VideoCapture(0)
    key = cv2.waitKey(1)

    while key == -1:
        key = cv2.waitKey(1)

        success, frame = cap.read()
        if not success:
            raise Exception("[Erreur matérielle] Brancher votre caméra")
        frame = imutils.resize(frame, width=800)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            hand_future = executor.submit(detect_hands, frame)
            if face_detect:
                faces_future = executor.submit(detect_faces, frame)
            if face_reco:
                recog_future = executor.submit(recognize_face, frame)

            hand_future.result()
            if face_detect:
                faces_future.result()
            if face_reco:
                recog_future.result()

        cv2.imshow('FaceHandDetect by Enzo DI MARIA', frame)

    cap.release()
    cv2.destroyAllWindows()
    return
