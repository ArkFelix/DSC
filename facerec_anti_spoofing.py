import random 
import cv2
import imutils
import f_liveness_detection
import questions
import numpy as np
from keras.preprocessing import image

# Load the pre-trained face recognition model from Teachable Machine
from keras.models import load_model
model_face_recognition = load_model('keras_model.h5')

# Labels for face recognition classes
face_labels = ['person1', 'person2', 'person3']  # Update with your own labels

# instanciar camara
cv2.namedWindow('liveness_detection')
cam = cv2.VideoCapture(0)

# parameters 
COUNTER, TOTAL = 0,0
counter_ok_questions = 0
counter_ok_consecutives = 0
limit_consecutives = 3
limit_questions = 6
counter_try = 0
limit_try = 50 

# Function to perform face recognition
def recognize_face(img):
    img = cv2.resize(img, (224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize
    prediction = model_face_recognition.predict(img)[0]
    label_index = np.argmax(prediction)
    confidence = prediction[label_index]
    label = face_labels[label_index]
    return label, confidence

def show_image(cam, text, color=(0,0,255)):
    ret, im = cam.read()
    im = imutils.resize(im, width=720)
    cv2.putText(im, text, (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
    return im

for i_questions in range(0, limit_questions):
    # Generate a random question
    index_question = random.randint(0, 5)
    question = questions.question_bank(index_question)
    
    im = show_image(cam, question)
    cv2.imshow('liveness_detection', im)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

    # Perform face recognition
    ret, frame = cam.read()
    recognized_person, confidence = recognize_face(frame)

    # Display the recognized person's name
    name_text = f"Recognized: {recognized_person} (Confidence: {confidence:.2f})"
    im_with_name = show_image(cam, name_text)
    cv2.imshow('liveness_detection', im_with_name)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

    # Proceed with anti-spoofing tests
    for i_try in range(limit_try):
        ret, im = cam.read()
        im = imutils.resize(im, width=720)
        im = cv2.flip(im, 1)

        TOTAL_0 = TOTAL
        out_model = f_liveness_detection.detect_liveness(im, COUNTER, TOTAL_0)
        TOTAL = out_model['total_blinks']
        COUNTER = out_model['count_blinks_consecutives']
        dif_blink = TOTAL - TOTAL_0
        if dif_blink > 0:
            blinks_up = 1
        else:
            blinks_up = 0

        challenge_res = questions.challenge_result(question, out_model, blinks_up)

        im = show_image(cam, question)
        cv2.imshow('liveness_detection', im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break 

        if challenge_res == "pass":
            im = show_image(cam, question + " : ok")
            cv2.imshow('liveness_detection', im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            counter_ok_consecutives += 1
            if counter_ok_consecutives == limit_consecutives:
                counter_ok_questions += 1
                counter_try = 0
                counter_ok_consecutives = 0
                break
            else:
                continue

        elif challenge_res == "fail":
            counter_try += 1
            show_image(cam, question + " : fail")
        elif i_try == limit_try - 1:
            break

    if counter_ok_questions ==  limit_questions:
        while True:
            im = show_image(cam, "LIFENESS SUCCESSFUL", color=(0, 255, 0))
            cv2.imshow('liveness_detection', im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    elif i_try == limit_try - 1:
        while True:
            im = show_image(cam, "LIFENESS FAIL")
            cv2.imshow('liveness_detection', im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        break 
    else:
        continue