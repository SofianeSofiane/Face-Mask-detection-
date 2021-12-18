import tensorflow as tf
import numpy as np
import cv2
import face_recognition

model=tf.keras.models.load_model('mask_detection.model')

video=cv2.VideoCapture(0)

text_dict={0:'Mask ON',1:'No Mask'}
rect_color_dict={0:(0,255,0),1:(0,0,255)}

while True:
    ret , image=video.read()

    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces =face_recognition.face_locations(image,model='hog')

    type = isinstance(faces, tuple)

    if type == False:
        for face in faces:
            face_location = np.array(face)

            x = face_location[3]
            y = face_location[0]
            w = face_location[1] - face_location[3]
            h = face_location[2] - face_location[0]
            im = image[y:y + w, x:x + w]
            cv2.imshow('face',im)

            grayscale_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            resized_image = cv2.resize(grayscale_image, (200, 200))
            imags = []
            imags.append(resized_image)
            imags = np.array(imags) / 255.0
            imags = np.reshape(imags, (imags.shape[0], 200, 200, 1))
            predictions = model.predict(imags)
            predictions = np.argmax(predictions)
            label = predictions
            cv2.rectangle(image, (x, y), (x + w, y + h), rect_color_dict[label], 2)
            cv2.rectangle(image, (x, y - 40), (x + w, y), rect_color_dict[label], -1)
            cv2.putText(image, text_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    else:
        print('no faces detected')
    cv2.imshow('image',image)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break