from keras.models import load_model
from tensorflow.keras.applications.vgg19 import preprocess_input
import cv2
import numpy as np


def emotion_class(face_classifier, classifier, emotion_labels):
    cap = cv2.VideoCapture(0) # запуск камеры

    while True:
        _, frame = cap.read() # чтение: возвращается флаг, считался ли кадр, и сам кадр 
        labels = []
        box = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # модель классификации обучалась на цветных изображениях, так что переводим картинку из BGR в RGB
        faces = face_classifier.detectMultiScale(box) # прогоняем детектор через изображение, полученное с камеры

        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255), 2) # к изображению добавляется прямоугольник туда, куда мы укажем
            roi_box = box[y:y+h, x:x+w]
            roi_box = cv2.resize(roi_box, (128,128), interpolation=cv2.INTER_AREA)

            if np.sum([roi_box])!=0: # если детектор нашёл лица
                roi = np.empty((1, 128, 128, 3))
                roi[0] = roi_box 
                roi = preprocess_input(roi).astype('float32') 
                prediction = classifier(roi) # получаем предсказание от нашеё модели
                label = emotion_labels[np.argmax(prediction)] # определяем эмоцию из словаря 
                label_position = (x,y)
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2) # добавляем название эмоции справа сверху detection box

            else: # если лица не обнаружены
                cv2.putText(frame, 'No Faces', (30,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow('Emotion Detector', frame) # вывод кадра

        if cv2.waitKey(1) & 0xFF == ord('q'): # корректная остановка окна с камерой произойдет, когда мы нажмем q на клавиатуре 
            break

    cap.release()
    cv2.destroyAllWindows()


face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml') # классический (не нейросетевой) детектор лица OpenCV
classifier = load_model(r'saved_model_trt') # обученная модель для классификации эмоций

emotion_labels = {8: "uncertain", 7: "surprise", 6: "sad", 5: "neutral", 4: "happy", 3: "fear", 2: "disgust", 1: "contempt", 0: "anger"} 


if __name__ == "__main__":
    emotion_class(face_classifier, classifier, emotion_labels)
