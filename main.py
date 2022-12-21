import cv2 as cv
import os 
import numpy as np
import math
from matplotlib import pyplot as plt

def get_path_list(root_path):

    train_images = os.listdir (root_path) 
    train_names = list()

    for folder_name in train_images:
        train_names.append(folder_name)
    
    return train_names

def get_class_id(root_path, train_names):
    image_train_list = []
    image_class_id = []

    for index, image in enumerate(train_names):
        full_name_path = root_path + '/' + image

        for image_path in os.listdir(full_name_path):
            full_image_path = full_name_path + '/' + image_path
            img_gray = cv.imread(full_image_path, 0)

            image_train_list.append (img_gray)
            image_class_id.append (index)
            
    return image_train_list, image_class_id

def detect_faces_and_filter(image_list, image_classes_list=None):

    face_cascade = cv.CascadeClassifier ('haarcascades/haarcascade_frontalface_default.xml')

    face_list = []
    class_list = []
    face_rects = []
    
    if image_classes_list is not None: #train
        for image, image_class in zip(image_list, image_classes_list):

            detected_faces = face_cascade.detectMultiScale(
                image,
                scaleFactor = 1.3,
                minNeighbors = 3
            )

            if(len(detected_faces) < 1):
                continue

            for face_rect in detected_faces:
                x, y, w, h = face_rect
                face_img = image[y:y+w, x:x+h]
                
                face_list.append(face_img)
                class_list.append(image_class)

        return face_list, face_rects, class_list
    else: # test
        for image in image_list:

            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

            detected_faces = face_cascade.detectMultiScale(
                image,
                scaleFactor = 1.3,
                minNeighbors = 3
            )

            if(len(detected_faces) < 1):
                continue

            for face_rect in detected_faces:
                x, y, w, h = face_rect
                face_img = image[y:y+w, x:x+h]
                
                face_list.append(face_img)
                face_rects.append([x,y,w,h])

        return face_list, face_rects, class_list

def train(train_face_grays, image_classes_list):

    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    face_recognizer.train(
        train_face_grays,
        np.array(image_classes_list)
    )
    return face_recognizer

def get_test_images_data(test_root_path):
    images_test = os.listdir (test_root_path)
    image_test_list = list()
    for image in (images_test):
        test_path = f'{test_root_path}/{image}'
        image = cv.imread(test_path)
        image_test_list.append(image)

    return image_test_list
    
def predict(recognizer, test_faces_gray):
    results = list()

    for test_face in test_faces_gray:    
        res, _ = recognizer.predict(test_face)
        results.append(res)
        
    return results
    
def draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names):

    image_predict_results = list()

    for predict_result, test_image, face_rect in zip(predict_results, test_image_list, test_faces_rects):
        x, y, w, h = face_rect
        face_img = test_image [y:y+w, x:x+h]
        
        test_img = cv.rectangle(
            test_image,
            (x,y), 
            (x+w, y+h), 
            (0, 255, 0), 
            1
        )
        
        if train_names[predict_result] in ['Pewdiepie', 'Jacksepticeye']:
            text = f'{train_names[predict_result]} - Youtube'
        else:
            text = f'{train_names[predict_result]} - Twitch'

        test_image = cv.putText (test_img, text, (x, y-10), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        image_predict_results.append(test_image)

    return image_predict_results

def combine_and_show_result(image_list):
    plt.figure(figsize=(10,7))
    for index, image in enumerate(image_list):
        plt.subplot(2, 3, index+1)
        plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    train_root_path = "dataset/train"

    train_names = get_path_list(train_root_path) #labels_list
    train_image_list, image_classes_list = get_class_id(train_root_path, train_names) #faces, indexes
    train_face_grays, _, filtered_classes_list = detect_faces_and_filter(train_image_list, image_classes_list)
    recognizer = train(train_face_grays, filtered_classes_list)

    test_root_path = "dataset/test"

    test_image_list = get_test_images_data(test_root_path)
    test_faces_gray, test_faces_rects, _ = detect_faces_and_filter(test_image_list)
    predict_results = predict(recognizer, test_faces_gray)
    predicted_test_image_list = draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names)
    
    combine_and_show_result(predicted_test_image_list)