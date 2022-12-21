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

    '''
        To get a list of path directories from root path

        Parameters
        ----------
        root_path : str
            Location of root directory
        
        Returns
        -------
        list
            List containing the names of the sub-directories in the
            root directory
    '''

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
    # print(zip(image_train_list,image_class_id))
    # for a,b in zip(image_train_list,image_class_id):
    #     print(f'{a} - {b}')
    return image_train_list, image_class_id
            
    '''
        To get a list of train images and a list of image classes id

        Parameters
        ----------
        root_path : str
            Location of images root directory
        train_names : list
            List containing the names of the train sub-directories
        Returns
        -------
        list
            List containing all image in the train directories
        list
            List containing all image classes id
    '''

def detect_faces_and_filter(image_list, image_classes_list=None):

    face_cascade = cv.CascadeClassifier ('haarcascades/haarcascade_frontalface_default.xml')

    face_list = []
    class_list = []

    for image, image_class in zip(image_list, image_classes_list):

        detected_faces = face_cascade.detectMultiScale(
            image,
            scaleFactor = 1.2,
            minNeighbors = 5
        )

        if(len(detected_faces) < 1):
            continue

        for face_rect in detected_faces:
            x, y, w, h = face_rect
            face_img = image[y:y+w, x:x+h]
            face_list.append(face_img)
            class_list.append(image_class)

    return face_list, face_img, class_list
    
    '''
        To detect a face from given image list and filter it if the face on
        the given image is less than one

        Parameters
        ----------
        image_list : list
            List containing all loaded images
        image_classes_list : list, optional
            List containing all image classes id
        
        Returns
        -------
        list
            List containing all filtered and cropped face images in grayscale
        list
            List containing all filtered faces location saved in rectangle (x, y, w, h)
        list
            List containing all filtered image classes id
    ''' 

def train(train_face_grays, image_classes_list):


    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    face_recognizer.train(
        train_face_grays,
        np.array(image_classes_list)
    )
    # print('yay')
    return face_recognizer
    '''
        To create and train face recognizer object

        Parameters
        ----------
        train_face_grays : list
            List containing all filtered and cropped face images in grayscale
        image_classes_list : list
            List containing all filtered image classes id
        
        Returns
        -------
        object
            Recognizer object after being trained with cropped face images
    '''

def get_test_images_data(test_root_path):
    images_test = os.listdir (test_root_path)
    image_test_list = list()
    for image in (images_test):
        test_path = f'{test_root_path}/{image}'
        image = cv.imread(test_path, 0)
        image_test_list.append(image)
        # print('halo')
    # print(image_test_list)
    return image_test_list
    '''
        To load a list of test images from given path list

        Parameters
        ----------
        test_root_path : str
            Location of images root directory
        
        Returns
        -------
        list
            List containing all loaded gray test images
    '''
    
def predict(recognizer, test_faces_gray):



    '''
        To predict the test image with the recognizer

        Parameters
        ----------
        recognizer : object
            Recognizer object after being trained with cropped face images
        train_face_grays : list
            List containing all filtered and cropped face images in grayscale

        Returns
        -------
        list
            List containing all prediction results from given test faces
    '''
    
def draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names):
    '''
        To draw prediction results on the given test images and acceptance status

        Parameters
        ----------
        predict_results : list
            List containing all prediction results from given test faces
        test_image_list : list
            List containing all loaded test images
        test_faces_rects : list
            List containing all filtered faces location saved in rectangle
        train_names : list
            List containing the names of the train sub-directories

        Returns
        -------
        list
            List containing all test images after being drawn with
            final result
    '''

def combine_and_show_result(image_list):
    '''
        To show the final image that already combine into one image

        Parameters
        ----------
        image_list : nparray
            Array containing image data
    '''

'''
You may modify the code below if it's marked between

-------------------
Modifiable
-------------------

and

-------------------
End of modifiable
-------------------
'''
if __name__ == "__main__":

    '''
        Please modify train_root_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    train_root_path = "dataset/train"
    '''
        -------------------
        End of modifiable
        -------------------
    '''

    train_names = get_path_list(train_root_path) #labels_list
    train_image_list, image_classes_list = get_class_id(train_root_path, train_names) #faces, indexes
    train_face_grays, _, filtered_classes_list = detect_faces_and_filter(train_image_list, image_classes_list)
    recognizer = train(train_face_grays, filtered_classes_list)

    '''
        Please modify train_root_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    test_root_path = "dataset/test"
    '''
        -------------------
        End of modifiable
        -------------------
    '''

    test_image_list = get_test_images_data(test_root_path)
    test_faces_gray, test_faces_rects, _ = detect_faces_and_filter(test_image_list)
    predict_results = predict(recognizer, test_faces_gray)
    # predicted_test_image_list = draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names)
    
    # combine_and_show_result(predicted_test_image_list)