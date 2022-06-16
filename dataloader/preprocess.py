# import cv2
# import dlib
# import numpy as np
# import os
# import random
#
# IMG_EXTENSIONS = [
#     '.jpg', '.JPG', '.jpeg', '.JPEG',
#     '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
# ]
#
# IMG_JPGS = ['.jpg', '.jpeg', '.JPG', '.JPEG']
# IMG_PNGS = ['.png', '.PNG']
#
# NUMPY_EXTENSIONS = ['.npy', '.NPY']
#
# data_dir = '/home/njuciairs/zmy/data'
#
# def is_image_file(filename):
#     return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
#
# # 根据几个关键点来裁剪，把人脸裁剪的更小
# def crop(path):
#     detector = dlib.get_frontal_face_detector()
#     predictor = dlib.shape_predictor(data_dir + '/shape_predictor_68_face_landmarks.dat')
#     img = cv2.imread(path, 1)
#     #print(type(img))
#     #print(len(detector(img, 1)))
#     det = detector(img, 1)
#     if len(det) == 0:
#         print(path)
#         return
#
#     landmarks = np.matrix([[p.x, p.y] for p in predictor(img,det[0]).parts()])
#     y_down = landmarks[57][0, 1] + 8
#     y_top = landmarks[19][0, 1] - 20
#     x_left = landmarks[0][0, 0]
#     x_right = landmarks[16][0, 0]
#     cropped = img[y_top:y_down+1,x_left:x_right + 1]
#     cv2.imwrite(path, cropped)
#
# # 用默认的裁剪方式
# # import cv_utils
# def crop_face(path):
#     detector = dlib.get_frontal_face_detector()
#     predictor = dlib.shape_predictor(data_dir + '/shape_predictor_68_face_landmarks.dat')
#     img = cv2.imread(path, 1)
#     # print(len(detector(img, 1)))
#     det = detector(img, 1)
#     if len(det) == 0:
#         print(path)
#         return
#     faces = dlib.full_object_detections()
#     faces.append(predictor(img, det[0]))
#     crop = dlib.get_face_chip(img, faces[0])
#     #print(type(crop))
#     #crop = cv_utils.resize(crop, 64)
#     cv2.imwrite(path, crop)
#
#
# mug_dir = data_dir + '/MUG/subjects'
# def crop_mug():
#     for root, dirs, files in os.walk(mug_dir):
#         for f in files:
#             if is_image_file(f):
#                 print(os.path.join(root, f))
#                 crop_face(os.path.join(root, f))
#
#
# def random_choose(num):
#     for root, dirs, files in os.walk(mug_dir):
#         _len = len(files)
#         files = sorted(files)
#         numbers = range(0, _len)
#         chosen = random.sample(numbers, num)
#
#
# def crop_oulu():
#     return
#
# crop('/home/njuciairs/zmy/data/000.jpeg')