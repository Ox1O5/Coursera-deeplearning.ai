import cv2 
import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt


with open('/home/long0420/文档/学习上的玩意/ai.challenger/ai_challenger_keypoint_train_20170902/keypoint_train_annotations_20170909.json', 'r') as f:
    data = json.load(f)
    

max_human = 0
max_shape = np.zeros([3,1])
for i in range(len(data)):
    max_human = max(max_human, len(data[i]['human_annotations']))
print(max_shape)

pic_dic = '/home/long0420/文档/学习上的玩意/ai.challenger/ai_challenger_keypoint_train_20170902/keypoint_train_images_20170902/'
pic_name = pic_dic + str(data[0]['image_id'])+ '.jpg'

temp = cv2.imread (pic_name)
cv2.imshow('ss',temp)
flag = temp.shape
plt.imshow(temp)
flag2 = cv2.resize(temp,(800*800*3,1))
print(flag2)


i = 0
max_length = 0
max_width = 0
for dat in data:
    i = i + 1
    print(i)
    t = cv2.imread(pic_dic + str(dat['image_id']) + '.jpg')
    max_length = max(max_length, t.shape[0])
    max_width = max(max_width, t.shape[1])