# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 15:44:57 2018

@author: lykha
"""

from keras.models import model_from_json
from Train_model import model

model = model_from_json(open('sports1M_model.json', 'r').read())
model.load_weights('sports1M_weights.h5')
model.compile(loss='mean_squared_error', optimizer='sgd')

with open('labels.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]
print('Total labels: {}'.format(len(labels)))

import cv2
import numpy as np

cap = cv2.VideoCapture('dM06AMFLsrc.mp4')

vid = []
while True:
    ret, img = cap.read()
    if not ret:
        break
    vid.append(cv2.resize(img, (171, 128)))
vid = np.array(vid, dtype=np.float32)

frames = vid[2000:2016, 8:120, 30:142,: ]
X = frames.transpose((3, 0, 1, 2))
output = model.predict_on_batch(np.array([X]))


print('Position of maximum probability: {}'.format(str(output[0].argmax())))
print('Maximum probability: {:.5f}'.format(str(max(output[0][0]))))
print('Corresponding label: {}'.format(str(labels[output[0].argmax()])))
