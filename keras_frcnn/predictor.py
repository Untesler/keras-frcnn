import os, sys, pickle
import numpy as np
from keras_frcnn import roi_helpers
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from utils import format_img

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
	try:
		# Currently, memory growth needs to be the same across GPUs
		for gpu in gpus:
			tf.config.experimental.set_memory_growth(gpu, True)
		logical_gpus = tf.config.experimental.list_logical_devices('GPU')
		print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
	except RuntimeError as e:
		# Memory growth must be set before GPUs have been initialized
		print(e)
sys.setrecursionlimit(40000)

class Predictor():
    def __init__(self, config_path:str, network_name:str = None, bbox_threshold:float = .8, resize_input:bool = True):
        self.bbox_thresh = bbox_threshold
        self.resize_input = resize_input
        with open(config_path, 'rb') as f_in:
            self.C = pickle.load(f_in)
        self.C.use_horizontal_flips = False
        self.C.use_vertical_flips = False
        self.C.rot_90 = False
        network_name = network_name if network_name != None else (self.C.network)
        if network_name == 'vgg':
            import keras_frcnn.vgg as nn
            feature_map_input = Input(shape=(None, None, 512))
        elif network_name == 'resnet50':
            import keras_frcnn.resnet as nn
            feature_map_input = Input(shape=(None, None, 1024))
        else:
            raise ValueError('Invalid Network Name')
        self.class_mapping = self.C.class_mapping
        if 'bg' not in self.class_mapping:
            self.class_mapping['bg'] = len(self.class_mapping)
        self.class_mapping = {v: k for k, v in self.class_mapping.items()}
        print(self.class_mapping)
        self.class_to_color = {self.class_mapping[v]: np.random.randint(0, 255, 3) for v in self.class_mapping} # random bounding box colors
        img_input = Input(shape=(None, None, 3))
        roi_input = Input(shape=(self.C.num_rois, 4))
        num_anchors = len(self.C.anchor_box_scales) * len(self.C.anchor_box_ratios)
        shared_layers = nn.nn_base(img_input, trainable=False)
        rpn_layers = nn.rpn(shared_layers, num_anchors)
        classifier = nn.classifier(feature_map_input, roi_input, self.C.num_rois, nb_classes=len(self.class_mapping), trainable=False)
        self.model_rpn = Model(img_input, rpn_layers)
        self.model_classifier = Model([feature_map_input, roi_input], classifier)
        print(f'Loading weights from {self.C.model_path}')
        self.model_rpn.load_weights(self.C.model_path, by_name=True)
        self.model_classifier.load_weights(self.C.model_path, by_name=True)
        self.model_rpn.compile(optimizer='sgd', loss='mse')
        self.model_classifier.compile(optimizer='sgd', loss='mse')
        print('Model complied')
    
    def get_class_color(self):
        return self.class_to_color

    def predict(self, img):
        X, ratio = format_img(img, self.C, self.resize_input)
        X = np.transpose(X, (0, 2, 3, 1))
        # get the feature maps and output from the RPN
        [Y1, Y2, F] = self.model_rpn.predict(X)
        R = roi_helpers.rpn_to_roi(Y1, Y2, self.C, 'tf', overlap_thresh=0.7)
        # convert from (x1,y1,x2,y2) to (x,y,w,h)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]
        # apply the spatial pyramid pooling to the proposed regions
        bboxes = {}
        probs = {}
        for jk in range(R.shape[0]//self.C.num_rois + 1):
            ROIs = np.expand_dims(R[self.C.num_rois*jk:self.C.num_rois*(jk+1), :], axis=0)
            if ROIs.shape[1] == 0:
                break
            if jk == R.shape[0]//self.C.num_rois:
                #pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0],self.C.num_rois,curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded
            [P_cls, P_regr] = self.model_classifier.predict([F, ROIs])
            for ii in range(P_cls.shape[1]):
                # suppression
                if np.max(P_cls[0, ii, :]) < self.bbox_thresh or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                    continue
                cls_name = self.class_mapping[np.argmax(P_cls[0, ii, :])]
                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []
                (x, y, w, h) = ROIs[0, ii, :]
                cls_num = np.argmax(P_cls[0, ii, :])
                try:
                    (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                    tx /= C.classifier_regr_std[0]
                    ty /= C.classifier_regr_std[1]
                    tw /= C.classifier_regr_std[2]
                    th /= C.classifier_regr_std[3]
                    x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                except:
                    pass
                bboxes[cls_name].append([self.C.rpn_stride*x, self.C.rpn_stride*y, self.C.rpn_stride*(x+w), self.C.rpn_stride*(y+h)])
                probs[cls_name].append(np.max(P_cls[0, ii, :]))
        return bboxes, probs, ratio