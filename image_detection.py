import logging, os, cv2, sys, time
import numpy as np
from os.path import join, abspath
from keras_frcnn import roi_helpers
from keras_frcnn.predictor import Predictor
from utils import get_real_coordinates, draw_bbox
from optparse import OptionParser

sys.setrecursionlimit(40000)

parser = OptionParser()
parser.add_option('-p', '--path', dest='image_path', help="Path to testing image")
parser.add_option('-c', '--config', dest='config', help="Path to model configuration file", default='config.pickle')
parser.add_option('--network', dest='network', help="Base network to use", default='vgg')

(options, args) = parser.parse_args()

img_path = join(abspath('.'), abspath(options.image_path))
config_path = join(abspath('.'), abspath(options.config))
model = Predictor(config_path, options.network)
class_to_color = model.get_class_color()

img = cv2.imread(img_path)
all_dets = []
st = time.time()
bboxes, probs, ratio = model.predict(img)
for key in bboxes:
    bbox = np.array(bboxes[key])

    new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
    for jk in range(new_boxes.shape[0]):
        (x1, y1, x2, y2) = new_boxes[jk,:]

        (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)
        (x1, y1, x2, y2) = new_boxes[jk,:]
        (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)
        textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
        color = (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2]))
        img = draw_bbox(img, real_x1, real_y1, real_x2, real_y2, textLabel, color)
        all_dets.append((key,100*new_probs[jk]))
print(f'Elapsed time = {time.time() - st}')
print(all_dets)
cv2.imshow(img_path, img)
cv2.waitKey(0)