import cv2
from numpy import transpose, expand_dims, float32

def format_img_size(img, C):
	""" formats the image size based on config """
	img_min_side = float(C.im_size)
	(height,width,_) = img.shape
		
	if width <= height:
		ratio = img_min_side/width
		new_height = int(ratio * height)
		new_width = int(img_min_side)
	else:
		ratio = img_min_side/height
		new_width = int(ratio * width)
		new_height = int(img_min_side)
	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	return img, ratio	

def format_img_channels(img, C):
	""" formats the image channels based on config """
	img = img[:, :, (2, 1, 0)] # RGB -> BGR
	img = img.astype(float32)
	img[:, :, 0] -= C.img_channel_mean[0] # normalize
	img[:, :, 1] -= C.img_channel_mean[1]
	img[:, :, 2] -= C.img_channel_mean[2]
	img /= C.img_scaling_factor
	img = transpose(img, (2, 0, 1))
	img = expand_dims(img, axis=0)
	return img

def format_img(img, C, resize=True):
    """ formats an image for model prediction based on config """
    if resize:
        img, ratio = format_img_size(img, C)
    else:
        ratio = 1
    img = format_img_channels(img, C)
    return img, ratio

# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):

	real_x1 = int(round(x1 // ratio))
	real_y1 = int(round(y1 // ratio))
	real_x2 = int(round(x2 // ratio))
	real_y2 = int(round(y2 // ratio))

	return (real_x1, real_y1, real_x2 ,real_y2)

def draw_bbox(img, x1:float, y1:float, x2:float, y2:float, text:str = None, color:tuple=(255, 255, 255)):
    cv2.rectangle(img,(x1, y1), (x2, y2), color,2)
    if text != None:
        (retval,baseLine) = cv2.getTextSize(text,cv2.FONT_HERSHEY_COMPLEX,1,1)
        textOrg = (x1, y1-0)
        cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
        cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
        cv2.putText(img, text, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
    return img