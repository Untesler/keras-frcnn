# Preparing a Dataset

Basically, You can use any dataset of image that you have had its label
and all you need todo with that dataset is to prepare a text file that is contains a 
path to image, coordinates of ground truth and its label in form like

`filepath,x1,y1,x2,y2,class_name`

For example:

    dataset/JPEGImages/00000.jpg,9,111,61,182,white

But currently I'm working around with safety-helmet detection problem, Therefore the data preparing process
in `DataPreprocessing.ipynb` is base on [safety-helmet detection dataset](https://pythonawesome.com/helmet-detection-on-construction-sites/)

So, if you just want to try you can use the same dataset as me for make it easy