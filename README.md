# CVYOLOX   Implementation of the object detection model in Pytorch
---

## Directory
1. [Data and weight download](#Data and weight download)
2. [Environment](#Environment)
3. [How2train](#How2train)
4. [How2predict](#How2predict)
5. [How2eval](#How2eval)
6. [Reference](#Reference)

## Data and weight download
Make a complete CV dataset download address：https://github.com/Hickey-Curry/CVYOLOX/releases/download/data/VOCdevkit.zip
YOLOX_s.pth Download address： https://github.com/Hickey-Curry/CVYOLOX/releases/download/V1.0/yolox_s.pth

## Environment
scipy==1.2.1
numpy==1.17.0
matplotlib==3.1.2
opencv_python==4.1.2.30
torch==1.2.0
torchvision==0.4.0
tqdm==4.60.0
Pillow==8.2.0
h5py==2.10.0

## How2train
### Train your own dataset
1. Dataset preparation  
**Use VOC format for training, you need to make your own dataset before training**    
Before training, put the label file in Annotation under the VOC2007 folder under the VOCdevkit folder.   
Before training, place the image file in JPEGImages under the VOC2007 folder under the VOCdevkit folder.    

2. Processing of datasets   
After completing the placement of the dataset, we need to use the voc_annotation.py to obtain the 2007_train.txt and 2007_val.txt for training.   
Modify the parameters in the voc_annotation.py. The first training can only modify the classes_path, classes_path used to point to the TXT corresponding to the detection class.   
When training your own dataset, you can create your own cls_classes.txt with the categories you need to distinguish.  
model_data/cls_classes.txt：      
```python
CVs
```
Modify the classes_path in the voc_annotation.py so that it corresponds to the cls_classes.txt and run the voc_annotation.py.  

3. start network training  
**There are many parameters for training, all in the train.py, you can carefully read the annotations after downloading the library, the most important part of which is still the classes_path in the train.py.**  
**classes_path is used to point to the txt corresponding to the detection category, this txt is the same as the txt in voc_annotation.py! Training your own data set must be modified!**  
After modifying the classes_path, you can run train.py start training, and after training multiple epochs, the weights will be generated in the logs folder.    

4. Training result prediction    
Two files are required to predict the training results, namely the yolo.py and the predict.py. Modify the model_path and classes_path in the yolo.py.  
**model_path point to the trained weight file, in the logs folder.  
classes_path point to the TXT corresponding to the detection category.**  
Once you have finished modifying it, you can run predict.py detection. After running, enter the image path to detect.    


## How2predict 
### Use the weights you trained
1. Follow the training steps to train.
2. In the yolo.py file, modify model_path and classes_path in the following parts to correspond to the trained files; **model_path corresponds to the weight file under the logs folder, and classes_path is the class that model_path corresponds to**.
```python
_defaults = {
    "model_path"        : 'model_data/yolox_s.pth',
    "classes_path"      : 'model_data/coco_classes.txt',
    "input_shape"       : [640, 640],
    "phi"               : 's',
    "confidence"        : 0.5,
    "nms_iou"           : 0.3,
    "letterbox_image"   : True,
    "cuda"              : True,
}
```
3. Run predict.py，input 
```python
img/test.jpg
```
4. Set up in the predict.py for FPS testing.  

## How2eval 
1. Evaluation using VOC format.  
2. If you have run the voc_annotation.py file before training, the code automatically divides the dataset into training set, validation set, and test set.
3. After dividing the test set with the voc_annotation.py, go to the get_map.py file to modify the classes_path, classes_path to point to the TXT corresponding to the detection category, which is the same as the TXT during training. Evaluating your own data set must be modified.
4. Modify the model_path and classes_path in the yolo.py. **model_path Point to the trained weight file, in the logs folder. classes_path point to the TXT corresponding to the detection category. **  
5. Run get_map.py to get the assessment results, which will be saved in the map_out folder.

## Reference
https://github.com/Megvii-BaseDetection/YOLOX
