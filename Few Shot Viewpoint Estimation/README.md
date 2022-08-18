# Few-Shot Viewpoint Estimation

![bLOCKdIAGRAM](https://user-images.githubusercontent.com/94464164/162575914-7f65b96c-a479-4194-b6ae-24eecf96bb39.jpg)



## Tabel of Contents
* [Introduction](https://github.com/YoungXIAO13/FewShotViewpoint#installation)
* [Data Preparation](https://github.com/YoungXIAO13/FewShotViewpoint#data-preparation)
* [Training](https://github.com/YoungXIAO13/FewShotViewpoint#getting-started)
* [Demo](https://github.com/YoungXIAO13/FewShotViewpoint#demo)


## Introduction

Code built on top of [PoseFromShape](https://github.com/YoungXIAO13/PoseFromShape).
and [Few-Shot Object Detection and Viewpoint Estimation for Objects in the Wild](http://imagine.enpc.fr/~xiaoy/FSDetView/)

**Build to Run the code**

Create conda env:
```sh
## Create conda env
conda create --name FSviewpoint --file spec-file.txt
conda activate FSviewpoint
pip install matplotlib
conda uninstall pytorch
pip install -u upgrade pip
pip install torchvision
```



## Data Preparation

### ObjectNet3D

We split the 100 object classes into 80 base classes and 20 novel classes.

Unzip ObjectNet3d.zip in data/ObjectNet3d folder it contains the annotation information.

Download [ObjectNet3D](http://ftp.cs.stanford.edu/cs/cvgl/ObjectNet3D/) and place the files inside data/ObjectNet3d/

Download [PointClouds](https://www.dropbox.com/s/0s61m3cvwir0tsc/ObjectNet3DPointclouds.zip?dl=0) and place inside data/ObjectNet3d/

Data structure after this should look like:
```
data/ObjectNet3D
    ObjectNet3d.txt
    Annotations/
    Images/
    ImageSets/
    CAD/
    Pointclouds/
    ...
```


## Training

### Base-Class Training

```bash
# Intra-Dataset
bash run/train_intra.sh
```


### Multiple Runs for few shot fine tuning and testing

Once the base-class training is done, you can run 10 times few-shot fine-tuning and testing with few-shot training data randomly selected for each run:
```bash
bash run/multiple_times_intra.sh
``` 

To get the performance averaged over multiple runs(In this case 10):
```bash
python mean_metrics.py save_models/IntraDataset_shot10

python mean_metrics.py save_models/InterDataset_shot10
``` 
![Result_chart](https://user-images.githubusercontent.com/24851079/162577912-0d4f3f39-6ba9-4400-a4a7-66b8303a2bea.png)


## Demo


To test the pre-trained model on a single object-centered image, place any **one** of the 10 provided model_weights and its mean_class_attention.pkl file provided at this drive **link** (https://drive.google.com/file/d/12NazlnPOaKVtN2aAg2MWUlsJTfAHWuSI/view?usp=sharing) in the main folder  run the following command: 
```
python demo.py \
--model {model_weight.pth} \
--class_data {mean_class_attention.pkl} \
--test_cls {test_class_name} \
--test_img {tets_image_path}
```

**Sample Run**: (replace --model path argument with your own .pth path and class_data argument with your own .pkl path.)
```
python demo.py --model save_models/IntraDataset/checkpoint.pth --class_data save_models/IntraDataset/mean_class_data.pkl --test_cls can --test_img testimg/can.JPEG
```
**Sample Output**:
```
Azimuth = 337.112        Elevation = -22.192     Inplane-Rotation = -97.720
```
The estimated viewpoint will be printed in format of Euler angles.


## Available classes:
aeroplane
ashtray
backpack
basket
bed
bench
bicycle
blackboard
boat
bookshelf
bottle
bucket
bus
cabinet
calculator
camera
can
cap
car
cellphone
chair
clock
coffee_maker
comb
computer
cup
desk_lamp
diningtable
dishwasher
door
eraser
eyeglasses
fan
faucet
filing_cabinet
fire_extinguisher
fish_tank
flashlight
fork
guitar
hair_dryer
hammer
headphone
helmet
iron
jar
kettle
key
keyboard
knife
laptop
lighter
mailbox
microphone
microwave
motorbike
mouse
paintbrush
pan
pen
pencil
piano
pillow
plate
pot
printer
racket
refrigerator
remote_control
rifle
road_pole
satellite_dish
scissors
screwdriver
shoe
shovel
sign
skate
skateboard
slipper
sofa
speaker
spoon
stapler
stove
suitcase
teapot
telephone
toaster
toilet
toothbrush
train
trash_bin
trophy
tub
tvmonitor
vending_machine
washing_machine
watch
wheelchair
