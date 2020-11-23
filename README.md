# HW2-Object-detection
Student ID: 309553007

### Introduce
Object detection: 
use Street View House Numbers(SVHN) dataset which containing 33,402 trianing images, 13,068 test images to train a not only accurate but fast digit detector

### Hardware
* Python 3.8
* Pytorch 1.5.1
* torchvision 0.6.1

### Reproducing Submission
You can reproduce this work by downloading  the pretrained model or doing the ```Steps``` in Train.
1. <Dataset Preparation>
2. <Train>
3. <Inference>

### Dataset Preparation
1. Prepare xml files: A corresponding xml file should be generated for each training image.
The xml file format:
```
<annotation>
  <filename>1.png</filename>
  <size>
    <width>741</width>
    <height>350</height>
    <depth>3</depth>
  </size>
  <object>
    <name>1</name>
    <bndbox>
      <xmin>246</xmin>
      <ymin>77</ymin>
      <xmax>327</xmax>
      <ymax>296</ymax>
     </bndbox>
   </object>
   <obect>
    <name>9</name>
    <bndbox>
      <xmin>323</xmin>
      <ymin>81</ymin>
      <xmax>419</xmax>
      <ymax>300</ymax>
     </bndbox>
   </object>
</annotation>
```
2. Prepare train.txt and valid.txt: the txt file contains each image filename.
3. Dataset structure:
```
VOCdevkit/
  -VOC2007/
    -Annotations/
      -1.xml
      -2.xml
      ...
      -33402.xml
    -ImageSets/
      -Main/
        -train.txt
        -valid.txt
    -JEPGImages/
      -1.png
      -2.png
      ...
      -33402.png
    -test/
      -1.png
      -2.png
      ...
      -13068.png
```

### Training
Use pretrained ssd network to train our object detector
* Download Pretrained models:
Download pretrained models from model folder
* Steps:
  1. Setup environment and mmdetection
  ```
  pip install -U torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html

  # install mmcv-full thus we could use CUDA operators
  pip install mmcv-full

  # Install mmdetection
  rm -rf mmdetection
  git clone https://github.com/open-mmlab/mmdetection.git
  cd mmdetection

  pip install -e .
  ```
  2. Create a data folder in the mmdetection folder and put your VOCdevkit folder into this data folder.
  3. Modify classes
  ```
    a.)*mmdetection/mmdet/datasets/voc.py*
        修改 CLASSES = ('1','2','3','4','5','6','7','8','9','10')
    b.)*mmdetection/mmdet/core/evaluation/class_names.py*
        修改 CLASSES = ('1','2','3','4','5','6','7','8','9','10')
  ```
    
  4. Modify the configs
  ```
    a.)*mmdetection/configs/pascal_voc/ssd300_voc0712.py*
        num_classes設為10
        img_scale設為512*512
        step設為[7,9]
        total_epochs設為10
    b.)
       
  ```
  5. Start training:
  ```python3 ./tools/train.py ./configs/pascal_voc/ssd300_voc0712.py```
  

### Inference
```python3 ./demo/image_demo.py ./configs/pascal_voc/ssd300_voc0712.py ./work_dirs/epoch10.pth --root data/VOCdevkit/VOC2007/test/```
