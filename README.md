# Faster-RCNN-With-Pytorch

Faster-RCNN implementation in PyTorch for object detecting purposes.

## Main Features:

- **Dataloader for YOLO-formatted dataset.**
- **Transfer-learning pipeline** to learn Faster-RCNN detector with MobileNet (or some other) backbone.
- **Script showing example of inference and visualizing** for training transfer-learning.

If interested to use this:
Change paths according to your scenario in train.py and detector.py.
And have yolo-like dataset structured as:

```
.
├── test
│   ├── images
│   └── labels
├── train
│   ├── images
│   └── labels
└── valid
    ├── images
    └── labels
```

## Where:

- Each mage in images-folder (.jpg, .png etc.) have corresponfing label (.txt) pair with same name in labels folder.
E.g: cat1.jpg and cat1.txt

- Annotation formatted as: **0 0.525 0.4796875 0.2546875 0.253125**
- Where first item is label and four floats being bounding-box coordinates **normalized between 1 and 0**. 

Todo: Fix detector.py
