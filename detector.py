import torch
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.io import read_image
from torchvision.transforms import functional as F
from torchvision.transforms import v2 as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import FasterRCNN
import torchvision
import cv2

# Load the model
model_path = 'fasterrcnn_model.pth'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Define the RPN anchor generator
anchor_generator = AnchorGenerator(
    sizes=((32, 64, 128, 256, 512),),
    aspect_ratios=((0.5, 1.0, 2.0),)
)

# Define the ROI pooler
roi_pooler = torchvision.ops.MultiScaleRoIAlign(
    featmap_names=['0'],
    output_size=7,
    sampling_ratio=2
)

# Load your backbone here (resnet, mobilenet, etc.)
backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features
backbone.out_channels = 1280

model = FasterRCNN(
    backbone,
    num_classes=2,
    rpn_anchor_generator=anchor_generator,
    box_roi_pool=roi_pooler
)

model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

base_path = './Drone-Detection/test/images/'

image_paths = ['0110_jpg.rf.84685a6245e4af80e66ae1589d4e8574.jpg', '0008.jpg','2772_jpg.rf.6e56dad1247d721b03f891104fd4f08d.jpg','D-B-2_001_jpg.rf.c2bd5230b34ecc439479c46459d35861.jpg', 'D-B-3_005_jpg.rf.4296f6b16b9b8015b69f0b56b0963a8b.jpg']

indexer = 0
for img_path in image_paths:
    image_path = base_path + image_paths[indexer] 
    image = read_image(image_path)
    w = image.size(dim=1)
    h = image.size(dim=2)
    print('w: ', w, ' h: ', h)
    eval_transform = get_transform(train=False)
    indexer += 1
    # Switch model to evaluation mode
    model.eval()
    with torch.no_grad():
        x = eval_transform(image)
        # Convert RGBA -> RGB and move to device
        x = x[:3, ...].to(device)
        predictions = model([x])
        pred = predictions[0]
        
    boxes_as_list = pred['boxes'].tolist()
    print(boxes_as_list)
    print(predictions)
    print(pred)
    for box in boxes_as_list:
        box[0] = int(box[0] * w) + int((box[0] * w) / 2)
        box[1] = int(box[1] * h) + int((box[1] * w) / 2)
        box[2] = int(box[2] * w) - int((box[2] * w) / 2)
        box[3] = int(box[3] * h) - int((box[3] * w) / 2)
    
    image = cv2.imread(image_path)
    for box in boxes_as_list:
        (x1,y1,x2,y2) = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue box with thickness of 2
        cv2.putText(image, 'Pred:', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    
    cv2.imshow("Predictions", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
