import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset, DataLoader
from Utils import *
import torch

# Path to save the model
model_path = 'fasterrcnn_model.pth'

torch.cuda.empty_cache()

# Using the CPU instead of the GPU, for hardware limitations
device = torch.device('cpu')

# Load a pre-trained model for classification and return only the features
backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features
backbone.out_channels = 1280

# Define the RPN anchor generator
anchor_generator = AnchorGenerator(
    sizes=((32, 64, 128, 256, 512),),
    aspect_ratios=((0.5, 1.0, 2.0),)
)

# Define the ROI pooling
roi_pooler = torchvision.ops.MultiScaleRoIAlign(
    featmap_names=['0'],
    output_size=7,
    sampling_ratio=2
)

# Combining with Faster-RCNN model
model = FasterRCNN(
    backbone,
    num_classes=2,
    rpn_anchor_generator=anchor_generator,
    box_roi_pool=roi_pooler
)

model.to(device)

# Construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

# Define a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

# Number of epochs
num_epochs = 10


# Train and test datasets
# TODO: Replace paths according to your ds
dataset = DsLoader('Drone-Detection/train', get_transform(train=True))
dataset_test = DsLoader('Drone-Detection/valid', get_transform(train=False))

# Creating subsets for training and validation
train_indices = list(range(4000))
val_indices = list(range(500))

# Create subsets
train_subset = Subset(dataset, train_indices)
val_subset = Subset(dataset_test, val_indices)

# Define training and validation data loaders
data_loader = DataLoader(
    train_subset,
    batch_size=1,
    shuffle=True,
    collate_fn=utils.collate_fn
)

data_loader_test = DataLoader(
    val_subset,
    batch_size=1,
    shuffle=False,
    collate_fn=utils.collate_fn
)

early_stopper = EarlyStopper(patience=1, min_delta=10)

# Train main loop
for epoch in range(num_epochs):
    # Train for one epoch, printing every 10 iterations
    avg_train_loss = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    
    if early_stopper.should_stop(avg_train_loss):
        print(f"Early stopping at epoch {epoch+1}.")
        break
    
    # Updating the learning rate
    lr_scheduler.step()

# Saving
torch.save(model.state_dict(), model_path)

