import os
import torch
from torchvision.io import read_image
from torchvision.transforms import Resize
from torch.utils.data import Dataset
import cv2
import os
import torch
from torchvision.io import read_image
from torchvision.transforms import Resize
from torch.utils.data import Dataset

class DsLoader(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.resize = Resize((224, 224))  # Tuneable this one
        self.imgs = sorted(os.listdir(os.path.join(root, "images")))
        self.annos = sorted(os.listdir(os.path.join(root, "labels")))
        """
        self.indexer = 0
        self.indexer_stop = 30
        """
        
    def _find_next_valid_index(self, start_idx):
        """Find the next index with non-empty annotation starting from `start_idx`."""
        idx = start_idx
        while idx < len(self.imgs):
            anno_path = os.path.join(self.root, "labels", self.annos[idx])
            if os.path.exists(anno_path) and os.path.getsize(anno_path) > 0:
                # Verify non-empty annotations
                 empty = True
                 with open(anno_path, "r") as file:
                     lines = file.readlines()
                     for line in lines:
                         parts = line.strip().split()
                         binary_label, a, b, c, d = map(float, parts)
                         if a < c and b < d:
                             empty = False 
                 if not empty:
                    return idx
            idx += 1
        return None  # Return None if no valid annotation is found

    def __getitem__(self, idx):
        # Find the next valid index with non-empty annotation
        valid_idx = self._find_next_valid_index(idx)
        if valid_idx is None:
            raise IndexError("No valid annotations found in the dataset.")  # Or handle as needed
        
        img_path = os.path.join(self.root, "images", self.imgs[valid_idx])
        anno_path = os.path.join(self.root, "labels", self.annos[valid_idx])
        
        img = read_image(img_path).float()  # Convert image to float
        original_size = img.shape[1:]  # Original (H, W)

        # Initialize empty list for bounding boxes
        boxes_per_anno = []
        
        with open(anno_path, "r") as file:
            lines = file.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue  # Skip lines that don't have the expected number of parts
                
                binary_label, a, b, c, d = map(float, parts)
                
                x1 = a - c / 2
                y1 = b - d / 2
                x2 = a + c / 2
                y2 = b + d / 2
                
                # Verify that bounding box(es) have positive height and width.
                if x1 >= x2 or y1 >= y2:
                    continue
                boxes_per_anno.append([x1,y1,x2,y2])
        
        # Convert list of bounding boxes to tensor
        anno = torch.tensor(boxes_per_anno, dtype=torch.float32)
        
        # Create target dictionary
        target = {
            "boxes": anno if len(boxes_per_anno) > 0 else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.ones((len(anno),), dtype=torch.int64) if len(boxes_per_anno) > 0 else torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([valid_idx])
        }

        # Resize the image
        img = self.resize(img)
        new_size = img.shape[1:]  # New (H, W)
        scale_x = new_size[1] / original_size[1]
        scale_y = new_size[0] / original_size[0]
        
        scale_text = 'Scaling-var-x: ' + str(scale_x) + ', Scaling-var-y: ' + str(scale_y)
        
        if self.transforms:
            img, target = self.transforms(img, target)
        """
        # ----- For displaying -----------------
        boxes_as_list = target['boxes'].tolist()
        imgd = cv2.imread(img_path)
        imgd = cv2.resize(imgd, (224,224))
        for box in boxes_as_list:
            (x1,y1,x2,y2) = box
            x1 = int(x1 * 224)
            y1 = int(y1 * 224)
            x2 = int(x2 * 224)
            y2 = int(y2 * 224)
            
            if self.indexer < self.indexer_stop:
                cv2.rectangle(imgd, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue box with thickness of 2
                cv2.putText(imgd, 'Drone', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                cv2.imshow(scale_text, imgd)
                cv2.waitKey(6000)
                cv2.destroyAllWindows()

        self.indexer += 1
        """
        # --------------------------------------
        
        return img, target

    def __len__(self):
        return len(self.imgs)
