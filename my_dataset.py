from PIL import Image
import torch
from torch.utils.data import Dataset

# 数据集类
class APTOSDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None, has_labels=True):
        self.labels_frame = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.has_labels = has_labels

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        img_name = self.labels_frame.iloc[idx, 0]
        image = Image.open(f"{self.root_dir}/{img_name}.png")

        if self.transform:
            image = self.transform(image)
        if self.has_labels:
          label = self.labels_frame.iloc[idx, 1]
          return image, label
        else:
          return image, _
        
    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels