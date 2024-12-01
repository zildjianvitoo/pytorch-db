
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import os

NUM_WORKERS = os.cpu_count()
BATCH_SIZE = 32

def create_dataloaders(
    train_dir: str,
    test_dir: str,
    train_transforms: v2.Compose,
    test_transforms: v2.Compose,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
):
    train_data = ImageFolder(root=train_dir,transform=train_transforms)
    test_data = ImageFolder(root=test_dir, transform=test_transforms)
    class_names = train_data.classes

    train_dataloader = DataLoader(
        dataset=train_data, batch_size=batch_size, num_workers=num_workers,shuffle=True,pin_memory=True
    )
    test_dataloader = DataLoader(
        dataset=test_data, batch_size=batch_size, num_workers=num_workers,pin_memory=True
    )
    
    len(train_dataloader)

    return train_dataloader,test_dataloader,class_names
