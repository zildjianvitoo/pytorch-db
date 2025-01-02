
from pathlib import Path
from going_modular.data_setup import create_dataloaders
from going_modular.model_builder import TinyVGG
from going_modular.engine import train_model
from going_modular.utils import save_model
from torchvision.transforms import v2
import torch
import os
from torchmetrics import Accuracy

device = "cuda" if torch.cuda.is_available() else "cpu"

data_path = Path("data/pizza_steak_sushi")

train_dir = data_path / "train"
test_dir = data_path / "test"

train_transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize(size=(64,64)),
    v2.TrivialAugmentWide(num_magnitude_bins=31),
    v2.ToDtype(dtype=torch.float32,scale=True)
])

test_transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.Resize(size=(64, 64)),
        v2.ToDtype(dtype=torch.float32, scale=True),
    ]
)

BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()

train_dataloader,test_dataloader, class_names = create_dataloaders(
                                                train_dir=train_dir,
                                                test_dir=test_dir,
                                                train_transforms=train_transforms,
                                                test_transforms=test_transforms,
                                                batch_size=BATCH_SIZE,
                                                num_workers=NUM_WORKERS)

model = TinyVGG(input_shape=3,hidden_units=32,output_shape=len(class_names)).to(device)

acc_metric = Accuracy(task="multiclass",num_classes=len(class_names)).to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(),lr=0.001)
from timeit import default_timer as timer 

model_results = train_model(model=model,train_dataloader=train_dataloader,test_dataloader=test_dataloader ,acc_metric=acc_metric, loss_fn=loss_fn,optimizer=optimizer,device=device,epochs=5)

save_model(model, "model", "model_1.pt")
print(model_results)
