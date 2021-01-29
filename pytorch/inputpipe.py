# %%
import torch
import torchvision
import torchvision.transforms as transforms

# %%
batch_size = 100

# Dataset
train_dataset = torchvision.datasets.MNIST(root='.',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)
# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for elems, labels in train_loader:
    elems = elems.to(device)
    labels = labels.to(device)
    break
# elems: (batch, data...)
# labels: (batch, data...)


# %%
# Custom dataset
# You should build your custom dataset as below.
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        # TODO
        # 1. Initialize file paths or a list of file names.
        pass

    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        pass

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return 0


# You can then use the prebuilt data loader.
custom_dataset = CustomDataset()
train_loader = torch.utils.data.DataLoader(dataset=custom_dataset, batch_size=64, shuffle=True)

# %%
