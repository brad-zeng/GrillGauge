import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

dataset = datasets.ImageFolder('./detect_steak.v2i.tensorflow/', transform=transforms.ToTensor())

loader = DataLoader(dataset, batch_size=10, num_workers=0, shuffle=False)

mean = 0.0
for images, _ in loader:
    batch_samples = images.size(0) 
    images = images.view(batch_samples, images.size(1), -1)
    mean += images.mean(2).sum(0)
mean = mean / len(loader.dataset)

var = 0.0
for images, _ in loader:
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1)
    var += ((images - mean.unsqueeze(1))**2).sum([0,2])
std = torch.sqrt(var / (len(loader.dataset)*416*416))

print(mean)
print(std)