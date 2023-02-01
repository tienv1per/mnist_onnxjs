import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset


train_loader = DataLoader(datasets.MNIST('data', train=True, download=True,
                                        transform=transforms.Compose([
                                            transforms.RandomAffine(
                                                degrees=30, translate=(0.5, 0.5), scale=(0.25, 1),
                                                shear=(-30, 30, -30, 30)
                                            ),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))
                                        ])),
                                        batch_size=64, shuffle=True)


test_loader = DataLoader(datasets.MNIST('data', train=False, 
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))   
                                        ])),
                                        batch_size=1000, shuffle=False)