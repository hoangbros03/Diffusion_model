import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms

class DiffSet(Dataset):
    def __init__(self, train, dataset="MNIST"):
        transform = transforms.Compose([transforms.ToTensor()])

        datasets = {
            "MNIST": MNIST,
            "Fashion": FashionMNIST,
            "CIFAR": CIFAR10
        }

        train_dataset = datasets[dataset](
            "./data", download=True, train=train, transform=transform
        )

        self.dataset_len = len(train_dataset.data)

        if dataset=="MNIST" or dataset=="Fashion":
            pad = transforms.Pad(2)
            data = pad(train_dataset.data)
            data = data.unsqueeze(3)
            self.depth = 1
            self.size = 32 # 28+2+2
        elif dataset == "CIFAR":
            data = torch.Tensor(train_dataset.data)
            self.depth=3
            self.size=32
        self.input_seq = ((data/255.0)*2.0)-1.0
        self.input_seq = self.input_seq.moveaxis(3,1) # batch_size, channel, w, h

    def __len__(self):
        return self.dataset_len
    
    def __getitem__(self, item):
        return self.input_seq[item]
    
def get_dataloader(choice, batch_size):
    """_summary_

    Args:
        choice (_type_): _description_
    """
    train_dataset = DiffSet(True, choice)
    val_dataset = DiffSet(False, choice)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, num_workers=4, shuffle=False)

    info_dict ={
        "depth": train_dataset.depth,
        "size": train_dataset.size
    }
    return train_loader, val_loader, info_dict