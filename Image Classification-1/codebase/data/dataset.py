# dataset class for image classfication task
import torchvision
from torchvision import datasets
from torchvision import transforms
import argparse
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0,currentdir)


from base_data_module import BaseDataModule, load_and_print_info

# Directory for downloading and storing data
DATA_DIRNAME = BaseDataModule.data_dirname() 

def _download_data():
    if(len(os.listdir(DATA_DIRNAME))):
        return
    zipurl = 'https://pytorch.tips/bee-zip'
    with urlopen(zipurl) as zipresp:
        with ZipFile(BytesIO(zipresp.read())) as zfile:
            zfile.extractall(DATA_DIRNAME)



class datamodule(BaseDataModule):
    """Image DataModule."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        
        self.train_transforms = transforms.Compose([
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize( [0.485, 0.456,0.406],
                                                              [0.229, 0.224, 0.225])
                                       ])
        self.val_transforms = transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize( [0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])
                                        ])

        self.input_dims = (3,224,224)
        self.output_dims = (1,)
        self.mapping = {0:"Ants", 1:"Bees"}
    
    

    
    def prepare_data(self, *args, **kwargs) -> None:
        """Download data."""
        _download_data()

    def setup(self, stage=None) -> None:
        """Split into train, val, test, and set dims."""
        self.data_train:torchvision.datasets.folder.ImageFolder = datasets.ImageFolder(
                                        root=DATA_DIRNAME/'hymenoptera_data'/'train',
                                        transform=self.train_transforms
                                    )
        self.data_val:torchvision.datasets.folder.ImageFolder = datasets.ImageFolder(
                                        root=DATA_DIRNAME/'hymenoptera_data'/'val',
                                        transform=self.val_transforms
                                  )
        self.data_test:torchvision.datasets.folder.ImageFolder = {}
    
    # change according to dataset properties 
    def __repr__(self):
        basic = f"Image Dataset\nNum classes: {len(self.mapping)}\nMapping: {self.mapping}\nDims: {self.input_dims}\n"
        if self.data_train is None and self.data_val is None and self.data_test is None:
            return basic
        
        x, y = next(iter(self.train_dataloader()))
        data = (
            f"Train/val/test sizes: {len(self.data_train)}, {len(self.data_val)}, {len(self.data_test)}\n"
            f"Batch x stats: {(x.shape, x.dtype, x.min(), x.mean(), x.std(), x.max())}\n"
            f"Batch y stats: {(y.shape, y.dtype, y.min(), y.max())}\n"
        )
        return basic + data

if __name__ == "__main__":
    load_and_print_info(datamodule)








