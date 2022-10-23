import config
import tarfile
import torchvision.transforms as tt
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_url


class Dataset:
    def download_dataset(self):
        download_url(url=config.DATASET_URL, root='.', filename='cifar10.tgz', md5='')
        with tarfile.open('cifar10.tgz', 'r:gz') as tar:
            
            import os
            
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, path="./input")
        print('[INFO] dataset download completed')

    def get_dataloader(self):
        stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        train_tfms = tt.Compose([tt.RandomCrop(32, padding=4),
                                 tt.RandomHorizontalFlip(),
                                 tt.ToTensor(),
                                 tt.Normalize(*stats)])
        valid_tfms = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)])
        train_ds = ImageFolder(config.DATA_DIR + '/cifar10/train', train_tfms)
        valid_ds = ImageFolder(config.DATA_DIR + '/cifar10/test', valid_tfms)
        train_dl = DataLoader(train_ds, config.TRAIN_BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
        valid_dl = DataLoader(valid_ds, config.VALID_BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)
        return train_dl, valid_dl
