import config
from dataset import Dataset
from model import WideResNet22
from fastai.train import Learner
from fastai.metrics import accuracy
from torch.nn import functional as f
from fastai.basic_data import DataBunch

cifar10 = Dataset()
# cifar10.download_dataset()
train_dataloader, valid_dataloader = cifar10.get_dataloader()
model = WideResNet22(3, 10)

data = DataBunch(train_dataloader, valid_dataloader)
learner = Learner(data, model, loss_func=f.cross_entropy, metrics=[accuracy])
learner.clip = 0.1
learner.fit_one_cycle(config.EPOCHS, config.LEARNING_RATE, wd=1e-4)
