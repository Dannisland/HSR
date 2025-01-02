
import os
import cv2
import torch
from torchvision import transforms
import torchvision.utils
from torch.utils import data
from torch.utils.data import Dataset
from options import TrainingOptions

class mydataset(Dataset):
    def __init__(self,path,transform=None,size=None):
        super(mydataset,self).__init__()
        # clean_image
        self.a_path=os.path.join(path,"hidden-encoded")
        self.a_image_list=[x for x in os.listdir(self.a_path)]
        # 对数据集进行排序
        self.a_image_list.sort()
        # 控制数据集大小
        if size is not None:
            self.a_image_list = self.a_image_list[0:size]

        # noise_image
        self.b_path=os.path.join(path,"hidden-encoded") # hidden-encoded GTmod12
        self.b_image_list=[x for x in os.listdir(self.b_path)]
        # 对数据集进行排序
        self.b_image_list.sort()
        # 控制数据集大小
        if size is not None:
            self.b_image_list = self.b_image_list[0:size]

        self.transform=transform

    def __len__(self):
        return len(self.a_image_list)

    def __getitem__(self, item):

        a_image_path=os.path.join(self.a_path,self.a_image_list[item])
        a_image=cv2.imread(a_image_path)
        a_image = cv2.cvtColor(a_image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            a_image = self.transform(a_image)

        b_image_path=os.path.join(self.b_path,self.b_image_list[item])
        b_image=cv2.imread(b_image_path)
        b_image = cv2.cvtColor(b_image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            b_image = self.transform(b_image)

        return a_image,b_image

def get_data_loaders(train_options: TrainingOptions):
    """ Get torch data loaders for training and validation. The data loaders take a crop of the image,
    transform it into tensor, and normalize it."""
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            # transforms.CenterCrop(600),
            # transforms.Resize(256),

            # transforms.CenterCrop(400),
            transforms.Resize([128, 128]),
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            # transforms.CenterCrop(600),
            # transforms.Resize(256),

            # transforms.CenterCrop(400),
            transforms.Resize([128, 128]),
        ])
    }

    train_images = mydataset(train_options.train_folder,
                             data_transforms['train'],train_options.train_data_len)

    train_loader = torch.utils.data.DataLoader(train_images,
                                               batch_size=train_options.batch_size, shuffle=True,)

    validation_images = mydataset(train_options.validation_folder,
                                  data_transforms['test'],train_options.val_data_len)

    validation_loader = torch.utils.data.DataLoader(validation_images, batch_size=train_options.batch_size,
                                                    shuffle=True, )

    return train_loader, validation_loader

def create_folder_for_run(experiment_name):

    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    os.makedirs(os.path.join(experiment_name, 'checkpoints'))
    os.makedirs(os.path.join(experiment_name, 'save_images'))

    writer_path=os.path.join(experiment_name, 'tensorboard_log')
    os.makedirs(writer_path)

    return writer_path
