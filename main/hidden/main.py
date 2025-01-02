
import torch
from options import *
from model.Combine import combine_model
from train import train
import utils


def main():
    # 选择device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # train的配置
    train_options = TrainingOptions(
        H=256,W=256,
        batch_size=10,
        number_of_epochs=100,
        train_folder='/opt/data/tiewei/datasets/tip2018',
        validation_folder='/opt/data/tiewei/datasets/tip2018/testData',
        # train_folder='/workshop/tiewei/datasets/tip2018',
        # validation_folder='/workshop/tiewei/datasets/tip2018/testData',
        start_epoch=1,
        # 训练
        # nrow=8, train_data_len=27000, val_data_len=4500,
        # 测试代码
        nrow=8, train_data_len=200, val_data_len=50,
        experiment_name='test1')

    model = combine_model(device)
    train(model, device, train_options)

if __name__ == '__main__':
    main()
