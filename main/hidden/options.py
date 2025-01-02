class TrainingOptions:
    """
    Configuration options for the training
    """

    def __init__(self,
                 batch_size: int,
                 train_folder: str, validation_folder: str,
                 train_data_len:int,val_data_len:int):
        self.batch_size = batch_size
        self.train_folder = train_folder
        self.validation_folder = validation_folder
        self.train_data_len=train_data_len
        self.val_data_len=val_data_len


