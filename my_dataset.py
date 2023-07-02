import numpy as np
from os.path import join as pjoin
import torch
from torch.utils.data import Dataset
from keras.utils import to_categorical


class MyDataset(Dataset):
    def __init__(self, split='train', loc='data/', transform = None):
        self.root = loc
        self.split = split
        self.n_classes = 6 
        self.transform = transform

        self.seismic = np.load(pjoin(self.root,'train','train_seismic.npy'))
        # seismic.shape = (701, 401, 255)
        self.labels  = np.load(pjoin(self.root,'train','train_labels.npy'   ))
        # labels.shape = (701, 401, 255)
        self.labels  = to_categorical(self.labels,num_classes=self.n_classes)
        # seismic.shape = (701, 401, 255, 6)

        path = pjoin(self.root, 'splits', self.split + '.txt')
        patch_list = tuple(open(path, 'r'))
        self.patch_list = patch_list

    def __len__(self):
        return len(self.patch_list)
    
    def __getitem__(self, index):
        indexes = self.patch_list[index]
        direction, number = indexes.split(sep='_')
        
        if direction == 'i':
            im = self.seismic[int(number),:,:]
            lbl = self.labels[int(number),:,:,:]
        elif direction == 'x':    
            im = self.seismic[:,int(number),:]
            lbl = self.labels[:,int(number),:,:]
        # im.shape = (701 or 401, 255)
        # lbl.shape = (701 or 401, 255, 6)

        assert im.shape[1] == 255 and lbl.shape[-1] == 6, f'wrong shapes: {im.shape} and {lbl.shape}'

        if self.transform:
            stacked = np.dstack([np.expand_dims(im, axis=-1), lbl])
            # stacked.shape = (701 or 401, 255, 7)
            stacked = self.transform(stacked)
            # stacke.shape = torch.Size([7, 256, 256])
            im, lbl = stacked[0], stacked[1:]
            # im.shape = torch.Size([256, 256])
            # lbl.shape = torch.Size([6, 256, 256])

        if isinstance(im, np.ndarray):
            im = torch.from_numpy(im)
        if isinstance(lbl, np.ndarray):
            lbl = torch.from_numpy(lbl)

        image_tensor = torch.unsqueeze(im, 0)
        image_tensor = image_tensor.float()
        label_tensor = lbl.float()
        # image_tensor.shape = torch.Size([1, 256, 256])
        # label_tensor.shape = torch.Size([6, 256, 256])

        return image_tensor, label_tensor