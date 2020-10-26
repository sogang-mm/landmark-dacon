from collections import defaultdict
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import default_loader
import pandas as pd
import os
from tqdm import tqdm
import random


class LandmarkDataset(Dataset):
    def __init__(self, base_dir, train_csv, category, transform=None, phase='train',seed=0):
        self.base_dir = base_dir
        self.category = category

        self.samples = self.split_samples(train_csv, phase, seed)

        self.loader = default_loader
        self.transform = transform if transform else transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        iid, cid = self.samples[idx]
        label = self.category[cid]
        path = os.path.join(self.base_dir, label, iid + '.JPG')
        img = self.transform(self.loader(path))
        return img, iid, cid, label

    def split_samples(self, train_csv,phase, seed):
        samples_per_class= defaultdict(list)
        for iid, cid in pd.read_csv(train_csv).values.tolist():
            samples_per_class[cid].append([iid, cid])

        samples = []
        random.seed(seed)
        for k, v in samples_per_class.items():
            random.shuffle(v)
            c = int(len(v) * 0.8)
            l = v[:c] if phase == 'train' else v[c:]
            samples.extend(l)
        return samples


class TestDataset(Dataset):
    def __init__(self, base_dir, submission_csv, category, transform=None):
        self.base_dir = base_dir
        self.cateogory = category
        self.test = pd.read_csv(submission_csv)['id'].values.tolist()

        self.loader = default_loader
        self.transform = transform if transform else transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()])

    def __len__(self):
        return len(self.test)

    def __getitem__(self, idx):
        iid = self.test[idx]
        path = os.path.join(self.base_dir, iid[0], iid + '.JPG')
        img = self.transform(self.loader(path))
        return img, iid


if __name__ == '__main__':
    train_dir = '/mldisk/nfs_shared_/ms/landmark_dacon/public/train_lnk'
    train_csv = '/mldisk/nfs_shared_/ms/landmark_dacon/public/train.csv'

    test_dir = '/mldisk/nfs_shared_/ms/landmark_dacon/public/test/'
    submission = '/mldisk/nfs_shared_/ms/landmark_dacon/public/sample_submission.csv'

    cat_csv = '/mldisk/nfs_shared_/ms/landmark_dacon/public/category.csv'
    category = [i[1] for i in pd.read_csv(cat_csv).values.tolist()]

    train_loader = DataLoader(LandmarkDataset(train_dir, train_csv, category,transform=None,phase='train'),
                              batch_size=32, shuffle=False, num_workers=4)
    valid_loader = DataLoader(LandmarkDataset(train_dir, train_csv, category, transform=None, phase='valid'),
                              batch_size=32, shuffle=False, num_workers=4)

    for i in tqdm(train_loader):
        pass

    test_loader = DataLoader(TestDataset(test_dir, submission, category), batch_size=32, shuffle=False, num_workers=4)
    for i in tqdm(test_loader):
        pass
