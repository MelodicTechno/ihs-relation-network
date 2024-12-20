import os
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
from PIL import Image
import torch


# OmniglotCharacterFoders: 加载Omniglot数据集中的所有字符文件夹
def omniglot_character_folders(data_folder):
    character_folders = [
        os.path.join(data_folder, family, character)
        for family in os.listdir(data_folder)
        if os.path.isdir(os.path.join(data_folder, family))
        for character in os.listdir(os.path.join(data_folder, family))
    ]
    random.shuffle(character_folders)
    num_train = 1200
    metatrain_character_folders = character_folders[:num_train]
    metaval_character_folders = character_folders[num_train:]
    return metatrain_character_folders, metaval_character_folders


class OmniglotTask:
    def __init__(self, character_folders, num_classes, train_num, test_num):
        self.character_folders = character_folders
        self.num_classes = num_classes
        self.train_num = train_num
        self.test_num = test_num
        self._prepare_task()

    def _prepare_task(self):
        class_folders = random.sample(self.character_folders, self.num_classes)
        labels = {c: i for i, c in enumerate(class_folders)}
        samples = {c: random.sample([os.path.join(c, x) for x in os.listdir(c)], len(os.listdir(c))) for c in class_folders}

        self.train_roots = [samples[c][:self.train_num] for c in class_folders]
        self.test_roots = [samples[c][self.train_num:self.train_num + self.test_num] for c in class_folders]

        self.train_labels = [labels[self.get_class(x)] for x in sum(self.train_roots, [])]
        self.test_labels = [labels[self.get_class(x)] for x in sum(self.test_roots, [])]

    def get_class(self, sample):
        return os.path.dirname(sample)


class OmniglotDataset(Dataset):
    def __init__(self, task, split='train', transform=None, target_transform=None):
        self.task = task
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.image_roots = self.task.train_roots if self.split == 'train' else self.task.test_roots
        self.labels = self.task.train_labels if self.split == 'train' else self.task.test_labels

    def __len__(self):
        return len(self.image_roots)

    def __getitem__(self, idx):
        # Ensure image_root points to a specific file (not a list)
        image_root = self.image_roots[idx]

        # Check if image_root is a list (it might be due to multiple images per class)
        if isinstance(image_root, list):
            image_root = image_root[0]  # Take the first image if it's a list

        # Open the image
        image = Image.open(image_root)
        image = image.convert('L')  # Convert to grayscale
        image = image.resize((28, 28), resample=Image.LANCZOS)  # Resize to 28x28

        # Apply transformations if any
        if self.transform is not None:
            image = self.transform(image)

        # Get the corresponding label
        label = self.labels[idx]

        # Apply target transformation if any
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label


class ClassBalancedSampler(Sampler):
    def __init__(self, num_per_class, num_cl, num_inst, shuffle=True):
        super().__init__()
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            batch = [random.sample(range(self.num_inst), self.num_per_class) for _ in range(self.num_cl)]
        else:
            batch = [range(self.num_inst)[:self.num_per_class] for _ in range(self.num_cl)]
        batch = [item for sublist in batch for item in sublist]
        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return self.num_cl * self.num_per_class


# 增强了数据处理和样本对生成
def get_data_loader(task, num_per_class=1, split='train', shuffle=True, rotation=0):
    transform = transforms.Compose([
        transforms.RandomRotation(30),  # 旋转增强
        transforms.RandomHorizontalFlip(),  # 水平翻转
        transforms.RandomResizedCrop(28),  # 随机裁剪并缩放
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.92206], std=[0.08426])
    ])

    # 创建数据集
    dataset = OmniglotDataset(task, split=split, transform=transform)

    # 创建采样器
    if split == 'train':
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.train_num, shuffle=shuffle)
    else:
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.test_num, shuffle=shuffle)

    # 创建DataLoader
    loader = DataLoader(dataset, batch_size=num_per_class * task.num_classes, sampler=sampler)
    return loader
