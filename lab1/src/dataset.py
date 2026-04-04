"""Модуль для загрузки и подготовки датасета Satellite Images of Water Bodies."""

import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2


class WaterBodiesDataset(Dataset):
    """Датасет спутниковых снимков водных объектов для бинарной сегментации.

    Args:
        image_paths: Список путей к изображениям.
        mask_paths: Список путей к маскам.
        transform: Аугментации albumentations.
        img_size: Размер изображения (высота, ширина) для ресайза.
    """

    def __init__(self, image_paths, mask_paths, transform=None, img_size=(256, 256)):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        """Возвращает количество изображений в датасете."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Возвращает пару (изображение, маска) по индексу.

        Args:
            idx: Индекс элемента.

        Returns:
            dict: Словарь с ключами 'image' (тензор [3, H, W]) и 'mask' (тензор [1, H, W]).
        """
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.img_size)

        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.img_size)
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]          # [3, H, W]
            mask = augmented["mask"]            # [H, W]
            mask = mask.unsqueeze(0).float()    # [1, H, W]  ← добавляем канал
        else:
            image = image.astype(np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1))
            mask = np.expand_dims(mask, axis=0)

        return {"image": image, "mask": mask}


def get_base_transforms(img_size=(256, 256)):
    """Базовые трансформации без аугментаций (для бейзлайна).

    Args:
        img_size: Размер изображения (высота, ширина).

    Returns:
        Трансформация albumentations для train и val.
    """
    transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    return transform


def get_augmented_transforms(img_size=(256, 256)):
    """Трансформации с аугментациями для улучшенного бейзлайна.

    Args:
        img_size: Размер изображения (высота, ширина).

    Returns:
        tuple: (train_transform, val_transform).
    """
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    val_transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    return train_transform, val_transform


def load_data_paths(data_dir):
    """Загружает пути к изображениям и маскам из директории датасета.

    Args:
        data_dir: Путь к корневой папке датасета.

    Returns:
        tuple: (image_paths, mask_paths) — отсортированные списки путей.
    """
    images_dir = os.path.join(data_dir, "Images")
    masks_dir = os.path.join(data_dir, "Masks")

    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif"))
    ])
    mask_files = sorted([
        f for f in os.listdir(masks_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif"))
    ])

    image_paths = [os.path.join(images_dir, f) for f in image_files]
    mask_paths = [os.path.join(masks_dir, f) for f in mask_files]

    assert len(image_paths) == len(mask_paths), (
        f"Количество изображений ({len(image_paths)}) не совпадает "
        f"с количеством масок ({len(mask_paths)})"
    )

    return image_paths, mask_paths


def get_dataloaders(data_dir, batch_size=16, img_size=(256, 256),
                    transform=None, val_split=0.2, test_split=0.1,
                    random_state=42, num_workers=2):
    """Создаёт DataLoader'ы для train, val и test.

    Args:
        data_dir: Путь к корневой папке датасета.
        batch_size: Размер батча.
        img_size: Размер изображения.
        transform: Словарь {'train': ..., 'val': ...} с трансформациями.
                   Если None, используются базовые.
        val_split: Доля валидационной выборки.
        test_split: Доля тестовой выборки.
        random_state: Seed для воспроизводимости.
        num_workers: Количество процессов для загрузки данных.

    Returns:
        dict: {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}.
    """
    image_paths, mask_paths = load_data_paths(data_dir)

    # Split: train / (val + test)
    train_imgs, temp_imgs, train_masks, temp_masks = train_test_split(
        image_paths, mask_paths, test_size=val_split + test_split,
        random_state=random_state
    )

    # Split: val / test
    relative_test = test_split / (val_split + test_split)
    val_imgs, test_imgs, val_masks, test_masks = train_test_split(
        temp_imgs, temp_masks, test_size=relative_test,
        random_state=random_state
    )

    if transform is None:
        base = get_base_transforms(img_size)
        transform = {"train": base, "val": base}

    train_dataset = WaterBodiesDataset(train_imgs, train_masks, transform["train"], img_size)
    val_dataset = WaterBodiesDataset(val_imgs, val_masks, transform["val"], img_size)
    test_dataset = WaterBodiesDataset(test_imgs, test_masks, transform["val"], img_size)

    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True),
    }

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    return dataloaders
