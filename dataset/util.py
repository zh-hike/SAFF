from torchvision.transforms import transforms as T


def get_default_aug(dataset):
    size = 256
    if dataset == 'Aerial':
        size = 600
    aug = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0, 0, 0], std=[0.5, 0.5, 0.5])
    ])

    return aug
