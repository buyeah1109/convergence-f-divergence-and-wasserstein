import torchvision.transforms as transforms
from utils import cifar_single_dataset, CustomImageDataset
from torchvision import datasets

def get_cifar10(opt, root):
    cifar_dataset = datasets.CIFAR10(
            root=root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(opt.img_size), 
                transforms.ToTensor(), 
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
            ),
        )

    return cifar_dataset

def get_onekind_cifar(label):
    return cifar_single_dataset(label=label)

def get_LSUN_church(opt, root, is_gray=False):

    lsun_transform = transforms.Compose([
        transforms.Resize((opt.img_size, opt.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, .5, .5], [.5, .5, .5])
    ])

    lsun_transform_gray = transforms.Compose([
        transforms.Resize((opt.img_size, opt.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize([0.5, .5, .5], [.5, .5, .5])
    ])

    if is_gray:
        return datasets.LSUN(root, ["church_outdoor_train"], transform=lsun_transform_gray)
    return datasets.LSUN(root, ["church_outdoor_train"], transform=lsun_transform)


def get_celeb(opt, root, is_gray=False):

    celeb_dataset = CustomImageDataset(
            img_dir=root, 
            transform=transforms.Compose(
                [transforms.Resize((opt.img_size, opt.img_size)), 
                transforms.ToTensor(), 
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
            ))

    celeb_dataset_gray = CustomImageDataset(
            img_dir=root, 
            transform=transforms.Compose(
                [transforms.Resize((opt.img_size, opt.img_size)), 
                transforms.Grayscale(3),
                transforms.ToTensor(), 
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
            ))
    
    if is_gray:
        return celeb_dataset_gray
    return celeb_dataset