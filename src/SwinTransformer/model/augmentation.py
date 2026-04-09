import torchvision.transforms as transforms

class DataAugmentation:

    def random_crop(self, x, patch_size):
        transform = transforms.RandomCrop(patch_size, padding=4)
        crop = transform(x)
        return crop

    def random_flip(self, x):
        transform = transforms.RandomHorizontalFlip(p=0.5)
        flip = transform(x)
        return flip

