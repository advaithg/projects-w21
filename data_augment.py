import torchvision.transforms as transforms

class DataAugmenter:
    def __init__(self): 
        # TODO: try RandomPerspective and Normalize
        self.affine = transforms.RandomAffine(degrees=45, scale=(0.8, 1.6))
        self.flip = transforms.RandomHorizontalFlip(0.5)
        self.perspective = transforms.RandomPerspective()

    def applyAugmentations(self, image):
        image = self.affine.forward(image)
        image = self.flip.forward(image)
        image = self.perspective.forward(image)

        return image