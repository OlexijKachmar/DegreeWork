
from torch.utils.data import Dataset, DataLoader
from skimage import io
import torchvision.transforms.functional as TF
from torchvision import transforms


class MyData(Dataset):

    def __init__(self, dataset, augmentation):

        self.image_sets = dataset.imgs  # [TF.to_pil_image(dataset[x][0]) for x in range(len(dataset))]
        #         self.labels = [dataset[x][1] for x in range(len(dataset))]
        self.const_augmentation = augmentation
        self.set_stage(0)

    def __getitem__(self, index):
        image = io.imread(self.image_sets[index][0])
        image = TF.to_pil_image(image)

        image = self.resize(image)
        x = self.const_augmentation(image)

        label = self.image_sets[index][1]
        return x, label

    @staticmethod
    def resizer(img):
        return TF.resize(TF.resize(img, (32, 32)), (64, 64))

    def set_stage(self, stage):

        if stage == 0:
            print('Using (32, 32) resize')
            self.resize = transforms.Resize((32, 32))
        elif stage == 1:
            print('Using (64,64) resize')
            self.resize = self.resizer

    def __len__(self):
        return len(self.image_sets)