import numpy as np

from tensorflow import keras
import cloud_utils
import cv2
import albumentations as albu

"""
# Data Generator

taken from kaggle kernel Unet_Final - see link above

Aggmentations are not explored here although they are turned on the effect of using them has not been examined

"""
class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, list_IDs, df, target_df=None, mode='fit',
                 base_path='/content/drive/MyDrive/understanding_clouds/data/train_images',
                 batch_size=32, dim=(1400, 2100), n_channels=1, reshape=None,
                 augment=False, n_classes=4, random_state=2019, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.df = df
        self.mode = mode
        self.base_path = base_path
        self.target_df = target_df
        self.list_IDs = list_IDs
        self.reshape = reshape
        self.n_channels = n_channels
        self.augment = augment
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.random_state = random_state

        self.on_epoch_end()
        np.random.seed(self.random_state)

        ###
        self.imgs = {}
        keys = list_IDs

        for k in keys:
            im_name = self.df['ImageID'].iloc[k]
            img_path = f"{self.base_path}/{im_name}"
            if self.reshape is None:
                self.imgs[k] = self.__load_grayscale(img_path)
            else:
                self.imgs[k] = cloud_utils.np_resize(self.__load_grayscale(img_path), self.reshape)

            self.imgs[k] = self.imgs[k].reshape((self.imgs[k].shape[0], self.imgs[k].shape[1], 1))

        #

        self.masks = {}

        for k in keys:
            im_name = self.df['ImageID'].iloc[k]
            img_path = f"{self.base_path}/{im_name}"
            if self.reshape is None:
                self.imgs[k] = self.__load_rgb(img_path)
            else:
                self.imgs[k] = cloud_utils.np_resize(self.__load_rgb(img_path), self.reshape)

            self.imgs[k] = self.imgs[k].reshape((self.imgs[k].shape[0], self.imgs[k].shape[1], 3))

        for k in keys:
            im_name = self.df['ImageID'].iloc[k]
            image_df = self.target_df[self.target_df['ImageID'] == im_name]

            rles = image_df['EncodedPixels'].values

            if self.reshape is not None:
                masks = cloud_utils.build_masks(rles, input_shape=self.dim, reshape=self.reshape)
            else:
                masks = cloud_utils.build_masks(rles, input_shape=self.dim)

            self.masks[k] = masks

        #

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_batch = [self.list_IDs[k] for k in indexes]

        X = self.__generate_X(list_IDs_batch)

        if self.mode == 'fit':
            y = self.__generate_y(list_IDs_batch)

            if self.augment:
                X, y = self.__augment_batch(X, y)

            return X, y

        elif self.mode == 'predict':
            return X

        else:
            raise AttributeError('The mode parameter should be set to "fit" or "predict".')

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:  # == True:
            np.random.seed(self.random_state)
            np.random.shuffle(self.indexes)

    def __generate_X(self, list_IDs_batch):
        """Generates data containing batch_size samples"""
        # Initialization
        if self.reshape is None:
            X = np.empty((self.batch_size, *self.dim, self.n_channels))
        else:
            X = np.empty((self.batch_size, *self.reshape, self.n_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_batch):
            im_name = self.df['ImageID'].iloc[ID]
            img_path = f"{self.base_path}/{im_name}"
            img = self.imgs[ID]

            X[i,] = img

        return X

    def __generate_y(self, list_IDs_batch):
        if self.reshape is None:
            y = np.empty((self.batch_size, *self.dim, self.n_classes), dtype=float)
        else:
            y = np.empty((self.batch_size, *self.reshape, self.n_classes), dtype=float) 

        for i, ID in enumerate(list_IDs_batch):
            y[i, ] = self.masks[ID]

        return y

    def __load_grayscale(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.
        img = np.expand_dims(img, axis=-1)

        return img

    def __load_rgb(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.

        return img

    def __random_transform(self, img, masks):
        composition = albu.Compose([
            albu.HorizontalFlip(p=0.25),
            albu.VerticalFlip(p=0.25),
            albu.Transpose(p=0.25)
        ])

        composed = composition(image=img, mask=masks)
        aug_img = composed['image']
        aug_masks = composed['mask']

        return aug_img, aug_masks

    def __augment_batch(self, img_batch, masks_batch):
        for i in range(img_batch.shape[0]):
            img_batch[i, ], masks_batch[i, ] = self.__random_transform(
                img_batch[i, ], masks_batch[i, ])

        return img_batch, masks_batch

    def getitem(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_batch = [self.list_IDs[k] for k in indexes]

        X = self.__generate_X(list_IDs_batch)

        if self.mode == 'fit':
            y = self.__generate_y(list_IDs_batch)

            if self.augment:
                X, y = self.__augment_batch(X, y)

            return X, y

        elif self.mode == 'predict':
            return X

        else:
            raise AttributeError('The mode parameter should be set to "fit" or "predict".')