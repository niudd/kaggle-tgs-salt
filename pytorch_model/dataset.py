import numpy as np
from torch.utils.data import DataLoader, Dataset
from augmentation import do_augmentation, get_seed
from augmentation_huang import *
import imgaug as ia


# Alternative resizing method by padding to (128,128,1) and reflecting the image to the padded areas
def pad_reflect(img, north=14, south=13, west=14, east=13):
    """
    ## usage:
    #new_image_arr = pad_reflect(img_arr, 14, 13, 14, 13)
    #plt.imshow(new_image_arr[0])
    """
    h = img.shape[0]
    w = img.shape[1]
    new_image = np.zeros((north+h+south,west+w+east))
    
    # Place the image in new_image
    new_image[north:north+h, west:west+w] = img
    
    new_image[north:north+h,0:west] = np.fliplr(img[:,:west])
    new_image[north:north+h,west+w:] = np.fliplr(img[:,w-east:])
    
    new_image[0:north,:] = np.flipud(new_image[north:2*north,:])
    new_image[north+h:,:] = np.flipud(new_image[north+h-south:north+h,:]) 
    
    return new_image.reshape(1,128,128)

def unpad_reflect(img, north=14, west=14, height=101, width=101):
    """
    #img_arr_back = unpad_reflect(new_image_arr, 14, 14, 101, 101)
    #plt.imshow(img_arr_back)
    """
    img = img.reshape(128, 128)
    return img[north:north+height,west:west+width]

def unpad_reflect256(img):
    img256 = img.reshape(256, 256)
    img202 = img256[27:27+202, 27:27+202]
    img101 = cv2.resize(img202.astype(np.float64), dsize=(101, 101))
    return img101


class TgsDataSet(Dataset):
    def __init__(self, x_train, y_train, transform=True):
        super().__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.transform = transform

    def __len__(self):
        return self.x_train.shape[0]
    
    def __getitem__(self, idx):
        image, mask = self.x_train[idx].reshape(101, 101), self.y_train[idx].reshape(101, 101)
        #image, mask = self.x_train[idx], self.y_train[idx]
        
        if self.transform:
            #seed = get_seed()
            #seed = int('1234'+str(idx))
            #seed = 1234
            #print('seed: ', seed)
            #ia.imgaug.seed(seed)
            #np.random.seed(seed)
            
            image_aug, mask_aug = do_augmentation(image, mask)
            #image_aug, mask_aug = do_center_pad_to_factor2(image_aug, mask_aug, factor=32)
            #return image_aug.reshape(1, 128, 128), mask_aug.reshape(1, 128, 128).astype(int)
            image_aug, mask_aug = do_center_pad_to_factor256(image_aug, mask_aug)
            return image_aug.reshape(1, 256, 256), mask_aug.reshape(1, 256, 256).astype(int)
        else:
            # for test set
            #image, mask = do_center_pad_to_factor2(image, mask, factor=32)
            #return image.reshape(1, 128, 128), mask.reshape(1, 128, 128).astype(int)
            image, mask = do_center_pad_to_factor256(image, mask)
            return image.reshape(1, 256, 256), mask.reshape(1, 256, 256).astype(int)





