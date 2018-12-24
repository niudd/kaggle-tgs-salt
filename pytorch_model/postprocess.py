import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm_notebook
from keras.models import load_model
from metrics import *
import cv2


def smooth_mask(_img, kernel_size=5):
    _img = _img.astype(np.uint8)*255
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    erosion = cv2.erode(_img, kernel, iterations = 1)
    dilation = cv2.dilate(erosion, kernel, iterations = 1)
    return (dilation/255).astype(int)


# model = load_model(save_model_name, custom_objects={'my_iou_metric_2': my_iou_metric_2,
#                                                    'lovasz_loss': lovasz_loss})

def predict_result(model,x_test,img_size_target):
    """predict both orginal and reflect x"""
    x_test_reflect =  np.array([np.fliplr(x) for x in x_test])
    preds_test = model.predict(x_test).reshape(-1, img_size_target, img_size_target)
    preds_test2_refect = model.predict(x_test_reflect).reshape(-1, img_size_target, img_size_target)
    preds_test += np.array([ np.fliplr(x) for x in preds_test2_refect] )
    return preds_test/2

def get_best_threshold(x_valid, y_valid, img_size_target, model, plot=True):

    preds_valid = predict_result(model,x_valid,img_size_target)

    ## Scoring for last model, choose threshold by validation data 
    thresholds_ori = np.linspace(0.3, 0.7, 31)
    # Reverse sigmoid function: Use code below because the  sigmoid activation was removed
    thresholds = np.log(thresholds_ori/(1-thresholds_ori)) 

    # ious = np.array([get_iou_vector(y_valid, preds_valid > threshold) for threshold in tqdm_notebook(thresholds)])
    # print(ious)
    ious = np.array([iou_metric_batch(y_valid, preds_valid > threshold) for threshold in tqdm_notebook(thresholds)])
    print(ious)

    # instead of using default 0 as threshold, use validation data to find the best threshold.
    threshold_best_index = np.argmax(ious) 
    iou_best = ious[threshold_best_index]
    threshold_best = thresholds[threshold_best_index]
    
    if plot:
        plt.plot(thresholds, ious)
        plt.plot(threshold_best, iou_best, "xr", label="Best threshold")
        plt.xlabel("Threshold")
        plt.ylabel("IoU")
        plt.title("Threshold vs IoU ({}, {})".format(threshold_best, iou_best))
        plt.legend()
    return threshold_best, iou_best


"""
used for converting the decoded image to rle mask
Fast compared to previous one
"""
def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten(order = 'F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# x_test = np.array([(np.array(load_img("data/raw/test_images/images/{}.png".format(idx), \
#                                       color_mode = "grayscale"))) / 255 \
#                    for idx in tqdm_notebook(test_df.index)]).reshape(-1, img_size_target, img_size_target, 1)
# preds_test = predict_result(model,x_test,img_size_target)#.shape=(800, 101, 101)









