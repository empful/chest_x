import cxr_dataset as CXR
import eval_model as E
import model as M
import os



# you will need to customize PATH_TO_IMAGES to where you have uncompressed
# NIH images
PATH_TO_IMAGES = "/mnt/bd/medai-cv/cvpr/CXR8/images/images"
PATH_TO_CHECKPOINT='/mnt/bd/medai-cv/cvpr/CXR8/reproduce-chexnet/results_split3/checkpoint9'
WEIGHT_DECAY = 1e-4
LEARNING_RATE = 0.03
preds, aucs = M.train_cnn_split2(PATH_TO_IMAGES, LEARNING_RATE, WEIGHT_DECAY, checkpoint_dir=PATH_TO_CHECKPOINT)
# preds, aucs = M.train_cnn(PATH_TO_IMAGES, LEARNING_RATE, WEIGHT_DECAY)

