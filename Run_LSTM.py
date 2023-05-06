import torch
import torchvision
import torch.nn as nn
import ModelFiles
import numpy as np

#create meta-learner
my_ML = ModelFiles.MetaLearn()

#set to training mode
my_ML.train_mode()

#perform training
for i in range(0, 80):
    my_ML.forward()

#perform testing
my_ML.eval_mode()
res = np.zeros([1, 20])
for i in range(0, 20):
    res[i] = my_ML.forward()

#perform validation (ONLY DO ONCE!)
my_ML.validate_mode()
val_res = np.zeros([1, 25])
for i in range(0, 25):
    val_res[i] = my_ML.forward()
