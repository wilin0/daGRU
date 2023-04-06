import os
import logging
import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True

def print_log(message):
    print(message)
    logging.info(message)

def output_namespace(namespace):
    configs = namespace.__dict__
    message = ''
    for k, v in configs.items():
        message += '\n' + k + ': \t' + str(v) + '\t'
    return message

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def reshape_patch(img_tensor, patch_size):
    # [batch, time, channel, height, width]
    img_tensor = img_tensor.permute(0, 1, 3, 4, 2).contiguous()
    assert 5 == img_tensor.ndim
    img_np = img_tensor.detach().cpu().numpy()
    batch_size = np.shape(img_np)[0]
    seq_length = np.shape(img_np)[1]
    img_height = np.shape(img_np)[2]
    img_width = np.shape(img_np)[3]
    num_channels = np.shape(img_np)[4]
    a = np.reshape(img_np, [batch_size, seq_length,
                                img_height//patch_size, patch_size,
                                img_width//patch_size, patch_size,
                                num_channels])
    b = np.transpose(a, [0,1,2,4,3,5,6])
    patch_np = np.reshape(b, [batch_size, seq_length,
                                  img_height//patch_size,
                                  img_width//patch_size,
                                  patch_size*patch_size*num_channels])
    patch_tensor = torch.from_numpy(patch_np)
    patch_tensor = patch_tensor.permute(0, 1, 4, 2, 3).contiguous()

    return patch_tensor


def reshape_patch_back(patch_tensor, patch_size):
    patch_tensor = patch_tensor.permute(0, 1, 3, 4, 2).contiguous()
    assert 5 == patch_tensor.ndim
    patch_np = patch_tensor.detach().cpu().numpy()
    batch_size = np.shape(patch_np)[0]
    seq_length = np.shape(patch_np)[1]
    patch_height = np.shape(patch_np)[2]
    patch_width = np.shape(patch_np)[3]
    channels = np.shape(patch_np)[4]
    img_channels = channels // (patch_size*patch_size)
    a = np.reshape(patch_np, [batch_size, seq_length,
                                  patch_height, patch_width,
                                  patch_size, patch_size,
                                  img_channels])
    b = np.transpose(a, [0,1,2,4,3,5,6])
    img_np = np.reshape(b, [batch_size, seq_length,
                                patch_height * patch_size,
                                patch_width * patch_size,
                                img_channels])
    img_tensor = torch.from_numpy(img_np)
    img_tensor = img_tensor.permute(0, 1, 4, 2, 3).contiguous()
    return img_tensor