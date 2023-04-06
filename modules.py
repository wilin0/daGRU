from torch import nn
import torch
import numpy as np


class STGRUCell(nn.Module):
    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride):
        super(STGRUCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = (filter_size // 2, filter_size // 2)
        self._forget_bias = 1.0
        self.conv_t = nn.Sequential(
            nn.Conv2d(in_channel, 4 * num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding,
                      ),
            nn.LayerNorm([4 * num_hidden, height, width])
        )
        self.conv_s = nn.Sequential(
            nn.Conv2d(num_hidden, 4 * num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding,
                      ),
            nn.LayerNorm([4 * num_hidden, height, width])
        )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0)

    def forward(self, T_t, S_t):
        # print('yes')
        T_concat = self.conv_t(T_t)
        S_concat = self.conv_s(S_t)
        t_z, t_r, t_t, t_s = torch.split(T_concat, self.num_hidden, dim=1)
        s_z, s_r, s_t, s_s = torch.split(S_concat, self.num_hidden, dim=1)
        Z_t = torch.sigmoid(t_z + s_z + self._forget_bias)
        R_t = torch.sigmoid(t_r + s_r)
        T_tmp = torch.tanh(t_t + R_t * s_t)
        S_tmp = torch.tanh(s_s + R_t * t_s)
        T_new = (1 - Z_t) * T_tmp + Z_t * T_t
        S_new = (1 - Z_t) * S_tmp + Z_t * S_t
        # H_new = self.conv_last(torch.cat([T_new, S_new], dim=1))

        return T_new, S_new


class DualAttention(nn.Module):

    def __init__(self, image_channel, filter_size, stride, width):
        """
        初始化函数
        :param image_channel: 输入图片的通道数
        :param filter_size: 卷积核的大小
        :param stride: 卷积步长
        :param width: 输入图片的宽和高，宽和高需要一样
        """
        super().__init__()
        self.padding = filter_size // 2
        self.c_attn_ = nn.Sequential(
            nn.Conv2d(image_channel, image_channel, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([image_channel, width, width]),
            nn.ReLU(),
            nn.Conv2d(image_channel, image_channel, kernel_size=1, stride=1, padding=0),
            # nn.LayerNorm([num_hidden, width, width]),
            # nn.ReLU(),
            # nn.Dropout2d(p=0.9)
        )
        self.s_attn_ = nn.Sequential(
            nn.Conv2d(image_channel, image_channel, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([image_channel, width, width]),
            nn.ReLU(),
            nn.Conv2d(image_channel, image_channel, kernel_size=1, stride=1, padding=0),
            # nn.LayerNorm([num_hidden, width, width]),
            # nn.ReLU(),
            # nn.Dropout2d(p=0.9)
        )

    def attention_channel(self, in_query, in_keys, in_values):
        """
        在时间上的注意力机制
        :param self:
        :param in_query: attention Q
        :param in_keys: attention K
        :param in_values: attention V
        :return: 注意力机制的输出
        """
        q_shape = in_query.shape
        k_shape = in_keys.shape
        batch = q_shape[0]
        num_channels = q_shape[1]
        width = q_shape[2]
        height = q_shape[3]
        length = k_shape[1]
        query = in_query.reshape([batch, num_channels, -1])
        key = in_keys.reshape([batch, -1, height * width]).permute((0, 2, 1))
        value = in_values.reshape([batch, -1, height * width]).permute((0, 2, 1))
        attn = torch.matmul(query, key)
        attn = torch.nn.Softmax(dim=2)(attn)
        attn = torch.matmul(attn, value.permute(0, 2, 1))
        attn = attn.reshape([batch, num_channels, width, height])

        return attn

    def attention_spatial(self, in_query, in_keys, in_values):
        """
        在空间上的注意力机制
        :param self:
        :param in_query: attention Q
        :param in_keys: attention K
        :param in_values: attention V
        :return: 注意力机制的输出
        """
        q_shape = in_query.shape
        k_shape = in_keys.shape
        batch = q_shape[0]
        num_channels = q_shape[1]
        width = q_shape[2]
        height = q_shape[3]
        length = k_shape[1]
        query = in_query.reshape([batch, num_channels, -1]).permute((0, 2, 1))
        key = in_keys.permute((0, 1, 3, 4, 2)).reshape([batch, -1, num_channels])
        value = in_values.permute((0, 1, 3, 4, 2)).reshape([batch, -1, num_channels])
        attn = torch.matmul(query, key.permute(0, 2, 1))
        attn = torch.nn.Softmax(dim=2)(attn)
        attn = torch.matmul(attn, value)
        attn = attn.reshape([batch, width, height, num_channels]).permute(0, 3, 1, 2)

        return attn

    def forward(self, in_query, in_keys, in_values):

        spatial_attn = self.attention_spatial(in_query, in_keys, in_values)
        channel_attn = self.attention_channel(in_query, in_keys, in_values)

        s_attn = self.s_attn_(spatial_attn + in_query)
        c_attn = self.c_attn_(channel_attn + in_query)

        return s_attn, c_attn


class attn_sum_fusion(nn.Module):

    def __init__(self, image_channel):
        super().__init__()
        self.attn_ = nn.Sequential(
            nn.Conv2d(image_channel, image_channel, kernel_size=1, stride=1, padding=0)
            # nn.LayerNorm([num_hidden, width, width])
            # nn.Dropout2d(p=0.9)
        )

    def forward(self, channel_attn, spatial_attn):
        attn = channel_attn + spatial_attn
        attn = self.attn_(attn)
        return attn


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

