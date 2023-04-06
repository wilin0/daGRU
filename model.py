import torch
from torch import nn
from modules import STGRUCell, DualAttention, attn_sum_fusion, reshape_patch_back, reshape_patch


class RNN(nn.Module):
    def __init__(self, in_shape, num_layers, num_hidden, time_stride, input_length, filter_size, stride, patch_size):
        super(RNN, self).__init__()

        self.img_channel = in_shape[1] * patch_size * patch_size
        self.total_length = in_shape[0]
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.time_stride = time_stride
        self.input_length = input_length
        self.width = in_shape[2] // patch_size
        self.patch_size = patch_size
        self.filter_size = filter_size
        self.stride = stride
        cell_list = []

        self.enc = DualAttention(self.img_channel, self.filter_size, self.stride, self.width)
        self.dec = attn_sum_fusion(self.img_channel)

        self.merge_t = nn.Conv2d(self.num_hidden[-1] * 2, self.num_hidden[-1], kernel_size=1, stride=1, padding=0)

        for i in range(num_layers):
            in_channel = self.img_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                STGRUCell(in_channel, num_hidden[i], self.width, self.width, filter_size, stride)
            )
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, x):

        x = reshape_patch(x, self.patch_size)
        # [batch, time, height, width, channel] -> [batch, time, channel, height, width]
        frames = x.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        image_list = []
        for time_step in range(self.time_stride):
            image_list.append(torch.zeros([batch, self.img_channel, height, width]))

        next_frames = []
        T_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
            T_t.append(zeros)

        for t in range(self.total_length - 1):

            net = frames[:, t]
            image_list.append(net)
            input_frm = torch.stack(image_list[t:])
            input_frm = input_frm.permute(1, 0, 2, 3, 4).contiguous()

            s_attn, t_attn = self.enc(net, input_frm, input_frm)

            T_t[0] = self.merge_t(torch.cat([T_t[0], t_attn], dim=1))

            T_t[0], S_t = self.cell_list[0](T_t[0], s_attn)

            for i in range(1, self.num_layers):
                T_t[i], S_t = self.cell_list[i](T_t[i - 1], S_t)

            attn = self.dec(T_t[-1], S_t)
            next_frames.append(attn)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        next_frames = (next_frames, self.patch_size)

        return next_frames
