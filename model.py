import torch
from torch import nn
from modules import STGRUCell, DualAttention, attn_sum_fusion


class RNN(nn.Module):
    def __init__(self, in_shape, num_layers, num_hidden, time_stride, input_length, filter_size, stride, patch_size, device):
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
        self.device = device
        cell_list = []

        self.enc = DualAttention(self.img_channel, num_hidden[0], self.filter_size, self.stride, self.width)
        self.dec = attn_sum_fusion(self.num_hidden[-1], self.img_channel)

        self.merge_t = nn.Conv2d(self.num_hidden[0] * 2, self.num_hidden[0], kernel_size=1, stride=1, padding=0)

        for i in range(num_layers):
            # in_channel = self.img_channel if i == 0 else self.num_hidden[i - 1]
            in_channel = self.num_hidden[0] if i == 0 else self.num_hidden[i - 1]
            cell_list.append(
                STGRUCell(in_channel, self.num_hidden[i], self.width, self.width, filter_size, stride)
            )
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, x):

        # [batch, time, channel, height, width]
        # [batch, time, height, width, channel] -> [batch, time, channel, height, width] [1 10 1 64 64]
        # frames = x.permute(0, 1, 4, 2, 3).contiguous()
        frames = x.contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        image_list = []
        for time_step in range(self.time_stride):
            image_list.append(torch.zeros([batch, self.img_channel, height, width]).to(self.device))

        next_frames = []
        T_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.device)
            T_t.append(zeros)

        for t in range(self.total_length - 1):
            if t < self.input_length:
                net = frames[:, t]
            else:
                net = attn
            image_list.append(net)
            input_frm = torch.stack(image_list[t:])
            input_frm = input_frm.permute(1, 0, 2, 3, 4).contiguous()

            s_attn, t_attn = self.enc(net, input_frm, input_frm)

            T_t[0] = self.merge_t(torch.cat([T_t[0], t_attn], dim=1))

            T_t[0], S_t = self.cell_list[0](T_t[0], s_attn)

            for i in range(1, self.num_layers):
                T_t[i], S_t = self.cell_list[i](T_t[i], S_t)

            attn = self.dec(T_t[-1], S_t)

            # 0-9输入，9-18输出
            if t >= self.input_length - 1:
                next_frames.append(attn)

        next_frames = torch.stack(next_frames, dim=0).permute((1, 0, 2, 3, 4)).contiguous()

        return next_frames

