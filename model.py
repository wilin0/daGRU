import torch
from torch import nn
from modules import STGRUCell, attention_spatial, attention_channel, attn_sum_fusion, Inception


class RNN(nn.Module):
    def __init__(self, in_shape, num_layers, num_hidden, time_stride, input_length, filter_size, stride, patch_size, device, num_inception):
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
        self.num_inception = num_inception
        cell_list = []

        self.enc_s = attention_spatial(self.img_channel, num_hidden[0], self.filter_size, self.stride, self.width)
        self.enc_c = attention_channel(self.img_channel, num_hidden[0], self.filter_size, self.stride, self.width)
        self.dec = attn_sum_fusion(self.img_channel, self.img_channel)

        self.merge_t = nn.Conv2d(self.num_hidden[0] * 2, self.num_hidden[0], kernel_size=1, stride=1, padding=0)

        for i in range(num_layers):
            # in_channel = self.img_channel if i == 0 else self.num_hidden[i - 1]
            in_channel = self.num_hidden[0] if i == 0 else self.num_hidden[i - 1]
            cell_list.append(
                STGRUCell(in_channel, self.num_hidden[i], self.width, self.width, filter_size, stride)
            )
        self.cell_list = nn.ModuleList(cell_list)

        # spatial inception
        self.N_T = self.num_inception
        incep_ker = [3, 5, 7, 11]
        groups = 8
        enc_layers_s = [Inception(self.num_hidden[0], self.num_hidden[0] // 2, self.num_hidden[0], incep_ker=incep_ker, groups=groups)]
        for i in range(1, self.N_T - 1):
            enc_layers_s.append(Inception(self.num_hidden[0], self.num_hidden[0] // 2, self.num_hidden[0], incep_ker=incep_ker, groups=groups))
        enc_layers_s.append(Inception(self.num_hidden[0], self.num_hidden[0] // 2, self.num_hidden[0], incep_ker=incep_ker, groups=groups))

        dec_layers_s = [Inception(2 * self.num_hidden[0], self.num_hidden[0] // 2, self.num_hidden[0], incep_ker=incep_ker, groups=groups)]
        for i in range(1, self.N_T - 1):
            dec_layers_s.append(
                Inception(2 * self.num_hidden[0], self.num_hidden[0] // 2, self.num_hidden[0], incep_ker=incep_ker, groups=groups))
        dec_layers_s.append(Inception(2 * self.num_hidden[0], self.num_hidden[0] // 2, self.img_channel, incep_ker=incep_ker, groups=groups))

        self.enc_inception_s = nn.Sequential(*enc_layers_s)
        self.dec_inception_s = nn.Sequential(*dec_layers_s)

        # channel inception
        enc_layers_c = [Inception(self.num_hidden[0], self.num_hidden[0] // 2, self.num_hidden[0], incep_ker=incep_ker,
                                  groups=groups)]
        for i in range(1, self.N_T - 1):
            enc_layers_c.append(
                Inception(self.num_hidden[0], self.num_hidden[0] // 2, self.num_hidden[0], incep_ker=incep_ker,
                          groups=groups))
        enc_layers_c.append(
            Inception(self.num_hidden[0], self.num_hidden[0] // 2, self.num_hidden[0], incep_ker=incep_ker,
                      groups=groups))

        dec_layers_c = [Inception(2 * self.num_hidden[0], self.num_hidden[0] // 2, self.num_hidden[0], incep_ker=incep_ker,
                                  groups=groups)]
        for i in range(1, self.N_T - 1):
            dec_layers_c.append(
                Inception(2 * self.num_hidden[0], self.num_hidden[0] // 2, self.num_hidden[0], incep_ker=incep_ker,
                          groups=groups))
        dec_layers_c.append(
            Inception(2 * self.num_hidden[0], self.num_hidden[0] // 2, self.img_channel, incep_ker=incep_ker,
                      groups=groups))

        self.enc_inception_c = nn.Sequential(*enc_layers_c)
        self.dec_inception_c = nn.Sequential(*dec_layers_c)

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
            # [time, batch, channel, height, width]->[batch, time, channel, height, width]
            input_frm = input_frm.permute(1, 0, 2, 3, 4).contiguous()

            B, T, C, H, W = input_frm.shape
            s_attn = input_frm.reshape(B*T, C, H, W)
            t_attn = s_attn
            # 加入inception encoder
            skip_s = []
            for i in range(self.N_T):
                s_attn = self.enc_inception_s[i](s_attn)
                s_img = s_attn.reshape(B, T, C, H, W)
                skip_s.append(s_img[:, -1])

            skip_t = []
            for i in range(self.N_T):
                t_attn = self.enc_inception_c[i](t_attn)
                t_img = t_attn.reshape(B, T, C, H, W)
                skip_t.append(t_img[:, -1])

            s_attn = self.enc_s(net, input_frm, input_frm)
            t_attn = self.enc_c(net, input_frm, input_frm)

            T_t[0] = self.merge_t(torch.cat([T_t[0], t_attn], dim=1))

            T_t[0], S_t = self.cell_list[0](T_t[0], s_attn)

            for i in range(1, self.num_layers):
                T_t[i], S_t = self.cell_list[i](T_t[i], S_t)

            z_s = S_t
            z_t = T_t[-1]
            # 加入inception decoder和残差
            for i in range(self.N_T):
                z_s = self.dec_inception_s[i](torch.cat([z_s, skip_s[-i]], dim=1))
                z_t = self.dec_inception_c[i](torch.cat([z_t, skip_t[-i]], dim=1))

            attn = self.dec(z_t, z_s)

            # 0-9输入，9-18输出
            if t >= self.input_length - 1:
                next_frames.append(attn)

        next_frames = torch.stack(next_frames, dim=0).permute((1, 0, 2, 3, 4)).contiguous()

        return next_frames

