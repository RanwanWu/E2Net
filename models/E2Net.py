import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Res2Net_v1b import res2net50_v1b_26w_4s

########################################   Convlusion layer     ######################################

class Convlayer(nn.Module):
    def __init__(self, in_channal, out_channal):
        super(Convlayer, self).__init__()

        self.conv1 = nn.Conv2d(in_channal, out_channal, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(out_channal, out_channal, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channal)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        y = self.relu(self.conv1(x))
        y = self.relu(self.bn(self.conv3(y)))

        return  y


########################################    Channel-Attention Module     #########################################

class ChannelAttentionModule(nn.Module):

    def __init__(self,in_channal):
        super(ChannelAttentionModule, self).__init__()

        self.stack = nn.Sequential(nn.Conv2d(in_channal, in_channal//4, kernel_size=1),nn.BatchNorm2d(in_channal//4),nn.ReLU(),
                     nn.Conv2d(in_channal//4, in_channal//16, kernel_size=1),nn.BatchNorm2d(in_channal//16),nn.ReLU(),
                     nn.Conv2d(in_channal//16, in_channal//4, kernel_size=1),nn.BatchNorm2d(in_channal//4),nn.ReLU(),
                     nn.Conv2d(in_channal//4, in_channal, kernel_size=1),nn.BatchNorm2d(in_channal),nn.ReLU(),
                                   )
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):

        x = x.mul(self.stack(self.max_pool(x)))

        return  x


########################################     Position-Attention Module     #########################################

class PositionAttentionModule(nn.Module):

    def __init__(self, in_channels):
        super(PositionAttentionModule, self).__init__()

        self.conv_b = nn.Conv2d(in_channels, in_channels//8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels//8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x

        return out


########################################     Joint-Attention     ######################################

class JointAttention(nn.Module):
    def __init__(self, in_channal):
        super(JointAttention, self).__init__()

        self.CA = ChannelAttentionModule(in_channal)
        self.PA = PositionAttentionModule(in_channal)

    def forward(self, x):

        x_c = self.CA(x)
        x_p = self.PA(x)

        return  x_c + x_p


########################################     Enhence Module     ######################################

class EnhenceModule(nn.Module):
    def __init__(self, in_channel):
        super(EnhenceModule, self).__init__()

        self.conv1 = nn.Conv2d(in_channel//4,in_channel//4,kernel_size=1,stride=1,padding=0)
        self.conv3 = nn.Conv2d(in_channel//4,in_channel//4,kernel_size=3,stride=1,padding=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)

        x1 = self.conv3(self.relu(self.conv1(x1)))
        x2 = self.conv3(self.relu(self.conv1(x2)))
        x3 = self.conv3(self.relu(self.conv1(x3)))
        x4 = self.conv3(self.relu(self.conv1(x4)))

        x_1_in = torch.reshape(x1,[x1.shape[0],1,x1.shape[1],x1.shape[2],x1.shape[3]])
        x_2_in = torch.reshape(x2,[x2.shape[0],1,x2.shape[1],x2.shape[2],x2.shape[3]])
        x_3_in = torch.reshape(x3,[x3.shape[0],1,x3.shape[1],x3.shape[2],x3.shape[3]])
        x_4_in = torch.reshape(x4,[x4.shape[0],1,x4.shape[1],x4.shape[2],x4.shape[3]])

        x_out  = torch.cat((x_1_in, x_2_in, x_3_in, x_4_in),dim=1)
        x_out = x_out.max(dim=1)[0]

        return  x_out


########################################          E2Net             ######################################

class E2Net(nn.Module):
    def __init__(self):
        super(E2Net, self).__init__()

        self.resnet_r = res2net50_v1b_26w_4s('rgb', pretrained=True)
        self.resnet_t = res2net50_v1b_26w_4s('rgb', pretrained=True)
        self.conv_in = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0)

        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.JA4 = JointAttention(2048)
        self.JA3 = JointAttention(1024)
        self.JA2 = JointAttention(512)
        self.JA1 = JointAttention(256)
        self.JA0 = JointAttention(64)

        self.EM4 = EnhenceModule(2048)
        self.EM3 = EnhenceModule(1024)
        self.EM2 = EnhenceModule(512)
        self.EM1 = EnhenceModule(256)
        self.EM0 = EnhenceModule(64)

        self.conv4 = Convlayer(512, 256)
        self.conv3 = Convlayer(256, 128)
        self.conv2 = Convlayer(128, 64)
        self.conv1 = Convlayer(64, 16)
        self.conv0 = Convlayer(16, 16)
        self.conv_out = Convlayer(16, 1)

    def forward(self, x_r, x_t):
        x_t = self.resnet_t.conv1(self.conv_in(x_t))
        x_t = self.resnet_t.bn1(x_t)
        x_t = self.resnet_t.relu(x_t)

        x_t_0 = self.resnet_t.maxpool(x_t)
        x_t_1 = self.resnet_t.layer1(x_t_0)
        x_t_2 = self.resnet_t.layer2(x_t_1)
        x_t_3 = self.resnet_t.layer3(x_t_2)
        x_t_4 = self.resnet_t.layer4(x_t_3)

        x_r = self.resnet_r.conv1(x_r)
        x_r = self.resnet_r.bn1(x_r)
        x_r = self.resnet_r.relu(x_r)

        x_r_0 = self.resnet_r.maxpool(x_r)
        x_r_1 = self.resnet_r.layer1(x_r_0)
        x_r_2 = self.resnet_r.layer2(x_r_1)
        x_r_3 = self.resnet_r.layer3(x_r_2)
        x_r_4 = self.resnet_r.layer4(x_r_3)

        pre_4 = self.JA4(x_r_4 + x_t_4)
        pre_3 = self.JA3(x_r_3 + x_t_3)
        pre_2 = self.JA2(x_r_2 + x_t_2)
        pre_1 = self.JA1(x_r_1 + x_t_1)
        pre_0 = self.JA0(x_r_0 + x_t_0)

        pre_4 = self.EM4(pre_4)
        pre_3 = self.EM3(pre_3)
        pre_2 = self.EM2(pre_2)
        pre_1 = self.EM1(pre_1)
        pre_0 = self.EM0(pre_0)

        pre_4 = self.conv4(self.up2(pre_4))

        pre_3 = pre_4 + pre_3
        pre_3 = self.conv3(self.up2(pre_3))

        pre_2 = pre_3 + pre_2
        pre_2 = self.conv2(self.up2(pre_2))

        pre_1 = pre_2 + pre_1
        pre_1 = self.conv1(self.up2(pre_1))

        pre_0 = pre_1 + self.up2(pre_0)
        pre_0 = self.conv0(self.up2(pre_0))
        pre_0 = self.conv_out(pre_0)

        return    pre_0
