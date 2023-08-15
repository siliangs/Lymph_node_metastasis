import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial

def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False)


def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if not no_cuda:
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop = nn.Dropout(0.4)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.drop(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.drop(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# class Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm3d(planes)
#         self.conv2 = nn.Conv3d(
#             planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
#         self.bn2 = nn.BatchNorm3d(planes)
#         self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm3d(planes * 4)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#         self.dilation = dilation

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)

#         return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_input_D,
                 sample_input_H,
                 sample_input_W,
                 num_seg_classes,
                 shortcut_type='B',
                 no_cuda = False):
        self.inplanes = 64
        self.no_cuda = no_cuda
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(2,64,kernel_size=7,stride=(2, 2, 2),padding=(3, 3, 3),bias=False)
            
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=1, dilation=4)

        # self.conv_seg = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)), 
        #                               nn.Flatten(),
        #                               nn.Linear(512, 128, bias=True),
        #                               nn.Dropout(0.3),
        #                               nn.Linear(128, 1, bias=True),
        #                               )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_feature = self.layer4(x)
        # print('x_feature:', x_feature.shape)
        # x = self.conv_seg(x_feature)

        return x_feature

def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [1, 1, 2, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def generate_model(model_type='resnet', model_depth=50,
                   input_W=224, input_H=224, input_D=224, resnet_shortcut='B',
                   no_cuda=False, gpu_id=[0],
                   nb_class=1):
    assert model_type in [
        'resnet'
    ]

    if model_type == 'resnet':
        assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = resnet10(
            sample_input_W=input_W,
            sample_input_H=input_H,
            sample_input_D=input_D,
            shortcut_type=resnet_shortcut,
            no_cuda=no_cuda,
            num_seg_classes=1)
        fc_input = 256
    elif model_depth == 18:
        model = resnet18(
            sample_input_W=input_W,
            sample_input_H=input_H,
            sample_input_D=input_D,
            shortcut_type=resnet_shortcut,
            no_cuda=no_cuda,
            num_seg_classes=1)
        fc_input = 512
    elif model_depth == 34:
        model = resnet34(
            sample_input_W=input_W,
            sample_input_H=input_H,
            sample_input_D=input_D,
            shortcut_type=resnet_shortcut,
            no_cuda=no_cuda,
            num_seg_classes=1)
        fc_input = 512

    # model.conv_seg = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)), nn.Flatten(),
    #                                nn.Linear(in_features=fc_input, out_features=nb_class, bias=True))


    return model


class model_resnet(nn.Module):
    def __init__(self):
        super(model_resnet, self).__init__()
        self.model = generate_model(model_type='resnet', model_depth=10,
                       input_W=64, input_H=64, input_D=64, resnet_shortcut='B',
                       no_cuda=False, gpu_id=[0],
                       nb_class=1)
        self.avgpool=nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)), 
                                   nn.Flatten()
                                   )
        
        self.out=nn.Sequential(nn.Linear(512 ,128),
                              nn.Dropout(0.4),
                              nn.Linear(128,1),
                              nn.Sigmoid())
        
        self.dia = nn.Sequential(nn.Linear(512, 128),
                                 nn.Dropout(0.4),
                                 nn.Linear(128, 32)
                                 )
        
        self.osmonth = nn.Sequential(nn.Linear(34, 1),
                                     nn.Sigmoid())
    def forward(self, x, doctor_label):
        out1 = self.model(x)
        out1 = self.avgpool(out1)
        out = self.out(out1)
        
        out_c = out.clone()
        out_c[out_c<0.5]=0
        out_c[out_c>=0.5]=1
        
        comper = self.dia(out1)
        
        pre_os = self.osmonth(torch.cat((out_c, comper, out_c), dim=1))
        doc_os = self.osmonth(torch.cat((doctor_label, comper, doctor_label), dim=1))
        # print(out)
        # print(out_c)
        return out, out_c, pre_os, doc_os
    

if __name__=='__main__':
    resnet=model_resnet()
    x=torch.randn(12,2,64,64,64)
    y=torch.randn(12,1)
    out, out_c, pre_os, doc_os=resnet(x, y)
    print(out.shape)
    print(out_c.shape)
    print(pre_os.shape)
    print(doc_os.shape)
