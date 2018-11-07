import math
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from resnet3d import *
from layers import conv1x1x1, conv3x3x3
from vgg3d import  *
from vgg3d_p3 import *
from vgg3d_p4 import *

def classification_layer_init(tensor):
    fill_constant = - math.log((1 - pi) / pi)
    if isinstance(tensor, Variable):
        classification_layer_init(tensor.data)
    return tensor.fill_(fill_constant)


class FeaturePyramid_v1(nn.Module):
    def __init__(self, resnet):
        super(FeaturePyramid_v1, self).__init__()

        self.resnet = resnet

        # applied in a pyramid
        self.pyramid_transformation_1 = conv3x3x3(64, 256, padding=1)
        self.pyramid_transformation_2 = conv1x1x1(256, 256)
        self.pyramid_transformation_3 = conv1x1x1(512, 256)
        self.pyramid_transformation_4 = conv1x1x1(1024, 256)
        self.pyramid_transformation_5 = conv1x1x1(2048, 256)

        # both based around resnet_feature_5
        #self.pyramid_transformation_6 = conv3x3x3(2048, 256, padding=1, stride=2)
        #self.pyramid_transformation_7 = conv3x3x3(256, 256, padding=1, stride=2) 

        # applied after upsampling
        self.upsample_transform_1 = conv3x3x3(256, 256, padding=1)
        self.upsample_transform_2 = conv3x3x3(256, 256, padding=1)
        self.upsample_transform_3 = conv3x3x3(256, 256, padding=1)
        self.upsample_transform_4 = conv3x3x3(256, 256, padding=1)

    def _upsample(self, original_feature, scaled_feature, scale_factor=2):
        # is this correct? You do lose information on the upscale...
        depth, height, width = scaled_feature.size()[2:]
        return F.upsample(original_feature, scale_factor=scale_factor)[:, :, :depth, :height, :width]

    def forward(self, x):

        resnet_feature_1, resnet_feature_2, resnet_feature_3, resnet_feature_4, resnet_feature_5 = self.resnet(x)

        # pyramid_feature_6 = self.pyramid_transformation_6(resnet_feature_5)
        # pyramid_feature_7 = self.pyramid_transformation_7(F.relu(pyramid_feature_6))

        pyramid_feature_5 = self.pyramid_transformation_5(resnet_feature_5)     # transform c5 from 2048d to 256d
        pyramid_feature_4 = self.pyramid_transformation_4(resnet_feature_4)     # transform c4 from 1024d to 256d
        upsampled_feature_5 = self._upsample(pyramid_feature_5, pyramid_feature_4)   # deconv c5 to c4.size

        pyramid_feature_4 = self.upsample_transform_4(
            torch.add(upsampled_feature_5, pyramid_feature_4)               # add up-c5 and c4, and conv
        )

        pyramid_feature_3 = self.pyramid_transformation_3(resnet_feature_3)     # transform c3 from 512d to 256d
        upsampled_feature_4 = self._upsample(pyramid_feature_4, pyramid_feature_3)    # deconv c4 to c3.size

        pyramid_feature_3 = self.upsample_transform_3(
            torch.add(upsampled_feature_4, pyramid_feature_3)               # add up-c4 and c3, and conv
        )
        
        pyramid_feature_2 = self.pyramid_transformation_2(resnet_feature_2)                                # c2 is 256d, so no need to transform
        upsampled_feature_3 = self._upsample(pyramid_feature_3, pyramid_feature_2)    # deconv c3 to c2.size

        pyramid_feature_2 = self.upsample_transform_2(
            torch.add(upsampled_feature_3, pyramid_feature_2)              # add up-c3 and c2, and conv
        )
        
        pyramid_feature_1 = self.pyramid_transformation_1(resnet_feature_1)  # use conv3x3x3 up c1 from 64d to 256d
        upsampled_feature_2 = self._upsample(pyramid_feature_2, pyramid_feature_1)    # deconv c2 to c1.size

        pyramid_feature_1 = self.upsample_transform_1(
            torch.add(upsampled_feature_2, pyramid_feature_1)             # add up-c2 and c1, and conv
        )
        


        return (pyramid_feature_1,             # 8
                pyramid_feature_2,             # 16
                pyramid_feature_3)             # 32


class FeaturePyramid_v2(nn.Module):
    def __init__(self, resnet):
        super(FeaturePyramid_v2, self).__init__()

        self.resnet = resnet

        # applied in a pyramid
        self.pyramid_transformation_1 = conv3x3x3(64, 256, padding=1)
        self.pyramid_transformation_2 = conv3x3x3(128, 256, padding=1)
        self.pyramid_transformation_3 = conv3x3x3(256, 256, padding=1)
        self.pyramid_transformation_4 = conv1x1x1(512, 256)
        self.pyramid_transformation_5 = conv1x1x1(512, 256)

        # both based around resnet_feature_5
        #self.pyramid_transformation_6 = conv3x3x3(2048, 256, padding=1, stride=2)
        #self.pyramid_transformation_7 = conv3x3x3(256, 256, padding=1, stride=2) 

        # applied after upsampling
        self.upsample_transform_1 = conv3x3x3(256, 256, padding=1)
        self.upsample_transform_2 = conv3x3x3(256, 256, padding=1)
        self.upsample_transform_3 = conv3x3x3(256, 256, padding=1)
        self.upsample_transform_4 = conv3x3x3(256, 256, padding=1)

    def _upsample(self, original_feature, scaled_feature, scale_factor=2):
        # is this correct? You do lose information on the upscale...
        depth, height, width = scaled_feature.size()[2:]
        return F.upsample(original_feature, scale_factor=scale_factor)[:, :, :depth, :height, :width]

    def forward(self, x):

        resnet_feature_1, resnet_feature_2, resnet_feature_3, resnet_feature_4, resnet_feature_5 = self.resnet(x)

        #pyramid_feature_6 = self.pyramid_transformation_6(resnet_feature_5)
        # pyramid_feature_7 = self.pyramid_transformation_7(F.relu(pyramid_feature_6))
        
        pyramid_feature_5 = self.pyramid_transformation_5(resnet_feature_5)     # transform c5 from 2048d to 256d
        pyramid_feature_4 = self.pyramid_transformation_4(resnet_feature_4)     # transform c4 from 1024d to 256d
        upsampled_feature_5 = self._upsample(pyramid_feature_5, pyramid_feature_4)   # deconv c5 to c4.size

        pyramid_feature_4 = self.upsample_transform_4(
           torch.add(upsampled_feature_5, pyramid_feature_4)               # add up-c5 and c4, and conv
        )
        
        pyramid_feature_3 = self.pyramid_transformation_3(resnet_feature_3)     # transform c3 from 512d to 256d
        upsampled_feature_4 = self._upsample(pyramid_feature_4, pyramid_feature_3)    # deconv c4 to c3.size
        
        pyramid_feature_3 = self.upsample_transform_3(
            torch.add(upsampled_feature_4, pyramid_feature_3)               # add up-c4 and c3, and conv
        )
        
        pyramid_feature_2 = self.pyramid_transformation_2(resnet_feature_2)                                # c2 is 256d, so no need to transform
        upsampled_feature_3 = self._upsample(pyramid_feature_3, pyramid_feature_2)    # deconv c3 to c2.size

        pyramid_feature_2 = self.upsample_transform_2(
            torch.add(upsampled_feature_3, pyramid_feature_2)              # add up-c3 and c2, and conv
        )
        
        pyramid_feature_1 = self.pyramid_transformation_1(resnet_feature_1)  # use conv3x3x3 up c1 from 64d to 256d
        upsampled_feature_2 = self._upsample(pyramid_feature_2, pyramid_feature_1)    # deconv c2 to c1.size

        pyramid_feature_1 = self.upsample_transform_1(
            torch.add(upsampled_feature_2, pyramid_feature_1)             # add up-c2 and c1, and conv
        )
        


        return (pyramid_feature_1,             # 8
                pyramid_feature_2,             # 16
                pyramid_feature_3)             # 32

class FeaturePyramid_v3(nn.Module):
    def __init__(self, resnet):
        super(FeaturePyramid_v3, self).__init__()

        self.resnet = resnet

        # applied in a pyramid
        self.pyramid_transformation_1 = conv3x3x3(64, 256, padding=1)
        self.pyramid_transformation_2 = conv3x3x3(64, 256, padding=1)
        self.pyramid_transformation_3 = conv3x3x3(128, 256, padding=1)
        self.pyramid_transformation_4 = conv1x1x1(256, 256)
        self.pyramid_transformation_5 = conv1x1x1(512, 256)

        # both based around resnet_feature_5
        #self.pyramid_transformation_6 = conv3x3x3(2048, 256, padding=1, stride=2)
        #self.pyramid_transformation_7 = conv3x3x3(256, 256, padding=1, stride=2) 

        # applied after upsampling
        self.upsample_transform_1 = conv3x3x3(256, 256, padding=1)
        self.upsample_transform_2 = conv3x3x3(256, 256, padding=1)
        self.upsample_transform_3 = conv3x3x3(256, 256, padding=1)
        self.upsample_transform_4 = conv3x3x3(256, 256, padding=1)

    def _upsample(self, original_feature, scaled_feature, scale_factor=2):
        # is this correct? You do lose information on the upscale...
        depth, height, width = scaled_feature.size()[2:]
        return F.upsample(original_feature, scale_factor=scale_factor)[:, :, :depth, :height, :width]

    def forward(self, x):

        resnet_feature_1, resnet_feature_2, resnet_feature_3, resnet_feature_4, resnet_feature_5 = self.resnet(x)

        # pyramid_feature_6 = self.pyramid_transformation_6(resnet_feature_5)
        # pyramid_feature_7 = self.pyramid_transformation_7(F.relu(pyramid_feature_6))

        pyramid_feature_5 = self.pyramid_transformation_5(resnet_feature_5)     # transform c5 from 2048d to 256d
        pyramid_feature_4 = self.pyramid_transformation_4(resnet_feature_4)     # transform c4 from 1024d to 256d
        upsampled_feature_5 = self._upsample(pyramid_feature_5, pyramid_feature_4)   # deconv c5 to c4.size

        pyramid_feature_4 = self.upsample_transform_4(
            torch.add(upsampled_feature_5, pyramid_feature_4)               # add up-c5 and c4, and conv
        )

        pyramid_feature_3 = self.pyramid_transformation_3(resnet_feature_3)     # transform c3 from 512d to 256d
        upsampled_feature_4 = self._upsample(pyramid_feature_4, pyramid_feature_3)    # deconv c4 to c3.size

        pyramid_feature_3 = self.upsample_transform_3(
            torch.add(upsampled_feature_4, pyramid_feature_3)               # add up-c4 and c3, and conv
        )
        
        pyramid_feature_2 = self.pyramid_transformation_2(resnet_feature_2)                                # c2 is 256d, so no need to transform
        upsampled_feature_3 = self._upsample(pyramid_feature_3, pyramid_feature_2)    # deconv c3 to c2.size

        pyramid_feature_2 = self.upsample_transform_2(
            torch.add(upsampled_feature_3, pyramid_feature_2)              # add up-c3 and c2, and conv
        )
        
        pyramid_feature_1 = self.pyramid_transformation_1(resnet_feature_1)  # use conv3x3x3 up c1 from 64d to 256d
        upsampled_feature_2 = self._upsample(pyramid_feature_2, pyramid_feature_1)    # deconv c2 to c1.size

        pyramid_feature_1 = self.upsample_transform_1(
            torch.add(upsampled_feature_2, pyramid_feature_1)             # add up-c2 and c1, and conv
        )
        


        return (pyramid_feature_1,             # 8
                pyramid_feature_2,             # 16
                pyramid_feature_3)             # 32


class FeaturePyramid_v4(nn.Module):
    def __init__(self, resnet):
        super(FeaturePyramid_v4, self).__init__()

        self.resnet = resnet

        # applied in a pyramid
        self.pyramid_transformation_1 = conv3x3x3(64, 256, padding=1)
        self.pyramid_transformation_2 = conv3x3x3(128, 256, padding=1)
        self.pyramid_transformation_3 = conv3x3x3(256, 256, padding=1)
        #self.pyramid_transformation_4 = conv1x1x1(512, 256)
        #self.pyramid_transformation_5 = conv1x1x1(512, 256)

        # both based around resnet_feature_5
        #self.pyramid_transformation_6 = conv3x3x3(2048, 256, padding=1, stride=2)
        #self.pyramid_transformation_7 = conv3x3x3(256, 256, padding=1, stride=2) 

        # applied after upsampling
        self.upsample_transform_1 = conv3x3x3(256, 256, padding=1)
        self.upsample_transform_2 = conv3x3x3(256, 256, padding=1)
        self.upsample_transform_3 = conv3x3x3(256, 256, padding=1)
        #self.upsample_transform_4 = conv3x3x3(256, 256, padding=1)

    def _upsample(self, original_feature, scaled_feature, scale_factor=2):
        # is this correct? You do lose information on the upscale...
        depth, height, width = scaled_feature.size()[2:]
        return F.upsample(original_feature, scale_factor=scale_factor)[:, :, :depth, :height, :width]

    def forward(self, x):

        resnet_feature_1, resnet_feature_2, resnet_feature_3 = self.resnet(x)

        #pyramid_feature_6 = self.pyramid_transformation_6(resnet_feature_5)
        # pyramid_feature_7 = self.pyramid_transformation_7(F.relu(pyramid_feature_6))
        '''
        pyramid_feature_5 = self.pyramid_transformation_5(resnet_feature_5)     # transform c5 from 2048d to 256d
        pyramid_feature_4 = self.pyramid_transformation_4(resnet_feature_4)     # transform c4 from 1024d to 256d
        upsampled_feature_5 = self._upsample(pyramid_feature_5, pyramid_feature_4)   # deconv c5 to c4.size

        pyramid_feature_4 = self.upsample_transform_4(
           torch.add(upsampled_feature_5, pyramid_feature_4)               # add up-c5 and c4, and conv
        )
        '''
        pyramid_feature_3 = self.pyramid_transformation_3(resnet_feature_3)     # transform c3 from 512d to 256d
        #upsampled_feature_4 = self._upsample(pyramid_feature_4, pyramid_feature_3)    # deconv c4 to c3.size
        '''
        pyramid_feature_3 = self.upsample_transform_3(
            torch.add(upsampled_feature_4, pyramid_feature_3)               # add up-c4 and c3, and conv
        )
        '''
        pyramid_feature_2 = self.pyramid_transformation_2(resnet_feature_2)                                # c2 is 256d, so no need to transform
        upsampled_feature_3 = self._upsample(pyramid_feature_3, pyramid_feature_2)    # deconv c3 to c2.size

        pyramid_feature_2 = self.upsample_transform_2(
            torch.add(upsampled_feature_3, pyramid_feature_2)              # add up-c3 and c2, and conv
        )
        
        pyramid_feature_1 = self.pyramid_transformation_1(resnet_feature_1)  # use conv3x3x3 up c1 from 64d to 256d
        upsampled_feature_2 = self._upsample(pyramid_feature_2, pyramid_feature_1)    # deconv c2 to c1.size

        pyramid_feature_1 = self.upsample_transform_1(
            torch.add(upsampled_feature_2, pyramid_feature_1)             # add up-c2 and c1, and conv
        )
        


        return (pyramid_feature_1,             # 8
                pyramid_feature_2,             # 16
                pyramid_feature_3)             # 32

class SubNet(nn.Module):
    def __init__(self, k, anchors=9, depth=4, activation=F.relu):
        super(SubNet, self).__init__()
        self.anchors = anchors
        self.activation = activation
        self.base = nn.ModuleList([conv3x3x3(256, 256, padding=1) for _ in range(depth)])
        self.output = nn.Conv3d(256, k * anchors, kernel_size=3, padding=1)
        self.output.weight = nn.init.xavier_normal(self.output.weight)

    def forward(self, x):
        for layer in self.base:
            x = self.activation(layer(x))
        x = self.output(x)      # (1, 54 or 36, 16, 16, 16)
        x = x.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), 
                                                       x.size(2) * x.size(3) * x.size(4) * self.anchors, 
                                                       -1)
        # (1, 16*16*16*9, 6 or 4)
        return x


class RetinaNet(nn.Module):
    backbones = {
        'vgg9': vgg9,
        'vgg16': vgg16,
        'resnet18': resnet18,
        'resnet34': resnet34,
        'resnet50': resnet50,
        'resnet101': resnet101,
        'resnet152': resnet152
    }

    def __init__(self, backbone='resnet50', num_classes=3, pretrained=False):
        super(RetinaNet, self).__init__()
        self.backbone_net = RetinaNet.backbones[backbone](pretrained=pretrained)	
        if backbone == 'resnet50' or backbone == 'resnet101' or backbone == 'resnet152':
            self.feature_pyramid = FeaturePyramid_v1(self.backbone_net)
        elif backbone == "vgg16":
            self.feature_pyramid = FeaturePyramid_v2(self.backbone_net)
        elif backbone == "vgg9":
            self.feature_pyramid = FeaturePyramid_v4(self.backbone_net)
        else:
            self.feature_pyramid = FeaturePyramid_v3(self.backbone_net)
        self.subnet_boxes = SubNet(6)  #(tz, ty, tx, td, th ,tw)
        self.subnet_classes = SubNet(num_classes+1)

    def forward(self, x):
        pyramid_features = self.feature_pyramid(x)
        class_predictions = [self.subnet_classes(p) for p in pyramid_features]
        bbox_predictions = [self.subnet_boxes(p) for p in pyramid_features]
        return torch.cat(bbox_predictions, 1), torch.cat(class_predictions, 1)


if __name__ == '__main__':
    net = RetinaNet().cuda()
    x = Variable(torch.rand(1, 1, 64, 64, 64).cuda())
    for l in net(x):
        print(l.size(), type(l))
