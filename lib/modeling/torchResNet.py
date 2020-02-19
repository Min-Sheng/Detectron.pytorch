import torch
import torch.nn as nn
import math
import copy
from collections import OrderedDict
import torch.utils.model_zoo as model_zoo
from core.config import cfg
import utils.net as net_utils

model_urls = {
    'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
    'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
    }

# ---------------------------------------------------------------------------- #
# Helper functions
# ---------------------------------------------------------------------------- #

def weight_mapping(state_dict):
    state_dict_v2 = copy.deepcopy(state_dict)
    layer0_mapping = {
        'conv1.weight': 'res1.conv1.weight',
        'bn1.weight': 'res1.bn1.weight',
        'bn1.bias': 'res1.bn1.bias',
        'bn1.running_mean': 'res1.bn1.running_mean',
        'bn1.running_var': 'res1.bn1.running_var'
    }
    for key in state_dict:
        if key in layer0_mapping.keys():
            new_key = layer0_mapping[key]
            state_dict_v2[new_key] = state_dict_v2.pop(key)
        if key.find('layer') != -1:
            layer_id = int(key[key.find('layer') + 5])
            new_key = key.replace(f'layer{layer_id}', f'res{layer_id+1}')
            state_dict_v2[new_key] = state_dict_v2.pop(key)

    return state_dict_v2

# ---------------------------------------------------------------------------- #
# Bits for specific architectures (ResNet50, ResNet101, ...)
# ---------------------------------------------------------------------------- #

def ResNet50_conv4_body(pretrained=True, model_path=None):
    """Constructs a ResNet-50 model.
    Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model_path = cfg.RESNETS.IMAGENET_PRETRAINED_WEIGHTS if model_path is None else model_path
    model = ResNet_convX_body((3, 4, 6))
    if pretrained:
        if model_path:
            print("Loading pretrained weights from %s" %(model_path))
            state_dict = torch.load(model_path)
            state_dict = state_dict['state_dict']
            state_dict_v2 = copy.deepcopy(state_dict)

            for key in state_dict:
                pre, post = key.split('module.')
                state_dict_v2[post] = state_dict_v2.pop(key)
            state_dict_v2 = weight_mapping(state_dict_v2)
        else:
            state_dict = model_zoo.load_url(model_urls['resnet50'])
            state_dict_v2 = weight_mapping(state_dict)
        model.load_state_dict(state_dict_v2, strict=False)
    return model

def ResNet50_conv5_body(pretrained=True, model_path=None):
    """Constructs a ResNet-50 model.
      Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model_path = cfg.RESNETS.IMAGENET_PRETRAINED_WEIGHTS if model_path is None else model_path
    model = ResNet_convX_body((3, 4, 6, 3))
    if pretrained:
        if model_path:
            print("Loading pretrained weights from %s" %(model_path))
            state_dict = torch.load(model_path)
            state_dict = state_dict['state_dict']
            state_dict_v2 = copy.deepcopy(state_dict)

            for key in state_dict:
                pre, post = key.split('module.')
                state_dict_v2[post] = state_dict_v2.pop(key)
            state_dict_v2 = weight_mapping(state_dict_v2)
        else:
            state_dict = model_zoo.load_url(model_urls['resnet50'])
            state_dict_v2 = weight_mapping(state_dict)
        model.load_state_dict(state_dict_v2, strict=False)
    return model

def ResNet101_conv4_body(pretrained=True, model_path = None):
    """Constructs a ResNet-101 model.
    Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model_path = cfg.RESNETS.IMAGENET_PRETRAINED_WEIGHTS if model_path is None else model_path
    model = ResNet_convX_body((3, 4, 23))
    if pretrained:
        if model_path:
            print("Loading pretrained weights from %s" %(model_path))
            state_dict = torch.load(model_path)
            state_dict = state_dict['state_dict']
            state_dict_v2 = copy.deepcopy(state_dict)

            for key in state_dict:
                pre, post = key.split('module.')
                state_dict_v2[post] = state_dict_v2.pop(key)
            state_dict_v2 = weight_mapping(state_dict_v2)
        else:
            state_dict = model_zoo.load_url(model_urls['resnet101'])
            state_dict_v2 = weight_mapping(state_dict)
        model.load_state_dict(state_dict_v2, strict=False)
    return model


def ResNet101_conv5_body(pretrained=True, model_path = None):
    """Constructs a ResNet-101 model.
    Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model_path = cfg.RESNETS.IMAGENET_PRETRAINED_WEIGHTS if model_path is None else model_path
    model = ResNet_convX_body((3, 4, 23, 3))
    if pretrained:
        if model_path:
            print("Loading pretrained weights from %s" %(model_path))
            state_dict = torch.load(model_path)
            state_dict = state_dict['state_dict']
            state_dict_v2 = copy.deepcopy(state_dict)

            for key in state_dict:
                pre, post = key.split('module.')
                state_dict_v2[post] = state_dict_v2.pop(key)
            state_dict_v2 = weight_mapping(state_dict_v2)
        else:
            state_dict = model_zoo.load_url(model_urls['resnet101'])
            state_dict_v2 = weight_mapping(state_dict)
        model.load_state_dict(state_dict_v2, strict=False)
    return model

def ResNet152_conv5_body(pretrained=True, model_path=None):
    """Constructs a ResNet-152 model.
    Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model_path = cfg.RESNETS.IMAGENET_PRETRAINED_WEIGHTS if model_path is None else model_path
    model = ResNet_convX_body((3, 8, 36, 3))
    if pretrained:
        if model_path:
            print("Loading pretrained weights from %s" %(model_path))
            state_dict = torch.load(model_path)
            state_dict = state_dict['state_dict']
            state_dict_v2 = copy.deepcopy(state_dict)

            for key in state_dict:
                pre, post = key.split('module.')
                state_dict_v2[post] = state_dict_v2.pop(key)
            state_dict_v2 = weight_mapping(state_dict_v2)
        else:
            state_dict = model_zoo.load_url(model_urls['resnet152'])
            state_dict_v2 = weight_mapping(state_dict)
        model.load_state_dict(state_dict_v2, strict=False)
    return model

# ---------------------------------------------------------------------------- #
# Generic ResNet components
# ---------------------------------------------------------------------------- #

class ResNet_convX_body(nn.Module):
    def __init__(self, block_counts):
        super().__init__()
        self.block_counts = block_counts
        self.convX = len(block_counts) + 1
        self.num_layers = (sum(block_counts) + 3 * (self.convX == 4)) * 3 + 2

        self.res1 = globals()[cfg.RESNETS.STEM_FUNC]()
        dim_in = 64
        dim_bottleneck = cfg.RESNETS.NUM_GROUPS * cfg.RESNETS.WIDTH_PER_GROUP #64
        self.res2, dim_in = add_stage(dim_in, 256, dim_bottleneck, block_counts[0],
                                      dilation=1, stride_init=1)
        self.res3, dim_in = add_stage(dim_in, 512, dim_bottleneck * 2, block_counts[1],
                                      dilation=1, stride_init=2)
        self.res4, dim_in = add_stage(dim_in, 1024, dim_bottleneck * 4, block_counts[2],
                                      dilation=1, stride_init=2)
        if len(block_counts) == 4:
            stride_init = 2 if cfg.RESNETS.RES5_DILATION == 1 else 1
            self.res5, dim_in = add_stage(dim_in, 2048, dim_bottleneck * 8, block_counts[3],
                                          cfg.RESNETS.RES5_DILATION, stride_init)
            self.spatial_scale = 1 / 32 * cfg.RESNETS.RES5_DILATION
        else:
            self.spatial_scale = 1 / 16  # final feature scale wrt. original image scale

        self.dim_out = dim_in

        # Initial weights
        self.apply(self._init_weights)

        self._init_modules()

    def _init_weights(self, m):
        classname = m.__class__.__name__ 
        if classname.find('Conv') != -1:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    
    def _init_modules(self):
        assert cfg.RESNETS.FREEZE_AT in [0, 2, 3, 4, 5]
        assert cfg.RESNETS.FREEZE_AT <= self.convX
        for i in range(1, cfg.RESNETS.FREEZE_AT + 1):
            freeze_params(getattr(self, 'res%d' % i))
        
        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad=False
        
        # Freeze all bn layers !!!
        self.apply(set_bn_fix)

    def train(self, mode=True):
        # Override
        self.training = mode

        for i in range(cfg.RESNETS.FREEZE_AT + 1, self.convX + 1):
            getattr(self, 'res%d' % i).train(mode)
        
        def set_bn_eval(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()
        # Set all bn layers to eval
        self.apply(set_bn_eval)

    def forward(self, x):
        for i in range(self.convX):
            x = getattr(self, 'res%d' % (i + 1))(x)
        return x

class ResNet_roi_conv5_head(nn.Module):
    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale

        dim_bottleneck = cfg.RESNETS.NUM_GROUPS * cfg.RESNETS.WIDTH_PER_GROUP
        stride_init = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION // 7
        self.res5, self.dim_out = add_stage(dim_in, 2048, dim_bottleneck * 8, 3,
                                            dilation=1, stride_init=stride_init)
        self.avgpool = nn.AvgPool2d(7)

        self._init_modules()

    def _init_modules(self):
        # Freeze all bn layers !!!
        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad=False
        # Freeze all bn layers !!!
        self.apply(set_bn_fix)
    
    def forward(self, x, rpn_ret):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='rois',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )
        res5_feat = self.res5(x)
        x = self.avgpool(res5_feat)
        if cfg.MODEL.SHARE_RES5 and self.training:
            return x, res5_feat
        else:
            return x

class ResNet_conv5_head(nn.Module):
    def __init__(self, dim_in):
        super().__init__()

        dim_bottleneck = cfg.RESNETS.NUM_GROUPS * cfg.RESNETS.WIDTH_PER_GROUP
        stride_init = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION // 7
        self.res5, self.dim_out = add_stage(dim_in, 2048, dim_bottleneck * 8, 3,
                                            dilation=1, stride_init=stride_init)
        self.avgpool = nn.AvgPool2d(7)

        self._init_modules()

    def _init_modules(self):
        # Freeze all bn layers !!!
        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad=False
        # Freeze all bn layers !!!
        self.apply(set_bn_fix)
    
    def forward(self, x):
        res5_feat = self.res5(x)
        x = self.avgpool(res5_feat)
        return x

def add_stage(inplanes, outplanes, innerplanes, nblocks, dilation=1, stride_init=2):
    """Make a stage consist of `nblocks` residual blocks.
    Returns:
        - stage module: an nn.Sequentail module of residual blocks
        - final output dimension
    """
    res_blocks = []
    stride = stride_init
    for _ in range(nblocks):
        res_blocks.append(add_residual_block(
            inplanes, outplanes, innerplanes, dilation, stride
        ))
        inplanes = outplanes
        stride = 1

    return nn.Sequential(*res_blocks), outplanes


def add_residual_block(inplanes, outplanes, innerplanes, dilation, stride):
    """Return a residual block module, including residual connection, """
    if stride != 1 or inplanes != outplanes:
        shortcut_func = globals()[cfg.RESNETS.SHORTCUT_FUNC]
        downsample = shortcut_func(inplanes, outplanes, stride)
    else:
        downsample = None

    trans_func = globals()[cfg.RESNETS.TRANS_FUNC]
    res_block = trans_func(
        inplanes, outplanes, innerplanes, stride,
        dilation=dilation, group=cfg.RESNETS.NUM_GROUPS,
        downsample=downsample)

    return res_block

# ------------------------------------------------------------------------------
# various downsample shortcuts (may expand and may consider a new helper)
# ------------------------------------------------------------------------------

def basic_bn_shortcut(inplanes, outplanes, stride):
    return nn.Sequential(
        nn.Conv2d(inplanes,
                  outplanes,
                  kernel_size=1,
                  stride=stride,
                  bias=False),
        nn.BatchNorm2d(outplanes),
    )


def basic_gn_shortcut(inplanes, outplanes, stride):
    return nn.Sequential(
        nn.Conv2d(inplanes,
                  outplanes,
                  kernel_size=1,
                  stride=stride,
                  bias=False),
        nn.GroupNorm(net_utils.get_group_gn(outplanes), outplanes,
                     eps=cfg.GROUP_NORM.EPSILON)
    )


# ------------------------------------------------------------------------------
# various stems (may expand and may consider a new helper)
# ------------------------------------------------------------------------------

def basic_bn_stem():
    return nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)),
        ('bn1', nn.BatchNorm2d(64)),
        ('relu', nn.ReLU(inplace=True)),
        ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True))]))
        #('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]))


def basic_gn_stem():
    return nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)),
        ('gn1', nn.GroupNorm(net_utils.get_group_gn(64), 64,
                             eps=cfg.GROUP_NORM.EPSILON)),
        ('relu', nn.ReLU(inplace=True)),
        ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]))

# ------------------------------------------------------------------------------
# various transformations (may expand and may consider a new helper)
# ------------------------------------------------------------------------------

class bottleneck_transformation(nn.Module):
    """ Bottleneck Residual Block """

    def __init__(self, inplanes, outplanes, innerplanes, stride=1, dilation=1, group=1,
                 downsample=None):
        super().__init__()
        # In original resnet, stride=2 is on 1x1.
        # In fb.torch resnet, stride=2 is on 3x3.
        (str1x1, str3x3) = (stride, 1) if cfg.RESNETS.STRIDE_1X1 else (1, stride)
        self.stride = stride

        self.conv1 = nn.Conv2d(
            inplanes, innerplanes, kernel_size=1, stride=str1x1, bias=False)
        self.bn1 = nn.BatchNorm2d(innerplanes)

        self.conv2 = nn.Conv2d(
            innerplanes, innerplanes, kernel_size=3, stride=str3x3, bias=False,
            padding=1 * dilation, dilation=dilation, groups=group)
        self.bn2 = nn.BatchNorm2d(innerplanes)

        self.conv3 = nn.Conv2d(
            innerplanes, outplanes, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class bottleneck_gn_transformation(nn.Module):
    expansion = 4

    def __init__(self, inplanes, outplanes, innerplanes, stride=1, dilation=1, group=1,
                 downsample=None):
        super().__init__()
        # In original resnet, stride=2 is on 1x1.
        # In fb.torch resnet, stride=2 is on 3x3.
        (str1x1, str3x3) = (stride, 1) if cfg.RESNETS.STRIDE_1X1 else (1, stride)
        self.stride = stride

        self.conv1 = nn.Conv2d(
            inplanes, innerplanes, kernel_size=1, stride=str1x1, bias=False)
        self.gn1 = nn.GroupNorm(net_utils.get_group_gn(innerplanes), innerplanes,
                                eps=cfg.GROUP_NORM.EPSILON)

        self.conv2 = nn.Conv2d(
            innerplanes, innerplanes, kernel_size=3, stride=str3x3, bias=False,
            padding=1 * dilation, dilation=dilation, groups=group)
        self.gn2 = nn.GroupNorm(net_utils.get_group_gn(innerplanes), innerplanes,
                                eps=cfg.GROUP_NORM.EPSILON)

        self.conv3 = nn.Conv2d(
            innerplanes, outplanes, kernel_size=1, stride=1, bias=False)
        self.gn3 = nn.GroupNorm(net_utils.get_group_gn(outplanes), outplanes,
                                eps=cfg.GROUP_NORM.EPSILON)

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.gn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def freeze_params(m):
    """Freeze all the weights by setting requires_grad to False
    """
    for p in m.parameters():
        p.requires_grad = False