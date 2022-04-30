import torch
import torch.nn as nn
from torchvision import models, transforms
import timm
import urllib
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import os

#Model Architechture

class MLP(nn.Module):

    # define model elements
    def __init__(self, size):
        super(MLP, self).__init__()
        
        self.model = nn.Sequential(nn.Linear(size[0], size[1]), 
                                   nn.BatchNorm1d(size[1]),
                                   nn.LeakyReLU(),
                                   nn.Dropout(0.1),
                                   nn.Linear(size[1], size[2]), 
                                   nn.BatchNorm1d(size[2]),
                                   nn.LeakyReLU(),
                                   nn.Dropout(0.1),
                                   nn.Linear(size[2], size[3]),
                                   nn.BatchNorm1d(size[3]),
                                   nn.LeakyReLU(),
                                   nn.Dropout(0.1),
                                   nn.Linear(size[3], size[4]),  
                                   nn.BatchNorm1d(size[4]),                                 
                                   nn.LeakyReLU(),
                                   nn.Dropout(0.1),
                                   nn.Linear(size[4], size[5]))
        self.model2 = nn.Sequential(nn.BatchNorm1d(size[5]),
                                   nn.LeakyReLU(),
                                   nn.Dropout(0.1),
                                   nn.Linear(size[5], size[6]))
        
    def forward(self, x, eval = True):
          # Model forward pass
          logits=self.model(x)
          logits2 = self.model2(logits)
          if eval:
            return logits, logits2
          return logits2

class Darknet():
    def __init__(self, darknet_model_file, mlp_model_file, args, eval=True, share_memory=False):

        # self.optimizer = optimizer
        self.model = timm.create_model('cspdarknet53', pretrained=True)
        self.config = resolve_data_config({}, model=self.model)
        self.transform = create_transform(**self.config)

        net = torch.load(os.path.join(darknet_model_file))
        self.model.load_state_dict(net['model_state_dict'])

        self.mlp_model = MLP([2000,4096,4096,2048,1024,512, 12])
        mlp_net = torch.load(os.path.join(mlp_model_file))
        self.mlp_model.load_state_dict(mlp_net['model_state_dict'])
        # self.optimizer.load_state_dict(net['optimizer_state_dict'])

        if args.gpu:
            self.model = self.model.to(torch.device('cuda'))
            self.mlp_model = self.mlp_model.to(torch.device('cuda')) 

        if eval:
            self.model = self.model.eval()
            self.mlp_model = self.mlp_model.eval()

        if share_memory:
            self.model.share_memory()
            self.mlp_model.share_memory()

    def extract(self, x):
        inp = self.transform(x.convert('RGB'))
        inp = torch.unsqueeze(inp,axis=0)
        out = self.model(inp.cuda())
        out = self.mlp_model(out)
        return out[0]


class Resnet18(object):
    '''
    pretrained Resnet18 from torchvision
    '''

    def __init__(self, args, eval=True, share_memory=False, use_conv_feat=True):
        self.model = models.resnet18(pretrained=True)

        if args.gpu:
            self.model = self.model.to(torch.device('cuda'))

        if eval:
            self.model = self.model.eval()

        if share_memory:
            self.model.share_memory()

        if use_conv_feat:
            self.model = nn.Sequential(*list(self.model.children())[:-2])

    def extract(self, x):
        return self.model(x)


class MaskRCNN(object):
    '''
    pretrained MaskRCNN from torchvision
    '''

    def __init__(self, args, eval=True, share_memory=False, min_size=224):
        self.model = models.detection.maskrcnn_resnet50_fpn(pretrained=True, min_size=min_size)
        self.model = self.model.backbone.body
        self.feat_layer = 3

        if args.gpu:
            self.model = self.model.to(torch.device('cuda'))

        if eval:
            self.model = self.model.eval()

        if share_memory:
            self.model.share_memory()


    def extract(self, x):
        features = self.model(x)
        print("masked_RCnn", x.shape, features[self.feat_layer].shape)
        return features[self.feat_layer]


class Resnet(object):

    def __init__(self, args, eval=True, share_memory=False, use_conv_feat=True):
        self.model_type = args.visual_model
        self.gpu = args.gpu

        # choose model type
        if self.model_type == "maskrcnn":
            self.resnet_model = MaskRCNN(args, eval, share_memory)
        else:
            self.resnet_model = Resnet18(args, eval, share_memory, use_conv_feat=use_conv_feat)

        # normalization transform
        self.transform = self.get_default_transform()


    @staticmethod
    def get_default_transform():
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])

    def featurize(self, images, batch=32):
        images_normalized = torch.stack([self.transform(i) for i in images], dim=0)
        if self.gpu:
            images_normalized = images_normalized.to(torch.device('cuda'))

        out = []
        with torch.set_grad_enabled(False):
            for i in range(0, images_normalized.size(0), batch):
                b = images_normalized[i:i+batch]
                out.append(self.resnet_model.extract(b))
        return torch.cat(out, dim=0)