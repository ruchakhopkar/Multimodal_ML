import torch
import torch.nn as nn
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import os
from torchvision import transforms as T
import numpy as np
#from torchvision import models, transforms


class Resnet18(object):
    '''
    pretrained Resnet18 from torchvision
    '''

    def __init__(self, args, eval=True, share_memory=False, use_conv_feat=False):
        self.model = timm.create_model('cspdarknet53', pretrained=True)
        net = torch.load(os.path.join('models/nn/model_04-17_16_50_49.pt'))
        self.model.load_state_dict(net['model_state_dict'])
        self.config = resolve_data_config({}, model=self.model)
        self.transform = create_transform(**self.config)
        #self.model = models.resnet18(pretrained=True)

        if args.gpu:
            self.model = self.model.to(torch.device('cuda'))

        if eval:
            self.model = self.model.eval()

        if share_memory:
            self.model.share_memory()

        #if use_conv_feat:
            #self.model = nn.Sequential(*list(self.model.children())[:-2])

    def extract(self, x):
        #print(self.model.summary())
        manav = T.ToPILImage()
        
        x = manav(x.squeeze(0).squeeze(0))
        inp = self.transform(x.convert('RGB'))
        h, w= x.size
        inp = torch.unsqueeze(inp,axis=0)
        #print('-'*80)
        #print(self.model)
        #print('inp',inp.shape)
        out = self.model(inp.cuda())
        '''actual_img = [out]
        print(out.shape)
        scale = 2
        arr = np.asarray(x)

        for i in range(0,h,int(h/scale)):
                for j in range(0,w, int(w/scale)):
                    cut = arr[i:int(i+h/scale), j:int(j+w/scale)]
                    cut = Image.fromarray(np.uint8(cut)).convert('RGB')
                    cut = cut.resize((h,w))
                    out = model(self.transform(cut).unsqueeze(0).cuda())
                    actual_img.append(out)
        print('actual_img', np.concatenate(actual_img, axis = 0).shape)'''
        out = out.unsqueeze(2).unsqueeze(3)
        out = torch.broadcast_to(out, (out.shape[0], out.shape[1], 5, 1))
        #print(out.shape)
        return out


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
        #self.transform = self.get_default_transform()
        config = resolve_data_config({}, model=self.resnet_model)
        self.transform = create_transform(**config)


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
                b = images_normalized[i:i+batch].unsqueeze(0)
                out.append(self.resnet_model.extract(b))
        return torch.cat(out, dim=0)