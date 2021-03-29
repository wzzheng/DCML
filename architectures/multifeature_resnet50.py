"""
The network architectures and weights are adapted and used from the great https://github.com/Cadene/pretrained-models.pytorch.
"""
import torch, torch.nn as nn
import pretrainedmodels as ptm

def soft_select(x, prob, ensemble_num, embed_dim):
    y = 0
    k = x.size(0)

    for i in range(ensemble_num):
        y = y + prob[:, i].view(k, 1)*x[:, i*embed_dim:(i+1)*embed_dim]

    return y



"""============================================================="""
class Network(torch.nn.Module):
    def __init__(self, opt):
        super(Network, self).__init__()

        self.pars  = opt
        self.model = ptm.__dict__['resnet50'](num_classes=1000, pretrained='imagenet')

        self.name = opt.arch

        if 'frozen' in opt.arch:
            for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
                module.eval()
                module.train = lambda _: None

        self.feature_dim = self.model.last_linear.in_features

        self.layer_blocks = nn.ModuleList([self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4])

        e_layer = nn.ModuleList()
        for n in range(opt.ensemble_num):
            e_layer.append(torch.nn.Linear(self.feature_dim, opt.embed_dim))
        self.ensemble_layer = e_layer
        
        c_layer = nn.ModuleList()
        for n in range(opt.compos_num):
            c_layer.append(torch.nn.Linear((opt.ensemble_num)*opt.embed_dim, opt.ensemble_num))
        self.compos_layer = c_layer

    def forward(self, x):
        x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))
        for layerblock in self.layer_blocks:
            x = layerblock(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0),-1)

        x_embedding = torch.cat([self.ensemble_layer[i](x) for i in range(self.pars.ensemble_num)], dim = 1)
        detach_embedding = x_embedding.detach()

        prob = []
        supervise_loss = 0
        mod_x = []

        for i in range(self.pars.compos_num):
            prob.append(torch.nn.functional.softmax(self.compos_layer[i](detach_embedding), dim = 1))
            prob0 = prob[i]
            log_prob = -torch.log(prob0)
            index = prob0.max(dim = 1, keepdim = True)[1]
            supervise = torch.zeros_like(prob0).scatter_(1, index, 1.0)
            supervise_loss = supervise_loss + torch.sum(log_prob*supervise)/x.size(0)

        for i in range(self.pars.compos_num):
            mod_x.append(soft_select(x_embedding, prob[i], self.pars.ensemble_num, self.pars.embed_dim))
        
        n = (self.pars.compos_num)/4
        if n==1:
            mod_x1 = mod_x[0]
            mod_x2 = mod_x[1]
            mod_x3 = mod_x[2]
            mod_x4 = mod_x[3]
        else:
            mod_x1 = torch.cat([mod_x[i] for i in range(0,n)], dim = 1)
            mod_x2 = torch.cat([mod_x[i] for i in range(n,2*n)], dim = 1)
            mod_x3 = torch.cat([mod_x[i] for i in range(2*n,3*n)], dim = 1)
            mod_x4 = torch.cat([mod_x[i] for i in range(3*n,4*n)], dim = 1)

        out_dict = {}

        if len(self.pars.diva_features) == 1:
            if not 'normalize' in self.pars.arch:
                out_dict['discriminative'] = mod_x
            else:
                out_dict['discriminative'] = torch.nn.functional.normalize(mod_x, dim = -1)
            return out_dict, x, supervise_loss
        else if len(self.pars.diva_features) == 2:
            mod_1 = torch.cat([mod_x1, mod_x2], dim = 1)
            mod_2 = torch.cat([mod_x3, mod_x4], dim = 1)
            if not 'normalize' in self.pars.arch:
                out_dict[self.pars.diva_features[0]] = mod_1
                out_dict[self.pars.diva_features[1]] = mod_2
            else:
                out_dict[self.pars.diva_features[0]] = torch.nn.functional.normalize(mod_1, dim = -1)
                out_dict[self.pars.diva_features[1]] = torch.nn.functional.normalize(mod_2, dim = -1)
            return out_dict, x, supervise_loss
        else if len(self.pars.diva_features) == 3:
            mod_1 = torch.cat([mod_x1, mod_x2], dim = 1)
            mod_2 = mod_x3
            mod_3 = mod_x4
            if not 'normalize' in self.pars.arch:
                out_dict[self.pars.diva_features[0]] = mod_1
                out_dict[self.pars.diva_features[1]] = mod_2
                out_dict[self.pars.diva_features[2]] = mod_3
            else:
                out_dict[self.pars.diva_features[0]] = torch.nn.functional.normalize(mod_1, dim = -1)
                out_dict[self.pars.diva_features[1]] = torch.nn.functional.normalize(mod_2, dim = -1)
                out_dict[self.pars.diva_features[2]] = torch.nn.functional.normalize(mod_3, dim = -1)
            return out_dict, x, supervise_loss

        if not 'normalize' in self.pars.arch:
            out_dict['discriminative'] = mod_x1
            out_dict['selfsimilarity'] = mod_x2
            out_dict['shared'] = mod_x3
            out_dict['intra'] = mod_x4
        else:
            out_dict['discriminative'] = torch.nn.functional.normalize(mod_x1, dim=-1)
            out_dict['selfsimilarity'] = torch.nn.functional.normalize(mod_x2, dim=-1)
            out_dict['shared'] = torch.nn.functional.normalize(mod_x3, dim=-1)
            out_dict['intra'] = torch.nn.functional.normalize(mod_x4, dim=-1)

        return out_dict, x, supervise_loss
