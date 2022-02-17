import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.nn import Parameter


__all__ = ['Classifier', 'NormalizedClassifier']


class Classifier(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes) #2048到751的全连接层？
        init.normal_(self.classifier.weight.data, std=0.001)
        #print(self.classifier.weight.data.size())
        init.constant_(self.classifier.bias.data, 0.0)


    def forward(self, x):
        y = self.classifier(x)
        #print("y={}/n".format(y))

        return y
        

class NormalizedClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.weight = Parameter(torch.Tensor(num_classes, feature_dim)) #same as cluster memory M
        self.weight.data.uniform_(-1, 1).renorm_(2,0,1e-5).mul_(1e5)    #if we use the way of that paper, weight should be initilized
                                                                        #by the mean feature of each class,讨论后应该是不一样的，这个就是weight
                                                                        #而M应该是一篇单独的空间


    def forward(self, x):
        w = self.weight  

        x = F.normalize(x, p=2, dim=1)
        w = F.normalize(w, p=2, dim=1)

        return F.linear(x, w)



