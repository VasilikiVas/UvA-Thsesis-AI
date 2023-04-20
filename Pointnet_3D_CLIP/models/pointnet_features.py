'''import torch.nn as nn
from pointnet_utils import PointNetEncoder

class PointNetFeatures(nn.Module):
    def __init__(self, normal_channel=True):
        super(PointNetFeatures, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        return x, trans_feat'''


import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torch
from pointnet_utils import PointNetEncoder

class get_model(nn.Module):
    def __init__(self, k=40, normal_channel=True):
        super(get_model, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        #x = F.relu(self.fc1(x))
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.fc2(x)
        #x = self.bn1(self.fc1(x))
        #or 
        #Linear layer(512, 512)

        return x, trans_feat

class get_loss(torch.nn.Module):
    #minimize the L2 distance between the feature and the ground truth
    def __init__(self, margin=0.0):
        super(get_loss, self).__init__()
        self.margin = margin

    def forward(self, feature, gt1, gt2):
        #L2 distance between the feature and the ground truth using the margin

        #diff1 = feature - gt1
        #diff2 = feature - gt2
        #loss1 = torch.mean(torch.sum(torch.clamp(diff1, min=0)**2, dim=1))
        #loss2 = torch.mean(torch.sum(torch.clamp(diff2, min=0)**2, dim=1))
        #loss = 0.5 * (loss1 + loss2)

        loss_func = torch.nn.MSELoss()
        #loss = loss_func(feature, gt1)
        loss = loss_func(feature, gt2)
        #loss = 0.5*(loss1 + loss2)

        #euclidean_distance1 = F.pairwise_distance(feature, gt1, p=2)
        #euclidean_distance2 = F.pairwise_distance(feature, gt2, p=2)
        #loss1 = torch.mean(torch.pow(euclidean_distance1 - self.margin, 2))
        #loss2 = torch.mean(torch.pow(euclidean_distance2 - self.margin, 2))
        #loss = 0.5 * (loss1 + loss2)

        #cos_sim1 = F.cosine_similarity(gt1, feature)
        #cos_sim2 = F.cosine_similarity(gt2, feature)
        #loss1 = torch.mean(1 - cos_sim1)
        #loss2 = torch.mean(1 - cos_sim2)
        #loss = 0.5 * (loss1 + loss2)

        #cos_sim1 = F.cosine_similarity(gt1, feature)
        #cos_sim2 = F.cosine_similarity(gt2, feature)
        #loss1 = torch.mean(torch.clamp(cos_sim1, min=0))
        #loss2 = torch.mean(torch.clamp(cos_sim2, min=0))
        #loss = 0.5 * (loss1 + loss2)

        return loss

    '''def forward(self, feature, gt1, gt2):
        euclidean_distance1 = F.pairwise_distance(feature, gt1, p=2)
        euclidean_distance2 = F.pairwise_distance(feature, gt2, p=2)
        #max (0,1) within the margin
        loss1 = torch.mean(torch.pow(euclidean_distance1 - self.margin, 2))
        loss2 = torch.mean(torch.pow(euclidean_distance2 - self.margin, 2))
        loss = 0.5 * (loss1 + loss2)
        return loss'''