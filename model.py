import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG19_Weights

class DeepPose(nn.Module):
    def __init__(self):
        super(DeepPose, self).__init__()
        vgg19 = models.vgg19(VGG19_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(vgg19.features.children())[:23])
        
        self.num_stages = 5     # these are the additional stages change this number to change everywhere

        self.relu = nn.ReLU()
        self.reduce_dim1 = nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=1)
        self.reduce_dim2 = nn.Conv2d(256, 128, kernel_size=3, padding=1, stride=1)

        self.interim_conv = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1) # uncomment out in forward

        self.stage1_b = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 9, kernel_size=1)
        )
        self.stage1_v = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 16, kernel_size=1)
        )

        self.last_five_stages_b = nn.ModuleList()
        self.last_five_stages_v = nn.ModuleList()
        for _ in range(self.num_stages):
            stage_b = nn.Sequential(
                nn.Conv2d(153, 128, kernel_size=7, padding=3),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=7, padding=3),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=7, padding=3),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=7, padding=3),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=7, padding=3),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 9, kernel_size=1)
            )
            stage_v = nn.Sequential(
                nn.Conv2d(153, 128, kernel_size=7, padding=3),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=7, padding=3),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=7, padding=3),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=7, padding=3),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=7, padding=3),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 16, kernel_size=1)
            )
            self.last_five_stages_b.append(stage_b)
            self.last_five_stages_v.append(stage_v)


    def forward(self, x):
        x = self.features(x)
        x = self.relu(self.reduce_dim1(x))
        x = self.relu(self.reduce_dim2(x))

        # x = self.relu(self.interim_conv(x))

        # First stage
        xb = self.stage1_b(x)
        xv = self.stage1_v(x)

        belief_maps = torch.unsqueeze(xb, 0)
        vector_fields = torch.unsqueeze(xv, 0)
        # Subsequent stages
        for i in range(self.num_stages):
            out = torch.cat([x, belief_maps[-1], vector_fields[-1]], dim=1)
            xb = self.last_five_stages_b[i](out)
            xv = self.last_five_stages_v[i](out)
            belief_maps = torch.cat((belief_maps, torch.unsqueeze(xb, 0)))
            vector_fields = torch.cat((vector_fields, torch.unsqueeze(xv, 0)))

        return belief_maps, vector_fields


