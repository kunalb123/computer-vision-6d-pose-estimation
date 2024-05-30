import torch
import torch.nn as nn
import torchvision.models as models

class DeepPose(nn.Module):
    def __init__(self):
        super(DeepPose, self).__init__()
        vgg19 = models.vgg19(pretrained=True)
        self.features = nn.Sequential(*list(vgg19.features.children())[:23])

        self.reduce_dim1 = nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=1)
        self.reduce_dim2 = nn.Conv2d(256, 128, kernel_size=3, padding=1, stride=1)

        self.stage1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.stage1_belief = nn.Conv2d(512, 9, kernel_size=1)
        self.stage1_vector = nn.Conv2d(512, 16, kernel_size=1)



        self.last_five_stages = nn.ModuleList()
        for _ in range(5):
            stage = nn.Sequential(
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
            )
            self.last_five_stages.append(stage)

        self.belief_layers = nn.ModuleList()
        self.vector_layers = nn.ModuleList()
        for _ in range(5):
            self.belief_layers.append(nn.Conv2d(128, 9, kernel_size=1))
            self.vector_layers.append(nn.Conv2d(128, 16, kernel_size=1))


    def forward(self, x):
        x = self.features(x)
        x = self.reduce_dim1(x)
        x = self.reduce_dim2(x)

        # First stage
        x1 = self.stage1(x)
        belief1 = self.stage1_belief(x1)
        vector1 = self.stage1_vector(x1)

        belief_maps = [belief1]
        vector_fields = [vector1]

        # Subsequent stages
        for i in range(5):
            x = torch.cat([x, belief_maps[-1], vector_fields[-1]], dim=1)
            x = self.last_five_stages[i](x)
            belief = self.belief_layers[i](x)
            vector = self.vector_layers[i](x)
            belief_maps.append(belief)
            vector_fields.append(vector)

        return belief_maps, vector_fields


