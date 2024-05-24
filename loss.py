import torch.nn as nn

class CompositeLoss(nn.Module):
    def __init__(self):
        super(CompositeLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, pred_belief, gt_belief, pred_vector, gt_vector):
        belief_loss = self.mse_loss(pred_belief, gt_belief)
        vector_loss = self.mse_loss(pred_vector, gt_vector)
        total_loss = belief_loss + vector_loss
        return total_loss
