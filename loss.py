import torch.nn as nn

class CompositeLoss(nn.Module):
    def __init__(self, stages):
        super(CompositeLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.stages = stages

    def forward(self, pred_belief, gt_belief, pred_vector, gt_vector):
        belief_loss = 0.0
        vector_loss = 0.0
        for i in range(self.stages):
            belief_loss += self.mse_loss(pred_belief[i], gt_belief)
            vector_loss += self.mse_loss(pred_vector[i], gt_vector)
        total_loss = belief_loss + vector_loss
        if total_loss < 0: print('LOSS WAS NEGATIVE:', total_loss)
        return total_loss
