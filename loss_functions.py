import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(reduction='sum')
        self.metrics = ['total_loss']

    def forward(self, input, target, *args):
        loss = self.loss_fn(input, target)
        metrics = {'total_loss': loss.detach().cpu().item()}
        return loss, metrics


if __name__ == '__main__':
    batch_size = 1
    num_classes = 4

    output = torch.zeros(batch_size, num_classes)
    y = torch.zeros(batch_size, dtype=torch.long)
    print(f'output\t: {output}')
    print(f'y\t: {y}')

    loss_fn = CrossEntropyLoss()
    loss, metrics = loss_fn(output, y)
    print(f'loss\t: {loss}')
    print(f'metrics\t: {metrics}')