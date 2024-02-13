from __future__ import absolute_import, division, print_function

from .cross_entropy_loss import CrossEntropyLoss
from .hard_mine_triplet_loss import TripletLoss


def DeepSupervision(criterion, xs, y):
    """DeepSupervision

    Applies criterion to each element in a list.

    Args:
        criterion: loss function
        xs: tuple of inputs
        y: ground truth
    """
    loss = 0.0
    for x in xs:
        loss += criterion(x, y)
    loss /= len(xs)
    return loss
