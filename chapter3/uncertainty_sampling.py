import torch
import math
from random import shuffle


class UncertaintySampler(object):
    """
    Uncertainty Sampling from human-in-the-loop book
    for numpy version please refer to:
    
    https://github.com/rmunro/uncertainty_sampling_numpy/blob/master/uncertainty_sampling_numpy.py
    """
    def __init__(self, istorch=True) -> None:
        # TODO: implement numpy version
        self.istorch = istorch
    
    def least_confidence(self, prob_dist, sorted=False):
        """
        Computes the uncertaity score of an array using
        least confidence sampling in a 0-1 range where 1 is the most uncertain
        
        Assumes probability distribution is a pytorch tensor like:
            tensor([0.0321, 0.6439, 0.0871, 0.2369])
            
        Args.
            prob_dist: a pytorch tensor of real numbers between 0 and 1 
            sorted: bool to signal distribution is sorted from largest to smallest
        """
        if sorted:
            simple_least_conf = prob_dist.data[0]  # most confident prediction
        else:
            simple_least_conf = torch.max(prob_dist)  # most confident prediction
        
        num_labels = prob_dist.numel()
        normalized_least_conf = (1 - simple_least_conf) * (num_labels / (num_labels - 1))
        return normalized_least_conf.item()
        
    def margin_confidence(self, prob_dist, sorted=False):
        """
        Computes the uncertainty score of a probability
        distribution using margin of confidence sampling
        in a 0-1 range where 1 is the most uncertain
        
        Assumes probability distribution is a pytorch tensor
            tensor([0.0321, 0.6439, 0.0871, 0.2369])
        
        Args.
            prob_dist: a pytorch tensor of real numbers between 0 and 1 that total to 1.0
            sorted: bool to signal distribution is sorted from largest to smallest
        """
        if not sorted:
            prob_dist, _ = torch.sort(prob_dist, descending=True)  # sort probs to largest first

        difference = (prob_dist.data[0] - prob_dist.data[1])  # difference between top two props
        margin_conf = 1 - difference
        return margin_conf.item()
    
    def ratio_confidence(self, prob_dist, sorted=False):
        """
        Computes the uncertainty score of an array using ratio confidence sampling in a 0-1 range 
        where 1 is the most uncertain
        """
        if not sorted:
            prob_dist, _ = torch.sort(prob_dist, descending=True)  # sort probs so largest is first
            
        ratio_conf = prob_dist.data[1] / prob_dist.data[0]  # ratio between top two props
        return ratio_conf.item()
    
    def entropy_based(self, prob_dist):
        """
        Computes the uncertainty score of a probability distribution using entropy

        Assumes probability distribution is a pytorch tensor
            tensor([0.0321, 0.6439, 0.0871, 0.2369])
            
        Args.
            prob_dist: pytorch tensor of real numbers between 0 and 1 that total to 1.0
            sorted: bool to signal distribution is sorted from largest to smallest
        """
        log_probs = prob_dist * torch.log2(prob_dist)  # multiply each prob with its base log base 2 
        raw_entropy = 0 - torch.sum(log_probs)
        
        normalized_entropy = raw_entropy / math.log2(prob_dist.numel())
        return normalized_entropy.item()
        
    