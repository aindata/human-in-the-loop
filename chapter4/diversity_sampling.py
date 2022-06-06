"""
Implementation of diversity sampling methods the original code can be found in:
https://github.com/rmunro/pytorch_active_learning


"""

from random import shuffle
import torch


class DiversitySampling():
    """
    Diversity Sampling methods from human-in-the-loop book
    """
    
    def __init__(self, verbose=False):
        self.verbose = verbose
    
    def get_validation_rankings(self, model, validation_data, feature_method):
        """
        Get model outliers from unlabeled data
        
        Args.
            model: Machine Learning model
            validation_data: drawn data from training data
            feature_method: a function creating feature vector from raw input
            
        An outlier defined as
            unlabeled_data with the lowest average from rank order of logits
            where rank order is defined by validation data inference
        """

        validation_rankings = []  # 2d array
        
        if self.verbose:
            print("Getting neuran activation scores from validation data")
        
        with torch.no_grad():
            v = 0
            for item in validation_data:
                feature_vector = feature_method(item[1])
                _, logits, _ = model(feature_vector, return_all_layers=True)
                
                neuron_outputs = logits.data.tolist()[0]  # this has a shape of 1 X 2
                
                # initalize array if we haven't yet
                if len(validation_rankings) == 0:
                    for output in neuron_outputs:
                        validation_rankings.append([0.0]* len(validation_data))
                        
                n = 0
                for output in neuron_outputs:
                    validation_rankings[n][v] = output
                    n+=1                   
                v+=1
            
            # Rank-order the validation scores
            v=0
            for validation in validation_rankings:
                validation.sort()
                validation_rankings[v] = validation
                v+=1
            # supppose you have 3 validation instances, and 2 logits here is what you get in validation_rankings
            #   [[0.39697607012505687, 0.5617226252978991, 0.7076910845440585],
            #    [0.9377881798691801, 0.42959004683922297, 0.7019144562834689]]
            return validation_rankings
        
    
    def get_rank(self, value, rankings):
        """
        get the rank of the vlaue in an ordered array as a percentage
        
        Args.
        
        Value: the value for which we want to return the ranked value
        rankings: the ordered array in which to determine the value's ranking
        
        """
        index = 0
        for ranked_number in rankings:
            if value < ranked_number:
                break
            index += 1
            
        if index >= len(rankings):
            index = len(rankings)
            
        elif index > 0:
            diff = rankings[index] - rankings[index-1]
            perc = value - rankings[index-1]
            linear = perc / diff
            index = float(index-1) + linear
                    
        absolute_ranking = index / len(rankings)
        return absolute_ranking
        
    def get_model_outliers(self,
                           model,
                           unlabeled_data,
                           validation_data,
                           feature_method,
                           number=5,
                           limit=10000):
        """
        Get model outliers from unlabeled data
        
        Args.
            model: model for the task
            unlabeled_data: data yet to be labeled
            validation_data: same distribution data drawn from training data
            feature_method: the method to create features from the raw text
            number: number of items to sample
            limit: sample from only this many items for faster sampling (-1 = no limit)
        """
        validation_rankings = self.get_validation_rankings(model, validation_data, feature_method)
        outliers = []
        # TODO: implement limit checking see line 280 from diversity_sampling.py
        
        shuffle(unlabeled_data)
        unlabeled_data = unlabeled_data[:limit]
        
        with torch.no_grad():
            for item in unlabeled_data:
                text = item[1]
                feature_vector = feature_method(text)
                
                _, logits, _ = model(feature_vector, return_all_layers=True)
                
                neuron_outputs = logits.data.tolist()[0]
                
                n=0
                ranks = []
                for output in neuron_outputs:
                    rank = self.get_rank(output, validation_rankings[n])
                    ranks.append(rank)
                    n+= 1
                item[3] = 'logit_rank_outlier'
                item[4] = 1 - (sum(ranks)/ len(neuron_outputs))
                outliers.append(item)
                
        outliers.sort(reverse=True, key=lambda x: x[4])
        return outliers[:number:]
        
        # Cluster-based sampling
        # use cosine similarity to assign clusters for each instance in unlabeled data
        # get_centroid, get_outlier, get_random samples for each unlabeled data
        # for more details please refer to:
        # https://github.com/rmunro/pytorch_active_learning/blob/master/pytorch_clusters.py