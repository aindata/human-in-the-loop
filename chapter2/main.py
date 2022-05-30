from asyncore import read
from tokenize import Number
from unittest import skip
from modelling import *
from constants import *
from annotator import *
import random
random.seed(4)
from random import shuffle

minimum_evaluation_items = 1200  # annotate this many randomly sampled items forst for evaluation data before creating training data
minimum_training_items = 400  #  minimum number of training items before we first train a model

def main():
    """
    Main function to trigger program
    """
    reader = DataReader()
    training_data = reader.load_data(training_related_data) + reader.load_data(training_not_related_data)
    training_count = len(training_data)
    print('Train data length {}'.format(training_count))

    evaluation_data = reader.load_data(evaluation_related_data) + reader.load_data(evaluation_not_related_data)
    evaluation_count = len(evaluation_data)
    print('Evaluation data length {}'.format(evaluation_count))

    data = reader.load_data(unlabeled_data, skip_labeled=True)
    print('Labeling data count {}'.format(len(data)))
    
    if evaluation_count < minimum_evaluation_items: # that means you need more evaluation data
        print('Creating evaluation data:\n ')
    elif training_count < minimum_training_items:
        print('Creating initial training data:\n')
        shuffle(data)
        needed = minimum_training_items - training_count
        data = data[:needed]
        print(str(needed)+" more annotations needed")
        data = get_annotations(data, reader.already_labeled)

        related = []
        not_related = []

        for item in data:
            label = item[2]
            if label == "1":
                related.append(item)
            elif label == "0":
                not_related.append(item)
        DataReader.append_data(training_related_data, related)
        DataReader.append_data(training_not_related_data, not_related)
    else:
        # let's do active learning!
        print("Let's make some active learning")
        feature_extractor = FeatureExtractor()
        vocab_size = feature_extractor.create_features(data, training_data)
        print(vocab_size)
        model = train_model(training_data, feature_extractor.feature_index,evaluation_data=evaluation_data, vocab_size=vocab_size)

        # apply following breakdown strategies
        # 1. random items
        random_data = get_random_items(data, reader.already_labeled, number=10)
        # 2. low confidence
        low_confident_data = get_low_confidenced(model, 
                                                 data, 
                                                 reader.already_labeled,
                                                 feature_extractor,
                                                 number=20)
        # 3. outliers
        outliers_data = get_outliers(training_data+random_data+low_confident_data, data, reader.already_labeled,number=10)
        
        sampled_data = random_data + low_confident_data + outliers_data
        shuffle(sampled_data)
        print("{} many more annotations needed".format(len(sampled_data)))
        
        sampled_data = get_annotations(sampled_data, reader.already_labeled)
        
        related, not_related = [], []
        for item in sampled_data:
            label = item[2]
            if label == '1':
                related.append(item)
            elif label == '0':
                not_related.append(item)
        
        # append training data
        DataReader.append_data(training_related_data, related)
        DataReader.append_data(training_not_related_data, not_related)
    
    if training_count > minimum_training_items:
        r2 = DataReader()
        training_data = r2.load_data(training_related_data) + r2.load_data(training_not_related_data)
        
        evaluation_data = r2.load_data(evaluation_related_data) + r2.load_data(evaluation_not_related_data)
        evaluation_count = len(evaluation_data)
        feature_extractor2 = FeatureExtractor()
        vocab_size = feature_extractor.create_features(data, training_data)
        model =  train_model(training_data, 
                             feature_extractor2.feature_index,
                             evaluation_data=evaluation_data, 
                             vocab_size=vocab_size)

if __name__ == '__main__':
    main()