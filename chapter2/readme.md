The data and the code is available in the original repo:
    github.com/rmunro/pytorch_active_learning

# Data
The format of the data is as follows:
 - 0. Text ID ( a unique ID for this item)
 - 1. Text (the text itself)
 - 2. Label (the label: 1='disaster-related'; 0='not disaster-related')
 - 3. Sampling strategy (the active learning strategy that we used to sample this item)
 - 4. Confidence (the machine learning confidence that this item is 'disaster-related')


 # Identifying outliers
Although there are many ways to ensure that we are maximizing the diversity of 
 1. For each item in the unlabeled data, count the average number of word matches it has with items already in the training data.
 2. Rank the items by their average match.
 3. Sample the item with the lowest average number of matches.
 4. Add that item ot the labeled data.
 5. Repeat these steps until you have sampled enough for one iteration of human review.


 # Iterations
 Iteration is everything! In the example code here's what happens through the iterations:

 - *First iteration* You are annotating mostly 'not disaster-related' headlines, which can feel tedious. The balance will improve when active learning kicks in, but for now, it is necessary to get the randomly sampled evaluation data. Edge cases could be a challenge.
 - *Second iteration* The first model is created F-score is probably terrible, only .20. AUC, however, around 0.75. So despite ad bad acuracy you can find disaster-related messages better than chance. You will immediately notice on your second iteration that a large number of items is disaster-related. Model will strill try to predict most things as 'not disaster-related' so anything close to 50% confidence is at the 'disaster-related' end of the scale. This example shows that active learning can be self-correcting: it is oversampling a lower-frequency label without requiring you to explicitly implement a targeted strategy for samplign important labels.
 - *Third and fourth iterations-- Model accuracy should start to increase at this stage, as you're now labeling many more 'disaster-related' healdines, brinigng the proposed annotation data ccloser to 50:50 for each label.
 - *Fifth to tenth iterations* Models start to rech reasonable levels of accuracy.

 # Evaluation
 It is important to get the evaluation data first, as there are many ways to inadvertently bias your evaluation data after you have started other sampling techniques. Some possibilities that could go wrong if you don't pull out your evaluation data first:

 - If you forget to sample evaluation data from your unlabeled items until after you have sampled by low confidence, your evaluation data will be biased toward the remaining high-confidence items, and your model will appear to be more accurate than it is.

 - If you forget to sample evaluation data and you pull evaluation data from oyour training data after you have sampled by confidence, your evaluation data will be biased toward low-confidence items, and your model will appear to be less accurate than it is.

 - If you have implemented outlier detection and later try to pull out evaluation data, it is almost impossible to avoid bias, as the items you pulled out have already contributed to the sampling of additional outliers. 