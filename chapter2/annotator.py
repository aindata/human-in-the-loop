"""
Implementation of active learning strategies. Please notice that most of the code in this repo is ported from:

https://github.com/rmunro/pytorch_active_learning
"""

"""
Main principles in annotator building:
1. Build an interface that allows annotators to foucs on one part of the screen.
2. Allow hot kes for all actions.
3. Include a back/undo option.
"""

import csv
annotation_instructions = "Please type 1 if this message is disaster-related, "
annotation_instructions += "or hit Enter if not.\n"
annotation_instructions += "Type 2 to go back to the last message, "
annotation_instructions += "type d to see detailed definitions, "
annotation_instructions += "or type s to save your annotations.\n"

last_instruction = "All done!\n"
last_instruction += "Type 2 to go back to change any labels,\n"
last_instruction += "or Enter to save your annotations."

detailed_instructions = "A 'disaster-related' headline is any story about a disaster.\n"
detailed_instructions += "It includes:\n"
detailed_instructions += "  - human, animal and plant disasters.\n"
detailed_instructions += "  - the response to disasters (aid).\n"
detailed_instructions += "  - natural disasters and man-made ones like wars.\n"
detailed_instructions += "It does not include:\n"
detailed_instructions += "  - criminal acts and non-disaster-related police work\n"
detailed_instructions += "  - post-response activity like disaster-related memorials.\n\n"

class DataReader(object):

    def __init__(self) -> None:
        self.already_labeled = {}
        self.feature_index = {}
    
    def load_data(self, path_to_file, skip_labeled=False):
        with open(path_to_file, 'r') as f:
            data = []
            reader = csv.reader(f)
            for row in reader:
                if skip_labeled and row[0] in self.already_labeled:
                    continue

                if len(row) < 3:
                    row.append("") # empty col for LABEL 
                if len(row) < 4:
                    row.append("") # empty col for SAMPLING_STRATEGY
                if len(row) < 5:
                    row.append(0)  # empty col for CONFIDENCE 
                data.append(row)

                label = str(row[2])
                if row[2] != "":
                    textid = row[0]
                    self.already_labeled[textid] = label
        return data
    @classmethod
    def append_data(cls, path_to_file, data):
        with open(path_to_file, 'a', errors='replace') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(data)
    @classmethod    
    def write_data(cls, path_to_file, data):
        with open(path_to_file, 'w', errors='replace') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(data)

def get_annotations(data,
                    already_labeled,
                    verbose=False,
                    default_sampling_strategy='random'
                    ):
    """
    Prompts annnotator to label from command line and appends corresponding annotations to data
    Args.
        data: a list of unlabeled items where each item consists of 
            [ID, TEXT, LABEL, SAMPLING_STRATEGY, CONFIDENCE]
        default_sampling_strategy: choice of sampling
    """
    ind = 0
    while ind <= len(data):
        if ind < 0:
            ind = 0
        if ind < len(data):
            textid, text, label, strategy, score = data[ind]

            strategy = "random" if strategy == "" else strategy

            if textid in already_labeled:
                if verbose:
                    print('Skipping labeled ' + str(textid)+ 'with label '+ label)
                ind+=1
            else:
                print(annotation_instructions)
                if verbose:
                    print("Sampled with strategy `"+str(strategy)+"` and score "+str(round(score,3)))

                label = str(input(text + '\n\n> '))
                if label == '2':
                    ind -= 1 # go back
                elif label == 'd':
                    print(detailed_instructions)
                elif label == 's':
                    break  # save and exit

                else:
                    if not label == '1':
                        label = '0' # treat everything other than 1 as 0
                    data[ind][2] = label  # add label to our data

                    if data[ind][3] is None or data[ind][3] == "":
                        data[ind][3] = default_sampling_strategy
                    ind += 1
        else:
            # last one - give annotator a chance to go back
            print(last_instruction)
            label = str(input('\n\n> '))
            if label == '2':
                ind -= 1
            else:
                ind += 1
    return data


