import numpy as np
from lipnet.lipreading.helpers import text_to_labels
class Align(object):
    def __init__(self, absolute_max_string_len=32, label_func=None):
        self.label_func = label_func
        self.absolute_max_string_len = absolute_max_string_len

    def from_file(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()
            # this is to spilt the timeline(frames)from 0 to 75 with start time, end time and word
        align = [(int(y[0])/1000, int(y[1])/1000, y[2]) for y in [x.strip().split(" ") for x in lines]]
        self.build(align)
        #return the self attribute align,sentence,label and padded_label
        return self

    def from_array(self, align):
        self.build(align)
        return self

    def build(self, align):
        # delete the blank align  
        self.align = self.strip(align, ['sp','sil'])
        self.sentence = self.get_sentence(align)
        # the label is the mapping from alphabet and number 0-25, 26 is the pause time
        self.label = self.get_label(self.sentence)
        self.padded_label = self.get_padded_label(self.label)

    def strip(self, align, items):
        return [sub for sub in align if sub[2] not in items]

    def get_sentence(self, align):
        return " ".join([y[-1] for y in align if y[-1] not in ['sp', 'sil']])

    def get_label(self, sentence):
        return self.label_func(sentence)

    def get_padded_label(self, label):
        # get -1 padding 
        padding = np.ones((self.absolute_max_string_len-len(label))) * -1
        return np.concatenate((np.array(label), padding), axis=0)

    @property
    def word_length(self):
        return len(self.sentence.split(" "))

    @property
    def sentence_length(self):
        return len(self.sentence)

    @property
    def label_length(self):
        return len(self.label)

a=Align(label_func=text_to_labels)
returntype=a.from_file('D:/align/bbae8n.align')
cc=returntype.align
align1=a.align
sentence1=a.sentence
label1=a.label
paddedlabel1=a.padded_label


