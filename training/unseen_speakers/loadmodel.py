from keras.optimizers import Adam
from keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint
from lipnet.lipreading.generators import BasicGenerator
from lipnet.lipreading.callbacks import Statistics, Visualize
from lipnet.lipreading.curriculums import Curriculum
from lipnet.core.decoders import Decoder
from lipnet.lipreading.helpers import labels_to_text
from lipnet.utils.spell import Spell
from lipnet.model2 import LipNet
import numpy as np
import datetime
import os
import glob

def curriculum_rules(epoch):
    return { 'sentence_length': -1, 'flip_probability': 0.5, 'jitter_probability': 0.05 }

np.random.seed(55)

run_name = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR  = os.path.join(CURRENT_PATH, 'datasets')
OUTPUT_DIR   = os.path.join(CURRENT_PATH, 'results')
LOG_DIR      = os.path.join(CURRENT_PATH, 'logs')

PREDICT_GREEDY      = False
PREDICT_BEAM_WIDTH  = 200
PREDICT_DICTIONARY  = os.path.join(CURRENT_PATH,'..','..','common','dictionaries','grid.txt')
curriculum = Curriculum(curriculum_rules)
#minibatch size 50 ->3
lip_gen = BasicGenerator(dataset_path=DATASET_DIR,
                                minibatch_size=20,
                                img_c=3,img_w=100, img_h=50, frames_n=75,
                                absolute_max_string_len=32,
                                curriculum=curriculum, start_epoch=0).build()


lipnet = LipNet(img_c=3, img_w=100, img_h=50, frames_n=75,
            absolute_max_string_len=32, output_size=lip_gen.get_output_size())
#show the summary of the network
lipnet.summary()

#import the adam optimizer
adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

#Here we import the model from the model2 LipNet.model()

#model compile, using ctc as loss cal and adam as optimizer
lipnet.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam) 
#import the spell autocorrection
spell = Spell(path=PREDICT_DICTIONARY)

# decode the labels to the text and form the Y_data spell.sentence 
decoder = Decoder(greedy=PREDICT_GREEDY, beam_width=PREDICT_BEAM_WIDTH,
          postprocessors=[labels_to_text, spell.sentence])

train_gen=lip_gen.get_batch(0,20,train=True)
val_gen=lip_gen.get_batch(0,5,train=False)



weight_file = 'D:/GitHub/LipNet/training/unseen_speakers/results/weights3000.h5'
lipnet.model.load_weights(weight_file)



#save the model and weights
#checkpoint  = ModelCheckpoint('D:/GitHub/LipNet/training/unseen_speakers/results/weights5000.h5', monitor='train_loss', save_weights_only=True, mode='auto', period=1)
csv_logger  = CSVLogger('D:/GitHub/LipNet/training/unseen_speakers/logs/losslog0824.csv', separator=',', append=True)
tensorboard = TensorBoard(log_dir='D:/GitHub/LipNet/training/unseen_speakers/logs/0824')
checkpoint  = ModelCheckpoint('D:/GitHub/LipNet/training/unseen_speakers/results/valweights3000.h5', monitor='val_loss', save_weights_only=True, mode='auto', period=1)

#lipnet.model
lipnet.model.fit(x=train_gen[0],y=train_gen[1],batch_size=5,epochs=20,verbose=1,callbacks=[csv_logger,tensorboard,checkpoint],
                 validation_split=0.0, validation_data=val_gen, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0)



#loss, accuracy = lipnet.model.evaluate(gen[2],gen[3])

#train_size=lip_gen.training_size
#steps=lip_gen.default_training_steps
#batchsize=lip_gen.minibatch_size

#a=lip_gen.train_list

#
#generator=lip_gen.next_train()
#

# test align hash

#a=lip_gen.align_hash
#print(a.keys())
#print(a['bbaszp'].align)
#print(a['bbaszp'].sentence)