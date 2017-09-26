
from lipnet.utils.spell import Spell
import os

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
PREDICT_DICTIONARY  = os.path.join(CURRENT_PATH,'..','..','common','dictionaries','big.txt')
spell=Spell(path=PREDICT_DICTIONARY)

corr=spell.sentence('thiis obwjecd is an redd apwle')
print(corr)
