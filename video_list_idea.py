# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 23:38:43 2021

@author: rehan
"""

import cv2
from difflib import SequenceMatcher
import os
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

video_similarity_dict={'basketball': 0, 'boxing': 0, 
                    'cricket': 0, 'formula1': 0, 'kabaddi': 0, 
                    'swimming': 0, 'table_tennis': 0, 'weight_lifting': 0}

vide_search=input('ADD:')# taking input

for key in video_similarity_dict:
    score=similar(vide_search,key)
    video_similarity_dict[key]=score
        
video_folder = max(video_similarity_dict, key=video_similarity_dict.get)

print(video_folder)

speech_folder='data/speech/'+video_folder
music_folder='data/music/'+video_folder

speech_list=os.listdir(speech_folder)
music_list=os.listdir(music_folder)

print(speech_list)
print(music_list)

