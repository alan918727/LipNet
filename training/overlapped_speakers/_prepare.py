import os
import glob
import sys


CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
cr=CURRENT_PATH
DATASET_VIDEO_PATH =sys.argv[1]        #"D:/My Files/Desktop/video/"
DATASET_ALIGN_PATH =sys.argv[2]        #'D:/My Files/Desktop/align/'
VAL_SAMPLES = sys.argv[3]           #No.of videos use to validation 
for speaker_path in glob.glob(os.path.join(DATASET_VIDEO_PATH, '*')):

    speaker_id = os.path.splitext(speaker_path)[0].split('\\')[-1]

    os.makedirs(os.path.join(CURRENT_PATH, speaker_id, 'datasets', 'train'))

    for s_path in glob.glob(os.path.join(DATASET_VIDEO_PATH, '*')):
        s_id = os.path.splitext(s_path)[0].split('\\')[-1]
    
        if s_path == speaker_path:

            os.makedirs(os.path.join(CURRENT_PATH, speaker_id, 'datasets', 'train', s_id))
            os.makedirs(os.path.join(CURRENT_PATH, speaker_id, 'datasets', 'val', s_id))
            n = 0

            for video_path in glob.glob(os.path.join(DATASET_VIDEO_PATH, speaker_id, '*')):
                video_id = os.path.splitext(video_path)[0].split('\\')[-1]
                if n < VAL_SAMPLES:
                    #make symbolic link
                   os.symlink(video_path, os.path.join(CURRENT_PATH, speaker_id, 'datasets', 'val', s_id, video_id))
                else:
                   os.symlink(video_path, os.path.join(CURRENT_PATH, speaker_id, 'datasets', 'train', s_id, video_id))
                n += 1
        else:
           os.symlink(s_path, os.path.join(CURRENT_PATH, speaker_id, 'datasets', 'train', s_id))
    os.symlink(DATASET_ALIGN_PATH, os.path.join(CURRENT_PATH, speaker_id, 'datasets', 'align'))