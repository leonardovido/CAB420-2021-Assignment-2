import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import csv

from os import system, name
from pathlib import Path

# Path of files origin
path = '/home/leonardo/Downloads/en/'

# define our clear function
def clear():
  
    # for windows
    if name == 'nt':
        _ = system('cls')
  
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = system('clear')

def Main():
    # Load validated dataset
    validated_data = pd.read_csv (path + 'validated.tsv', sep = '\t')

    # Remove unnecessary columns
    columns_to_remove = ['client_id', 'sentence', 'up_votes', 'down_votes', 'age','gender']
    validated_data.drop(columns_to_remove, axis=1, inplace=True)

    # Remove NaN values
    validated_data.replace('nan', np.nan, inplace=True)
    validated_data.dropna(axis=0,inplace=True)

    # Create new folder to save the images
    Path(path + 'spectrograms').mkdir(parents=True, exist_ok=True)

    # Create header for the CVS file
    header = 'filename rmse spectral_centroid_mean spectral_centroid_std spectral_centroid_median rolloff_mean rolloff_std rolloff_median spectral_bandwidth_mean spectral_bandwidth_std spectral_bandwidth_median zero_crossing_rate_mean zero_crossing_rate_std zero_crossing_rate_median'
    for i in range(1, 21):
        header += f' mfcc{i}'
    for i in range(1, 13):
        header += f' chroma_stft{i}'
    header += ' accent'
    header = header.split()

    # Set parameter for the for loop
    cmap = plt.get_cmap('inferno')
    counter = 0;
    total = len(validated_data['path'])

    for filename in validated_data['path']:
        counter += 1
        file_path = path + 'clips/' + filename + '.mp3'
        audio_sample, sr_sample = librosa.load(file_path, mono=True, duration=3)
        clear()
        print('Saving Images in Progres... %d/%d' %(counter, total))
        
        # Check if audio file is 2 seconds
        if len(audio_sample) == (sr_sample*3):
            audio_sample = audio_sample[sr_sample:]
            plt.specgram(audio_sample, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
            plt.axis('off');
            plt.savefig(path + 'spectrograms/' + filename +'.png', bbox_inches='tight', pad_inches=0, dpi=80)
            plt.clf()

            rmse = librosa.feature.rms(y=audio_sample)
            chroma_stft = librosa.feature.chroma_stft(y=audio_sample, sr=sr_sample)
            spec_cent = librosa.feature.spectral_centroid(y=audio_sample, sr=sr_sample)
            spec_bw = librosa.feature.spectral_bandwidth(y=audio_sample, sr=sr_sample)
            rolloff = librosa.feature.spectral_rolloff(y=audio_sample, sr=sr_sample)
            zcr = librosa.feature.zero_crossing_rate(audio_sample)
            mfcc = librosa.feature.mfcc(y=audio_sample, sr=sr_sample)

            to_append = f'{filename} {np.mean(rmse)} {np.mean(spec_cent)} {np.std(spec_cent)} {np.median(spec_cent)} {np.mean(rolloff)} {np.std(rolloff)} {np.median(rolloff)} {np.mean(spec_bw)} {np.std(spec_bw)} {np.median(spec_bw)} {np.mean(zcr)} {np.std(zcr)} {np.median(zcr)}'    
            for e in mfcc:
                to_append += f' {np.mean(e)}'
            for e in chroma_stft:
                to_append += f' {np.mean(e)}'
            to_append += f' ' + str(validated_data['accent'].to_numpy()[counter-1])
            file = open(path + 'dataset.csv', 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())

        #if counter == 10:
        #    break

    print('Saving Images Completed')

if __name__ == '__main__':

    clear()
    print('======== Running Conversion ========\n\n')
    Main()