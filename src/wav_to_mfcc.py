import random
import librosa
from librosa import feature
from pydub import AudioSegment
from src import settings


def trim_audio_to_fixed_length(file_path, duration_ms):
    filename = file_path.split('/')[-1]
    output_path = settings.BASE_PATH + 'tmp/' + filename
    audio = AudioSegment.from_wav(file_path)

    if len(audio) > duration_ms:
        max_start_time = len(audio) - duration_ms
        start_time = random.randint(0, max_start_time)
        trimmed_audio = audio[start_time:start_time + duration_ms]
    else:
        trimmed_audio = audio

    trimmed_audio.export(output_path, format='wav')
    return output_path


def extract_mfcc(file_path, n_mfcc=13):
    path = trim_audio_to_fixed_length(file_path, duration_ms=20000)
    y, sr = librosa.load(path, sr=None)
    mfcc = feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T
