# Imports

## General Imports
import numpy as np

## Visualization
import matplotlib.pyplot as plt

## Audio Imports
import librosa, librosa.display           #https://librosa.github.io/librosa/index.html
from music21.duration import DurationException
from music21.tempo import MetronomeMark   #http://web.mit.edu/music21/
from music21.note import Note, Rest
from music21.stream import Stream
from music21 import metadata
from music21 import instrument
import wave


# Configurations

## Matplotlib
plt.rc("figure", figsize=(10, 5))


# Parameters
## Signal Processing
fs = 44100                               # Sampling Frequency
nfft = 2048                              # length of the FFT window
overlap = 0.5                            # Hop overlap percentage
hop_length = int(nfft*(1-overlap))       # Number of samples between successive frames
n_bins = 72                              # Number of frequency bins
mag_exp = 4                              # Magnitude Exponent
pre_post_max = 6                         # Pre- and post- samples for peak picking
cqt_threshold = -60                      # Threshold for CQT dB levels, all values below threshold are set to -120 dB
backtrack = False

filename = "sweet_child_o_mine_intro"

# Load audio file
x, fs = librosa.load("input/" + filename + ".wav", sr=None, mono=True)
# Audio data information
print("x Shape=", x.shape)
print("Sample rate fs=", fs)
print("Audio Length in seconds=%d [s]" % (x.shape[0]/fs))


# CQT Function
def calc_cqt(x,fs=fs,hop_length=hop_length, n_bins=n_bins, mag_exp=mag_exp):
    C = librosa.cqt(x, sr=fs, hop_length=hop_length, fmin=None, n_bins=n_bins)
    C_mag = librosa.magphase(C)[0]**mag_exp
    CdB = librosa.core.amplitude_to_db(C_mag ,ref=np.max)
    return CdB


# CQT Threshold
def cqt_thresholded(cqt,thres=cqt_threshold):
    new_cqt=np.copy(cqt)
    new_cqt[new_cqt<thres]=-120
    return new_cqt


# Onset Envelope from CQT
def calc_onset_env(cqt):
    return librosa.onset.onset_strength(S=cqt, sr=fs, aggregate=np.mean, hop_length=hop_length)


# Onset from Onset Envelope
def calc_onset(cqt, pre_post_max=pre_post_max, backtrack=True):
    onset_env=calc_onset_env(cqt)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env,
                                           sr=fs, units='frames',
                                           hop_length=hop_length,
                                           backtrack=backtrack,
                                           pre_max=pre_post_max,
                                           post_max=pre_post_max)
    onset_boundaries = np.concatenate([[0], onset_frames, [cqt.shape[1]]])
    onset_times = librosa.frames_to_time(onset_boundaries, sr=fs, hop_length=hop_length)
    return [onset_times, onset_boundaries, onset_env]


# Fine Tuning
# without UI because jupyter widgets not available
def inter_cqt_tuning(mag_exp,thres,pre_post_max, backtrack):
    global CdB
    CdB = calc_cqt(x,fs,hop_length, n_bins, mag_exp)
    plt.figure()
    new_cqt=cqt_thresholded(CdB,thres)
    librosa.display.specshow(new_cqt, sr=fs, hop_length=hop_length, x_axis='time', y_axis='cqt_note', cmap='coolwarm')
    plt.ylim([librosa.note_to_hz('B2'),librosa.note_to_hz('B6')])
    global onsets
    onsets=calc_onset(new_cqt,pre_post_max, backtrack)
    plt.vlines(onsets[0], 0, fs/2, color='k', alpha=0.8)
    plt.title("CQT - " + filename)
    plt.colorbar()
    plt.show()

inter_cqt_tuning(mag_exp, cqt_threshold, pre_post_max, backtrack)


# Estimate Tempo
tempo, beats=librosa.beat.beat_track(y=None, sr=fs, onset_envelope=onsets[2], hop_length=hop_length,
               start_bpm=120.0, tightness=100, trim=True, bpm=None,
               units='frames')
tempo=int(2*round(tempo/2))
print("BPM: {}".format(tempo))
mm = MetronomeMark(referent='quarter', number=tempo)


# Convert Seconds to Quarter-Notes
def time_to_beat(duration, tempo):
    return (tempo*duration/60)


# Remap input to 0-1 for Sine Amplitude or to 0-127 for MIDI
def remap(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


# Generate Sinewave and music21 notes
def generate_sine_midi_note(f0_info, sr, n_duration):
    f0 = f0_info[0]
    A = remap(f0_info[1], CdB.min(), CdB.max(), 0, 1)
    duration = librosa.frames_to_time(n_duration, sr=fs, hop_length=hop_length)
    # Generate music21 note
    note_duration = 0.02 * np.around(duration / 2 / 0.02)  # Round to 2 decimal places for music21 compatibility
    midi_velocity = int(round(remap(f0_info[1], CdB.min(), CdB.max(), 0, 127)))
    if f0 == None:
        try:
            note_info = Rest(type=mm.secondsToDuration(note_duration).type)
        except DurationException:
            note_info = None
        f0 = 0
    else:
        midi_note = round(librosa.hz_to_midi(f0))
        try:
            note = Note(midi_note, type=mm.secondsToDuration(note_duration).type)
            note.volume.velocity = midi_velocity
            note_info = [note]
        except DurationException:
            note_info = None

    if note_info is None:
        return None

    # Generate Sinewave
    n = np.arange(librosa.frames_to_samples(n_duration, hop_length=hop_length))
    sine_wave = A * np.sin(2 * np.pi * f0 * n / float(sr))
    return [sine_wave, note_info]


# Estimate Pitch
def estimate_pitch(segment, threshold):
    freqs = librosa.cqt_frequencies(n_bins=n_bins, fmin=librosa.note_to_hz('C1'),
                            bins_per_octave=12)
    if segment.max()<threshold:
        return [None, np.mean((np.amax(segment,axis=0)))]
    else:
        f0 = int(np.mean((np.argmax(segment,axis=0))))
    return [freqs[f0], np.mean((np.amax(segment,axis=0)))]


# Generate notes from Pitch estimation
def estimate_pitch_and_notes(x, onset_boundaries, i, sr):
    n0 = onset_boundaries[i]
    n1 = onset_boundaries[i+1]
    f0_info = estimate_pitch(np.mean(x[:,n0:n1],axis=1),threshold=cqt_threshold)
    return generate_sine_midi_note(f0_info, sr, n1-n0)


# Array of music information - Sinewave and music21 notes
music_info = np.array([
    estimate_pitch_and_notes(CdB, onsets[1], i, sr=fs)
    for i in range(len(onsets[1])-1)
], dtype=object)

music_info = np.array([info for info in music_info if info is not None], dtype=object)

# Get sinewave
synth_audio=np.concatenate(music_info[:,0])

# Convert audio to 16-bit int, generated audio is 64-bit float
synth_audio_converted = np.array([
    sample * 32767 for sample in synth_audio
], dtype=np.int16)

# write to wav file
file = wave.open("output/" + filename + "_sine.wav", "wb")
file.setnchannels(1)
file.setsampwidth(2)    # 2 bytes = 16 bit
file.setframerate(fs)
file.writeframes(synth_audio_converted)
file.close()


# Get music21 notes
note_info = list(music_info[:,1])

# Create music21 stream
s = Stream()
s.append(mm)
electricguitar = instrument.fromString('electric guitar')
electricguitar.midiChannel=0
electricguitar.midiProgram=30  #Set program to Overdriven Guitar
s.append(electricguitar)
s.insert(0, metadata.Metadata())
for note in note_info:
    s.append(note)

# Analyse music21 stream to get song Key
key=s.analyze('key')
print("Key: " + key.name)
# Insert Key to Stream
s.insert(0, key)

# Save MIDI to file
s.write('midi', "output/" + filename + "_music21.mid")
