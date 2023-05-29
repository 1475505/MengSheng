# 概念代码
from musicpy.sampler import *
import matplotlib.pyplot as plt
from piano_transcription_inference import PianoTranscription, sample_rate, load_audio
import os
import subprocess
import torch
from test_input.work import get_audio_result

def Visualization(filepath):
    a = read(filepath)
    notes_set = {}  # 从C0到B8
    notes_list = []
    b = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', "A#", 'B']
    c = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    for i in c:
        for j in b:
            z = j + str(i)
            notes_set[z] = 0
            notes_list.append(z)
    a1_notes = [1]
    for i in range(0, a.track_number):
        a1_notes = a1_notes + a[i].content.notes
        a1_notes = list(one for one in a1_notes if str(one) in notes_list)
    y = []
    m = len(a1_notes)
    for i in range(0, m):
        y.append(a1_notes[i].name + str(a1_notes[i].num))
    for i in y:
        notes_set[i] = notes_set[i] + 1
    d1 = list(notes_set.keys())
    d2 = list(notes_set.values())
    plt.figure(figsize=(32, 10))
    plt.bar(d1, d2)
    plt.savefig('./notes_frequency.jpg', dpi=80, bbox_inches='tight')


def trans2midi():
    # Load audio
    (audio, _) = load_audio('./in.mp3', sr=sample_rate, mono=True)

    # Transcriptor
    transcriptor = PianoTranscription(device='cpu')  # 'cuda' | 'cpu'

    # Transcribe and write out to MIDI file
    transcribed_dict = transcriptor.transcribe(audio, 'in.mid')


def separate_audio(audio_path, outdir='./ests', model_path='./pretrained-model/', sr=44100):
    command = f"python -m xumx_slicq.predict --outdir {outdir} --no-cuda --model {model_path} --sr {sr} {audio_path}"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()
    print(f"Output saved in {outdir}")


def transform_audio(pitch, loudness, model_path='ddsp_pretrained_violin.ts'):
    model = torch.jit.load(model_path)
    audio = model(pitch, loudness)
    return audio


def identify_instrument(path):
   # https://github.com/jinzhaochaliang/Chinese-Musical-Instruments-Classification/blob/master/test_input/work.ipynb
   instrument = get_audio_result(path)
   return instrument

def selector():
    # TODO
    return 1

input = "./input.wav"

Visualization(input)
print("当前音频的分析结果已生成")
separate_audio(input)
ddsp_audio = transform_audio("./ests/voice.wav", loudness=30, model_path="./pth/note_F1=0.9677_pedal_F1=0.9186.pth")
not_voice_part = "./ests/erhu.wav"
instrument = identify_instrument("./ests/erhu.wav")
print("该音频主要乐器是：" + instrument + "！ ")
trans2midi()
a = read('in.mid')
a.clear_program_change()
a.change_instruments([selector(instrument)])
write(a, name='out.mid')
merge(ddsp_audio, "out.mid") # TODO


