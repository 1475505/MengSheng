> to midi开源项目：[bytedance/piano_transcription (github.com)](https://github.com/bytedance/piano_transcription)

### tips

远端运行示例：https://colab.research.google.com/github/qiuqiangkong/piano_transcription_inference/blob/master/resources/inference.ipynb

### linux环境配置

安装依赖项：

```shell
torch
#包体比较大，记得换源。
```

然后

```shell
pip install piano_transcription_inference
```

### 工作代码

```python
from piano_transcription_inference import PianoTranscription, sample_rate, load_audio

# Load audio
(audio, _) = load_audio('in.mp3', sr=sample_rate, mono=True)

# Transcriptor
transcriptor = PianoTranscription(device='cpu')    # 'cuda' | 'cpu'

# Transcribe and write out to MIDI file
transcribed_dict = transcriptor.transcribe(audio, 'in.mid')
```

笔者在linux环境下跑了一下，结果已打包。**不需要使用新环境**。

首次运行需要约20min的训练时间，会在/home生成172MB的数据文件。转换会有一定的耗时。

**windows环境下的具体配置与linux下有所区别，请搜索对应的安装教程。遇到奇怪的报错（如可能会遇到路径`/` 和`\\`的区别问题），根据错误信息搜索解决。**

网易云音乐mp3解析下载工具：flvcd
