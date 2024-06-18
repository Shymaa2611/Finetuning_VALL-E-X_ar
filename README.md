# VALL-E X: Multilingual Text-to-Speech Synthesis and Voice Cloning 🔊
An open source implementation of Microsoft's [VALL-E X](https://arxiv.org/pdf/2303.03926) zero-shot TTS model.<br>
![vallex-framework](/images/vallex_framework.jpg "VALL-E X framework")

VALL-E X is an amazing multilingual text-to-speech (TTS) model proposed by Microsoft. While Microsoft initially publish in their research paper, they did not release any code or pretrained models. Recognizing the potential and value of this technology, our team took on the challenge to reproduce the results and train our own model. We are glad to share our trained VALL-E X model with the community, allowing everyone to experience the power next-generation TTS! 🎧
<br>
<br>


## Dataset
 - MGB2 <br/> 
   The second edition of the Multi-Genre Broadcast (MGB-2) Challenge is an evaluation of speech recognition and lightly supervised alignment using TV recordings in Arabic.
   The speech data is broad and multi-genre, spanning the whole range of TV output, and represents a challenging task for speech technology.
  - Data provided includes:<br/>
    Approximately 1,200 hours of Arabic broadcast data, obtained from about 4,000 programmes broadcast on Aljazeera Arabic TV channel over a span of 10 years, from 2005 until September 2015.Time-aligned transcription as an output from light supervised alignment, with a varying quality of human transcription for the whole episode.More than 110 million words of Aljazeera.net website collected between 2004, and the year of 2011.




### Prepare Data
  - python mgb2.py

### train 
 - python VALL_E_X_finetuning.py


## Usage in Python

```python
from utils.generation import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
from IPython.display import Audio
from utils.prompt_making import make_prompt
preload_models()
make_prompt(name="paimon", audio_prompt_path="paimon_prompt.wav")
text_prompt = """
مرحبا كيف حالكم اليوم 
"""
audio_array = generate_audio(text_prompt, prompt="paimon")
write_wav("vallex_generation.wav", SAMPLE_RATE, audio_array)
Audio(audio_array, rate=SAMPLE_RATE)
```
