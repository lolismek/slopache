import torch, numpy as np, scipy.io.wavfile as wav
from transformers import VitsModel, AutoTokenizer
m = VitsModel.from_pretrained("facebook/mms-tts-ron")
tk = AutoTokenizer.from_pretrained("facebook/mms-tts-ron")
t = "Astăzi mănânc pâine cu brânză și mâine vând mărar în târg."
x = tk(t, return_tensors="pt")
with torch.no_grad():
    y = m(**x).waveform.squeeze().cpu().numpy()
sr = m.config.sampling_rate
wav.write("/ephemeral/slopache/outputs/voice/mms_ron.wav", sr, (y*32767).astype(np.int16))
print("OK sr=", sr, "samples=", len(y), "dur=", round(len(y)/sr,2))
