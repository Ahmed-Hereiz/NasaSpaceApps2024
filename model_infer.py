import torchaudio.transforms as T
import torch
import torch.nn.functional as F
from obspy import read
from model_arch import VitConfig, VitModel
from scipy.signal import butter, filtfilt



MAX_LENGTH = 572427
def bandpass_filter(signal, sampling_rate,lowcut=0.01, highcut=1.1, order=5):
    nyquist = 0.5 * sampling_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band', analog=False)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal



class VitPipeLine:
  def __init__(
      self,
      band=False,
      n_fft=254,
      win_len=254,
      max_feat_length=4508,
      **kwargs
  ):
      self.band = band
      self.spec_transform = T.Spectrogram(n_fft, win_len)
      self.max_feat_length = max_feat_length

  def __call__(self, data, mode='lunar'):
      if self.band:
        sr = 6.625 if mode == 'lunar' else 20.0
        data = bandpass_filter(data, sr, **self.kwargs)
      data = self.spec_transform(data)
      if data.shape[-1] < self.max_feat_length:
          data = F.pad(data, (0, self.max_feat_length - data.shape[-1]))
      return data


def from_ms2data(mseed_data):

    if isinstance(mseed_data, str):
      mseed_data = read(mseed_data)

    tr = mseed_data[0]
    st = tr.stats.start_time.datetime
    mode = tr.stats.sampling_rate
    vel = tr.data
    return vel, mode, st


class VitInference:
   def __init__(self, model_path, data_max_length,**kwargs):
      self.data_pipe = VitPipeLine(**kwargs)
      config = VitConfig(img_size=(4508,))
      config.acts_final = "ReLU"
      self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      self.model = VitModel(config).to(self.dev)
      self.model.infer()
      self.model.load_state_dict(torch.load(model_path, map_location=self.dev, weights_only=True))
      self.data_max_length = data_max_length
      

   def __call__(self, vel, mode=None):
       
       if mode is None:
         vel, mode, st = from_ms2data(vel)
       vel = torch.tensor(vel).to(torch.float32)
       sr = 6.625 if mode == "lunar" else 20.0  
       feats = self.data_pipe(vel)
       feats = feats.unsqueeze(0)
       with torch.no_grad():
         output = self.model(feats.to(self.dev)).cpu().squeeze().item()
       output = self.data_max_length * output / sr 
       return output