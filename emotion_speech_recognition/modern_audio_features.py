import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.transforms as T


class RawAudioProcessor:
	"""
	Preprocesses audio to raw waveform for Wav2Vec2.
	"""

	def __init__(self, sample_rate, audio_duration):
		"""
		Initialize the RawAudioProcessor.

		Args:
			sample_rate (int): Target sample rate.
			audio_duration (float): Target duration of audio clips.
		"""
		self.sample_rate = sample_rate
		self.audio_duration = audio_duration

	def get_features_pipeline(self, path):
		"""
		Process an audio file to raw waveform.

		Args:
			path (str): Path to the audio file.

		Returns:
			Tensor: Processed audio waveform.
		"""
		waveform_np, sr = sf.read(path, always_2d=True)
		waveform = torch.from_numpy(waveform_np.T).float()

		if sr != self.sample_rate:
			resampler = T.Resample(sr, self.sample_rate)
			waveform = resampler(waveform)

		if waveform.shape[0] > 1:
			waveform = waveform.mean(dim=0, keepdim=True)

		target_len = int(self.sample_rate * self.audio_duration)
		current_len = waveform.shape[1]
		if current_len < target_len:
			waveform = F.pad(waveform, (0, target_len - current_len))
		else:
			waveform = waveform[:, :target_len]

		return waveform.squeeze(0)
