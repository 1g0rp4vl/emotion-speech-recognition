import pathlib
import random

import pytorch_lightning as pl
import torch
from modern_audio_features import RawAudioProcessor
from torch.utils.data import DataLoader, Dataset

label_mapping = {
	'neutral': 0,
	'calm': 1,
	'happy': 2,
	'sad': 3,
	'angry': 4,
	'fearful': 5,
	'disgust': 6,
	'surprised': 7,
}


class AudioDataset(Dataset):
	"""
	Dataset for loading and processing audio files.
	"""

	def __init__(self, files, sample_rate, audio_duration, mode):
		"""
		Initialize the AudioDataset.

		Args:
			files (list): List of file paths.
			sample_rate (int): Target sample rate.
			audio_duration (float): Target duration of audio clips.
			mode (str): Dataset mode ('train', 'val', 'test').
		"""
		super().__init__()
		self.files = sorted(files)
		self.mode = mode
		self.feature_extractor = RawAudioProcessor(
			sample_rate=sample_rate, audio_duration=audio_duration
		)
		DATA_MODES = ['train', 'val', 'test']

		if self.mode not in DATA_MODES:
			print(f'{self.mode} is not correct; correct modes: {DATA_MODES}')
			raise NameError

		self.len_ = len(self.files)

		if self.mode != 'test':
			self.labels = [label_mapping[path.parent.name] for path in self.files]

	def __len__(self):
		return self.len_

	def load_sample(self, file):
		"""
		Load and process an audio sample.

		Args:
			file (Path): Path to the audio file.

		Returns:
			Tensor: Processed audio features.
		"""
		features = self.feature_extractor.get_features_pipeline(file)
		return features

	def augment(self, waveform):
		"""
		Apply data augmentation to the waveform.

		Args:
			waveform (Tensor): Input audio waveform.

		Returns:
			Tensor: Augmented waveform.
		"""
		# Add Gaussian Noise
		if random.random() < 0.5:
			noise_level = 0.005
			waveform = waveform + torch.randn_like(waveform) * noise_level

		# Gain (Volume) Perturbation
		if random.random() < 0.5:
			gain_level = 0.2  # +/- 10%
			gain = 1.0 + (random.random() - 0.5) * gain_level
			waveform = waveform * gain

		# Time Shift (Roll)
		if random.random() < 0.5:
			shift_max = int(waveform.shape[0] * 0.1)  # 10% shift
			shift = random.randint(-shift_max, shift_max)
			waveform = torch.roll(waveform, shifts=shift, dims=0)

		return waveform

	def __getitem__(self, index):
		features = self.load_sample(self.files[index])

		if self.mode == 'train':
			features = self.augment(features)

		features = features.unsqueeze(0)
		if self.mode != 'test':
			label = self.labels[index]
			return features, label
		return features


class AudioDataModule(pl.LightningDataModule):
	"""
	LightningDataModule for handling audio data.
	"""

	def __init__(
		self,
		train_dir: pathlib.Path,
		val_dir: pathlib.Path,
		batch_size: int,
		num_workers: int,
		sample_rate: int,
		audio_duration: float,
	):
		"""
		Initialize the AudioDataModule.

		Args:
			train_dir (pathlib.Path): Directory containing training data.
			val_dir (pathlib.Path): Directory containing validation data.
			batch_size (int): Batch size for data loaders.
			num_workers (int): Number of workers for data loaders.
			sample_rate (int): Sample rate for audio processing.
			audio_duration (float): Duration of audio clips.
		"""
		super().__init__()
		self.train_dir = train_dir
		self.val_dir = val_dir
		self.batch_size = batch_size
		self.num_workers = num_workers
		self.sample_rate = sample_rate
		self.audio_duration = audio_duration

	def setup(self, stage=None):
		self.train_files = sorted(list(self.train_dir.rglob('*.wav')))
		self.val_files = sorted(list(self.val_dir.rglob('*.wav')))
		self.train_dataset = AudioDataset(
			self.train_files,
			sample_rate=self.sample_rate,
			audio_duration=self.audio_duration,
			mode='train',
		)
		self.val_dataset = AudioDataset(
			self.val_files,
			sample_rate=self.sample_rate,
			audio_duration=self.audio_duration,
			mode='val',
		)

	def train_dataloader(self):
		return DataLoader(
			self.train_dataset,
			batch_size=self.batch_size,
			shuffle=True,
			num_workers=self.num_workers,
			persistent_workers=self.num_workers > 0,
		)

	def val_dataloader(self):
		return DataLoader(
			self.val_dataset,
			batch_size=self.batch_size,
			shuffle=False,
			num_workers=self.num_workers,
			persistent_workers=self.num_workers > 0,
		)
