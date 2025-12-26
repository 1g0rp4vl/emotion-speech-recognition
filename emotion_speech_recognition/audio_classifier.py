import torch.nn as nn
import torchaudio


class AudioClassifier(nn.Module):
	"""
	Audio classifier model using Wav2Vec2 as a feature extractor.
	"""

	def __init__(self, n_classes):
		"""
		Initialize the AudioClassifier.

		Args:
			n_classes (int): Number of output classes.
		"""
		super().__init__()

		try:
			bundle = torchaudio.pipelines.WAV2VEC2_BASE
			self.wav2vec = bundle.get_model()
			for param in self.wav2vec.parameters():
				param.requires_grad = False
		except Exception as e:
			print(f'Warning: Could not load Wav2Vec2 model: {e}')
			self.wav2vec = None

		self.classifier = nn.Sequential(
			nn.Linear(768, 512), nn.ReLU(), nn.Dropout(0.1), nn.Linear(512, n_classes)
		)

	def forward(self, x):
		if x.ndim == 3:
			x = x.squeeze(1)

		if self.wav2vec is None:
			raise RuntimeError('Wav2Vec2 model not loaded.')

		features, _ = self.wav2vec(x)
		embedding = features.mean(dim=1)
		return self.classifier(embedding)
