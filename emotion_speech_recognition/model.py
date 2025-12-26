import typing as tp

import pytorch_lightning as pl
import torch
from audio_classifier import AudioClassifier
from torch import nn
from torchmetrics import Accuracy, F1Score


class EmotionModel(pl.LightningModule):
	"""
	LightningModule for emotion recognition.
	"""

	def __init__(self, mode, n_classes, gamma):
		"""
		Initialize the EmotionModel.

		Args:
			mode (str): Optimization mode ('Base' or 'Better').
			n_classes (int): Number of output classes.
			gamma (float): Gamma value for exponential learning rate scheduler.
		"""
		super().__init__()
		self.save_hyperparameters()
		self.model = AudioClassifier(n_classes=n_classes)
		self.criterion = nn.CrossEntropyLoss()
		self.f1_score = F1Score(task='multiclass', num_classes=n_classes, average='macro')
		self.accuracy = Accuracy(task='multiclass', num_classes=n_classes)

	def forward(self, x):
		return self.model(x)

	def configure_optimizers(self):
		"""
		Choose what optimizers and learning-rate schedulers to use.
		"""
		if self.hparams.mode == 'Base':
			optimizer = torch.optim.Adam(self.parameters())
			return {'optimizer': optimizer}
		elif self.hparams.mode == 'Better':
			optimizer = torch.optim.Adam(self.parameters())
			scheduler = torch.optim.lr_scheduler.ExponentialLR(
				optimizer=optimizer, gamma=self.hparams.gamma
			)
			return {'optimizer': optimizer, 'lr_scheduler': scheduler}
		else:
			raise ValueError("Mode should be one of 'Base' or 'Better'.")

	def training_step(self, batch: tp.Any):
		input, label = batch
		output = self.forward(input)
		loss = self.criterion(output, label)
		self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)

		return loss

	def validation_step(self, batch: tp.Any):
		input, label = batch
		output = self.forward(input)
		loss = self.criterion(output, label)

		pred = torch.argmax(output, 1)
		acc = self.accuracy(pred, label)

		metric = self.f1_score(pred, label)

		self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
		self.log('val_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
		self.log('val_F1', metric, prog_bar=True, on_step=False, on_epoch=True)

		return {'val_loss': loss, 'val_acc': acc, 'val_F1': metric}

	def predict_step(self, input: tp.Any):
		return self.forward(input).cpu()
