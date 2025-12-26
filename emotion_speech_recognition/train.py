import datetime
import os
import subprocess
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from datamodule import AudioDataModule
from model import EmotionModel
from omegaconf import DictConfig


class MetricsPlotterCallback(pl.Callback):
	"""
	Callback to plot training metrics at the end of training.
	"""

	def __init__(self, save_dir='plots'):
		"""
		Initialize the MetricsPlotterCallback.

		Args:
			save_dir (str): Directory to save plots.
		"""
		super().__init__()
		self.save_dir = save_dir
		self.metrics = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_F1': []}

	def on_train_epoch_end(self, trainer, pl_module):
		if 'train_loss' in trainer.callback_metrics:
			self.metrics['train_loss'].append(trainer.callback_metrics['train_loss'].item())

	def on_validation_epoch_end(self, trainer, pl_module):
		if trainer.sanity_checking:
			return

		for metric in ['val_loss', 'val_acc', 'val_F1']:
			if metric in trainer.callback_metrics:
				self.metrics[metric].append(trainer.callback_metrics[metric].item())

	def on_fit_end(self, trainer, pl_module):
		os.makedirs(self.save_dir, exist_ok=True)

		for metric_name, values in self.metrics.items():
			if not values:
				continue

			plt.figure()
			plt.plot(values, label=metric_name)
			plt.title(f'{metric_name} per epoch')
			plt.xlabel('Epoch')
			plt.ylabel('Value')
			plt.legend()
			plt.grid(True)

			filename = metric_name.lower() + '.png'
			plt.savefig(os.path.join(self.save_dir, filename))
			plt.close()


@hydra.main(version_base=None, config_path='../conf', config_name='config')
def main(cfg: DictConfig):
	"""
	Main training function.

	Args:
		cfg (DictConfig): Hydra configuration.
	"""

	timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
	save_dir = f'./{cfg.callbacks.dirpath}/run_{timestamp}/'

	try:
		commit_id = (
			subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
		)
	except Exception:
		commit_id = 'unknown'

	pl.seed_everything(cfg.trainer.seed)
	data_module = AudioDataModule(
		train_dir=Path(cfg.module.train_dir),
		val_dir=Path(cfg.module.val_dir),
		batch_size=cfg.module.batch_size,
		num_workers=cfg.module.num_workers,
		sample_rate=cfg.module.sample_rate,
		audio_duration=cfg.module.audio_duration,
	)
	model = EmotionModel(
		n_classes=cfg.module.n_classes, mode=cfg.module.mode, gamma=cfg.module.gamma
	)

	loggers = [
		pl.loggers.MLFlowLogger(
			experiment_name=cfg.logger.experiment_name,
			run_name=f'time:{timestamp}_hparams:[mode={cfg.module.mode},n_classes={cfg.module.n_classes},gamma={cfg.module.gamma}]_commit:{commit_id}',
			tracking_uri=cfg.logger.tracking_uri,
		)
	]

	callbacks = [
		pl.callbacks.LearningRateMonitor(logging_interval='step'),
		pl.callbacks.DeviceStatsMonitor(),
		pl.callbacks.EarlyStopping(
			monitor='val_loss',
			patience=5,
			mode='min',
		),
	]

	callbacks.append(
		pl.callbacks.ModelCheckpoint(
			dirpath=save_dir,
			filename=cfg.callbacks.filename,
			monitor='val_loss',
			save_top_k=1,
			every_n_epochs=1,
		)
	)

	callbacks.append(MetricsPlotterCallback(save_dir='plots'))

	trainer = pl.Trainer(
		max_epochs=cfg.trainer.epoch,
		log_every_n_steps=10,
		accelerator='auto',
		devices='auto',
		logger=loggers,
		callbacks=callbacks,
	)

	trainer.fit(model, datamodule=data_module)


if __name__ == '__main__':
	main()
