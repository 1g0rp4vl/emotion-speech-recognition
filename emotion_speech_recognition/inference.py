# import sys
# sys.path.append("C:/Users/kiraa/simpsons")
import os
import shutil
from pathlib import Path

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from datamodule import AudioDataset, label_mapping
from model import EmotionModel
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm


@hydra.main(version_base=None, config_path='../conf', config_name='config')
def main(cfg: DictConfig):
	"""
	Main inference function.

	Args:
		cfg (DictConfig): Hydra configuration.
	"""
	dir = Path(cfg.inference.dir)
	files = sorted(list(dir.rglob('*.wav')))
	if not files:
		print(f'No .wav files found in {dir}')
		return

	data = AudioDataset(
		files=files,
		mode='test',
		sample_rate=cfg.module.sample_rate,
		audio_duration=cfg.module.audio_duration,
	)

	logger = pl.loggers.MLFlowLogger(
		experiment_name='inference_emotion_recognition',
		tracking_uri=cfg.logger.tracking_uri,
	)

	model = EmotionModel.load_from_checkpoint(
		cfg.inference.ckpt,
		mode=cfg.module.mode,
		n_classes=cfg.module.n_classes,
		gamma=cfg.module.gamma,
	)
	model.eval()

	trainer = pl.Trainer(accelerator='auto', devices='auto', logger=logger)
	idx_to_label = {v: k for k, v in label_mapping.items()}
	dataloader = DataLoader(data, batch_size=cfg.inference.batch_size, shuffle=False)
	predictions = trainer.predict(model, dataloaders=dataloader)

	run_id = logger.run_id
	print(f'Logging to MLflow run: {run_id}')

	shutil.rmtree('predictions', ignore_errors=True)
	os.makedirs('predictions', exist_ok=True)
	for label in label_mapping:
		os.makedirs(f'predictions/{label}', exist_ok=True)

	for i, output in enumerate(tqdm(predictions)):
		logits = output
		probs = torch.softmax(logits, dim=1).numpy().flatten()
		y_pred = np.argmax(probs)
		predicted_label = idx_to_label[y_pred]

		file_path = files[i]
		filename = file_path.name

		caption = f'{predicted_label} ({np.max(probs):.2f}%)'
		print(f'File: {filename} -> {caption}')
		artifact_path = f'predictions/{predicted_label}'
		logger.experiment.log_artifact(run_id, str(file_path), artifact_path)
		shutil.copy(str(file_path), artifact_path)


if __name__ == '__main__':
	main()
