import os
import shutil
import subprocess
import zipfile

import hydra
from omegaconf import DictConfig


def download_data(data_root, zip_file, dataset_url):
	"""
	Download the dataset.

	Args:
		data_root (str): Root directory for data.
		zip_file (str): Path to the zip file.
		dataset_url (str): URL of the dataset.
	"""
	if not os.path.exists(data_root):
		os.makedirs(data_root)

	if os.path.exists(zip_file):
		print(f'{zip_file} already exists, skipping download.')
		return

	print(f'Downloading dataset to {zip_file}...')
	try:
		subprocess.run(['curl', '-L', '-o', zip_file, dataset_url], check=True)
	except subprocess.CalledProcessError as e:
		print(f'Error downloading data: {e}')
		exit(1)


def extract_data(zip_file, extract_dir):
	"""
	Extract the dataset.

	Args:
		zip_file (str): Path to the zip file.
		extract_dir (str): Directory to extract to.
	"""
	if not os.path.exists(extract_dir):
		os.makedirs(extract_dir)

	print(f'Extracting {zip_file} to {extract_dir}...')
	try:
		with zipfile.ZipFile(zip_file, 'r') as zip_ref:
			zip_ref.extractall(extract_dir)
	except zipfile.BadZipFile:
		print('Error: The downloaded file is not a valid zip file.')
		exit(1)


def organize_data(data_root, extract_dir, emotions_mapping):
	"""
	Organize the dataset into train and val splits.

	Args:
		data_root (str): Root directory for data.
		extract_dir (str): Directory where data was extracted.
		emotions_mapping (dict): Mapping of emotion codes to names.
	"""
	print('Organizing data into train and val...')

	train_dir = os.path.join(data_root, 'train')
	val_dir = os.path.join(data_root, 'val')

	for split_dir in [train_dir, val_dir]:
		for emotion in emotions_mapping.values():
			os.makedirs(os.path.join(split_dir, emotion), exist_ok=True)

	for item in os.listdir(extract_dir):
		if item.startswith('Actor_'):
			actor_dir = os.path.join(extract_dir, item)
			if not os.path.isdir(actor_dir):
				continue

			actor_id = int(item.split('_')[1])

			target_base = train_dir if actor_id <= 18 else val_dir
			for filename in os.listdir(actor_dir):
				if not filename.endswith('.wav'):
					continue

				emotion_name = emotions_mapping.get(filename.split('.')[0].split('-')[2])
				if emotion_name:
					src = os.path.join(actor_dir, filename)
					dst = os.path.join(target_base, emotion_name, filename)
					shutil.move(src, dst)
				shutil.rmtree(actor_dir, ignore_errors=True)

	shutil.rmtree(extract_dir, ignore_errors=True)


def check_dataset_ready(data_root, emotions_mapping):
	"""
	Check if the dataset is already prepared.

	Args:
		data_root (str): Root directory for data.
		emotions_mapping (dict): Mapping of emotion codes to names.

	Returns:
		bool: True if dataset is ready, False otherwise.
	"""
	train_dir = os.path.join(data_root, 'train')
	val_dir = os.path.join(data_root, 'val')

	for split_dir in [train_dir, val_dir]:
		for emotion in emotions_mapping.values():
			emotion_dir = os.path.join(split_dir, emotion)
			if not os.path.exists(emotion_dir) or not os.listdir(emotion_dir):
				return False
	return True


def prepare_data(data_root, zip_file, dataset_url, extract_dir, emotions_mapping, use_dvc):
	"""
	Prepare the dataset.

	Args:
		data_root (str): Root directory for data.
		zip_file (str): Path to the zip file.
		dataset_url (str): URL of the dataset.
		extract_dir (str): Directory to extract to.
		emotions_mapping (dict): Mapping of emotion codes to names.
		use_dvc (bool): Whether to use DVC for data pulling.
	"""
	if check_dataset_ready(data_root, emotions_mapping):
		print('Dataset is already prepared, skipping download and extraction.')
		return
	if use_dvc:
		try:
			subprocess.run(['dvc', 'pull'], check=True)
		except subprocess.CalledProcessError as e:
			print(f'Error pulling data with DVC: {e}')
			exit(1)
	else:
		download_data(data_root, zip_file, dataset_url)
		extract_data(zip_file, extract_dir)
		organize_data(data_root, extract_dir, emotions_mapping)


@hydra.main(version_base=None, config_path='../conf', config_name='config')
def main(cfg: DictConfig):
	"""
	Prepare the dataset for training and validation.

	Args:
		cfg (DictConfig): Hydra configuration.
	"""
	prepare_data(
		data_root=cfg.prepare_data.data_root,
		zip_file=cfg.prepare_data.zip_file,
		dataset_url=cfg.prepare_data.dataset_url,
		extract_dir=cfg.prepare_data.extract_dir,
		emotions_mapping=cfg.prepare_data.emotions_mapping,
		use_dvc=cfg.prepare_data.use_dvc,
	)


if __name__ == '__main__':
	main()
