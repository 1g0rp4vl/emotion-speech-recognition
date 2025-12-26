import sys

import hydra
import requests
from modern_audio_features import RawAudioProcessor
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path='../conf', config_name='config')
def main(cfg: DictConfig):
	"""
	Main function to run the inference client.

	Args:
		cfg (DictConfig): Hydra configuration.
	"""
	file_path = cfg.get('file_path')
	if not file_path:
		print(
			'Error: Please provide file_path. Usage: python client.py file_path=/path/to/audio.wav'
		)
		return

	print(f'Processing {file_path}...')

	processor = RawAudioProcessor(
		sample_rate=cfg.module.sample_rate, audio_duration=cfg.module.audio_duration
	)

	server_url = f'http://{cfg.inference.host}:{cfg.inference.port}/predict'

	try:
		features_tensor = processor.get_features_pipeline(file_path)
		features_list = features_tensor.unsqueeze(0).tolist()
		request_data = {'data': features_list}

		print(f'Sending request to {server_url}...')
		response = requests.post(server_url, json=request_data)

		if response.status_code == 200:
			result = response.json()
			predictions = result.get('predictions', [])

			if predictions:
				pred = predictions[0]
				print('\nPrediction Result:')
				print(f'  Emotion: {pred["label"]}')
				print('  Probabilities:')
				for label, prob in sorted(pred['probabilities'].items()):
					print(f'    {label}: {prob:.4f}')
			else:
				print('No predictions returned.')
		else:
			print(f'Error: Server returned status code {response.status_code}')
			print(response.text)

	except Exception as e:
		print(f'Error: {e}')
		sys.exit(1)


if __name__ == '__main__':
	main()
