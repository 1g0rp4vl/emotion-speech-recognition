import sys
import time

import numpy as np
import requests


def test_server():
	# Generate dummy audio data: 1 batch, 40000 samples (approx 2.5s at 16kHz)
	data = np.random.randn(1, 40000).astype(np.float32).tolist()

	payload = {'inputs': data}
	url = 'http://127.0.0.1:5000/invocations'

	print(f'Sending request to {url}...')

	max_retries = 10
	for i in range(max_retries):
		try:
			response = requests.post(
				url, json=payload, headers={'Content-Type': 'application/json'}
			)
			response.raise_for_status()
			print('Response received:')
			print(response.json())
			return True
		except requests.exceptions.ConnectionError:
			print(f'Connection refused, retrying ({i + 1}/{max_retries})...')
			time.sleep(2)
		except Exception as e:
			print(f'Error: {e}')
			if hasattr(e, 'response') and e.response is not None:
				print(e.response.text)
			return False

	print('Failed to connect to server after multiple retries.')
	return False


if __name__ == '__main__':
	if test_server():
		sys.exit(0)
	else:
		sys.exit(1)
