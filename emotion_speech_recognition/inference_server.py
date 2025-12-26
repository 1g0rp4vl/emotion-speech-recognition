import hydra
import numpy as np
import onnxruntime as ort
import uvicorn
from fastapi import FastAPI, HTTPException
from omegaconf import DictConfig
from pydantic import BaseModel
from scipy.special import softmax as scipy_softmax

app = FastAPI(title='Emotion Speech Recognition Inference Server')

try:
	ort_session = ort.InferenceSession('model.onnx')
except Exception as e:
	print(f'Error loading ONNX model from model.onnx: {e}')
	ort_session = None

LABEL_MAPPING = {
	0: 'neutral',
	1: 'calm',
	2: 'happy',
	3: 'sad',
	4: 'angry',
	5: 'fearful',
	6: 'disgust',
	7: 'surprised',
}

BATCH_SIZE = 8


class InputData(BaseModel):
	data: list[list[float]]


class PredictionResult(BaseModel):
	label: str
	probabilities: dict[str, float]


class BatchPredictionResponse(BaseModel):
	predictions: list[PredictionResult]


ONNX_MODEL_PATH = 'model.onnx'


@app.get('/health')
def health_check():
	"""Check if the server and model are ready."""
	if ort_session is None:
		raise HTTPException(status_code=503, detail='Model not loaded')
	return {'status': 'healthy', 'model_path': ONNX_MODEL_PATH}


@app.post('/predict', response_model=BatchPredictionResponse)
def predict(input_data: InputData):
	"""
	Run inference on the input data.

	Args:
		input_data (InputData): Batch of preprocessed audio features.

	Returns:
		BatchPredictionResponse: List of predictions with probabilities.
	"""
	if ort_session is None:
		raise HTTPException(status_code=503, detail='Model not loaded')

	try:
		input_name = ort_session.get_inputs()[0].name
		input_data = np.array(input_data.data, dtype=np.float32)
		outputs = []
		for i in range(0, len(input_data), BATCH_SIZE):
			outputs.append(
				ort_session.run(
					None,
					{
						input_name: input_data[
							i * BATCH_SIZE : min((i + 1) * BATCH_SIZE, len(input_data))
						]
					},
				)[0]
			)

		outputs = np.vstack(outputs)
		probs = scipy_softmax(outputs, axis=1)
		results = []

		for i in range(len(probs)):
			predicted_idx = np.argmax(probs[i])
			predicted_label = LABEL_MAPPING.get(predicted_idx, 'unknown')

			prob_dict = {LABEL_MAPPING[idx]: float(prob) for idx, prob in enumerate(probs[i])}

			results.append(PredictionResult(label=predicted_label, probabilities=prob_dict))

		return BatchPredictionResponse(predictions=results)

	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e)) from e


@hydra.main(version_base=None, config_path='../conf', config_name='config')
def main(cfg: DictConfig):
	"""
	Main function to start the inference server.

	Args:
		cfg (DictConfig): Hydra configuration.
	"""
	global ONNX_MODEL_PATH
	global BATCH_SIZE
	BATCH_SIZE = cfg.inference.batch_size
	ONNX_MODEL_PATH = cfg.inference.onnx_path
	uvicorn.run(app, port=cfg.inference.port, host=cfg.inference.host)


if __name__ == '__main__':
	main()
