import hydra
import mlflow
import numpy as np
import onnx
import onnxruntime as ort
from mlflow.models.signature import infer_signature
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path='conf', config_name='config')
def register_model(cfg: DictConfig):
	onnx_model_path = 'model.onnx'
	print(f'Loading ONNX model from: {onnx_model_path}')

	onnx_model = onnx.load(onnx_model_path)

	input_len = int(cfg.module.sample_rate * cfg.module.audio_duration)
	input_example = np.random.randn(1, input_len).astype(np.float32)
	ort_session = ort.InferenceSession(onnx_model_path)
	input_name = ort_session.get_inputs()[0].name
	output_name = ort_session.get_outputs()[0].name

	signature = infer_signature(
		input_example, ort_session.run([output_name], {input_name: input_example})[0]
	)
	mlflow.set_tracking_uri(cfg.logger.tracking_uri)
	mlflow.set_experiment('model_registry')

	with mlflow.start_run() as run:
		print(f'Logging ONNX model to MLflow run: {run.info.run_id}')

		model_info = mlflow.onnx.log_model(
			onnx_model=onnx_model,
			name='emotion_model_onnx',
			signature=signature,
			input_example=input_example,
			pip_requirements=['onnx', 'onnxruntime', 'numpy'],
		)

		print('Model logged successfully!')
		print('To serve this model, run:')
		print(
			f'MLFLOW_TRACKING_URI={cfg.logger.tracking_uri} mlflow models serve -m \
                {model_info.model_uri} -p 5000 --no-conda'
		)


if __name__ == '__main__':
	register_model()
