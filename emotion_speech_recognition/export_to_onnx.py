import hydra

# import hydra.utils
import numpy as np
import onnxruntime as ort
import torch
import torch.onnx
from model import EmotionModel
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path='../conf', config_name='config')
def convert_to_onnx(cfg: DictConfig):
	"""
	Convert the trained model to ONNX format.

	Args:
		cfg (DictConfig): Hydra configuration.
	"""
	pl_model = EmotionModel.load_from_checkpoint(
		cfg.inference.ckpt,
		mode=cfg.module.mode,
		n_classes=cfg.module.n_classes,
		gamma=cfg.module.gamma,
		map_location='cpu',
	)
	model = pl_model.model
	model.eval()

	dummy_input = torch.randn(1, int(cfg.module.sample_rate * cfg.module.audio_duration))

	torch.onnx.export(
		model,
		dummy_input,
		cfg.inference.onnx_path,
		dynamo=False,
		export_params=True,
		input_names=['PREPROCESSED_AUDIO'],
		output_names=['OUTPUT'],
		dynamic_axes={'PREPROCESSED_AUDIO': {0: 'batch_size'}, 'OUTPUT': {0: 'batch_size'}},
	)

	print('Model successfully converted to ONNX')

	ort_sess = ort.InferenceSession(cfg.inference.onnx_path)
	ort_sess.run(None, {'PREPROCESSED_AUDIO': dummy_input.numpy().astype(np.float32)})
	print('ONNX model check passed!')


if __name__ == '__main__':
	convert_to_onnx()
