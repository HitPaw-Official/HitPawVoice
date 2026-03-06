import numpy as np
from madmom.features import DBNDownBeatTrackingProcessor
from log_spect import LOG_SPECT
import librosa
import onnxruntime


class BeatNet:
    def __init__(self, model_path, mode='offline', inference_model='DBN', plot=["activations"], thread=False, device='cpu'):
        self.mode = mode
        self.inference_model = inference_model
        self.plot= plot
        self.thread = thread
        self.device = device
        if plot and thread:
            raise RuntimeError('Plotting cannot be accomplished in the threading mode')
        self.sample_rate = 22050
        self.log_spec_sample_rate = self.sample_rate
        self.log_spec_hop_length = int(20 * 0.001 * self.log_spec_sample_rate)
        self.log_spec_win_length = int(64 * 0.001 * self.log_spec_sample_rate)
        self.proc = LOG_SPECT(sample_rate=self.log_spec_sample_rate, win_length=self.log_spec_win_length,
                             hop_size=self.log_spec_hop_length, n_bands=[24], mode = self.mode)
        self.estimator = DBNDownBeatTrackingProcessor(beats_per_bar=[2, 3, 4], fps=50)

        self.model = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
                               
    def process(self, audio_path=None):   
        preds = self.activation_extractor_onnx(audio_path)    # Using BeatNet causal Neural network to extract activations
        output = self.estimator(preds)  # Using DBN offline inference to infer beat/downbeats
        return output
                
    def activation_extractor_onnx(self, audio_path):
        if isinstance(audio_path, str):
            audio, _ = librosa.load(audio_path, sr=self.sample_rate)
        elif len(np.shape(audio_path))>1:
            audio = np.mean(audio_path ,axis=1)
        else:
            audio = audio_path
        feats = self.proc.process_audio(audio).T
        feats = np.expand_dims(feats, 0)

        onnx_input = {self.model.get_inputs()[0].name: feats}
        preds = self.model.run(None, onnx_input)[0]
        preds = np.transpose(preds[:2, :])
        return preds

