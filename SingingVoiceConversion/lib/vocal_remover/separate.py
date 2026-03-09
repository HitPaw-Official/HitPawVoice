import os, math,  torch, librosa, gc
import numpy as np
import soundfile as sf
import audioread, pydub
from scipy import signal
from . import nets, spec_utils
from .model_param_init import ModelParameters

VOCAL_STEM = 'Vocals'
INST_STEM = 'Instrumental'
OTHER_STEM = 'Other'
BASS_STEM = 'Bass'
DRUM_STEM = 'Drums'
GUITAR_STEM = 'Guitar'
PIANO_STEM = 'Piano'
SYNTH_STEM = 'Synthesizer'
STRINGS_STEM = 'Strings'
WOODWINDS_STEM = 'Woodwinds'
BRASS_STEM = 'Brass'
WIND_INST_STEM = 'Wind Inst'

NON_ACCOM_STEMS = (
            VOCAL_STEM,
            OTHER_STEM,
            BASS_STEM,
            DRUM_STEM,
            GUITAR_STEM,
            PIANO_STEM,
            SYNTH_STEM,
            STRINGS_STEM,
            WOODWINDS_STEM,
            BRASS_STEM,
            WIND_INST_STEM)


class SeperateVR():
    def __init__(self, device="cuda:0",
                 config='lib/vocal_remover/4band_v2_sn.json',
                 model_path='lib/vocal_remover/5_HP-Karaoke-UVR.pth'):
        self.device = device
        self.mp = ModelParameters(config)
        self.model_path = model_path
        self.model_samplerate = 44100
        self.wav_type_set = 'PCM_16'
        self.primary_stem = 'Instrumental'
        self.secondary_stem = 'Vocals'
        self.window_size = 512
        self.save_format = 'wav'
        self.mp3_bit_set = '320k'
        self.post_process_threshold = 0.2
        self.overlap = 0.25
        self.aggression_setting = 0.05
        self.aggressiveness = {'value': self.aggression_setting,
                               'split_bin': self.mp.param['band'][1]['crop_stop'],
                               'aggr_correction': self.mp.param.get('aggr_correction')}
        self.is_secondary_stem_only = False
        self.is_primary_stem_only = False
        self.high_end_process = 'None'
        self.batch_size = 1
        self.progress_value = 0

    def seperate(self, audio, export_path):
        audio_file_base = os.path.basename(audio).split('.')[0]
        nn_arch_sizes = [31191, 33966, 56817, 123821, 123812, 129605, 218409, 537238, 537227]

        model_size = math.ceil(os.stat(self.model_path).st_size / 1024)
        nn_arch_size = min(nn_arch_sizes, key=lambda x:abs(x-model_size))

        self.model_run = nets.determine_model_capacity(self.mp.param['bins'] * 2, nn_arch_size)
                        
        self.model_run.load_state_dict(torch.load(self.model_path, map_location="cpu")) 
        self.model_run.to(self.device) 
                    
        y_spec, v_spec = self.inference_vr(self.loading_mix(audio), self.device, self.aggressiveness)
            
        if not self.is_secondary_stem_only:
            primary_stem_path = os.path.join(export_path, f'{audio_file_base}_{self.primary_stem}.wav')
            self.primary_source = self.spec_to_wav(y_spec).T
            if not self.model_samplerate == 44100:
                self.primary_source = librosa.resample(self.primary_source.T, orig_sr=self.model_samplerate, target_sr=44100).T
                
            self.primary_source_map = self.final_process(primary_stem_path, self.primary_source, None, self.primary_stem, 44100)  

        if not self.is_primary_stem_only:
            secondary_stem_path = os.path.join(export_path, f'{audio_file_base}_{self.secondary_stem}.wav')
            self.secondary_source = self.spec_to_wav(v_spec).T
            if not self.model_samplerate == 44100:
                self.secondary_source = librosa.resample(self.secondary_source.T, orig_sr=self.model_samplerate, target_sr=44100).T
            
            self.secondary_source_map = self.final_process(secondary_stem_path, self.secondary_source, None, self.secondary_stem, 44100)

        gc.collect()
        torch.cuda.empty_cache()

        return primary_stem_path, secondary_stem_path

    def loading_mix(self, audio):

        X_wave, X_spec_s = {}, {}
        bands_n = len(self.mp.param['band'])
        
        audio_file = spec_utils.write_array_to_mem(audio, subtype=self.wav_type_set)
        is_mp3 = audio_file.endswith('.mp3') if isinstance(audio_file, str) else False

        for d in range(bands_n, 0, -1):
            bp = self.mp.param['band'][d]
        
            wav_resolution = bp['res_type']
        
            if d == bands_n: # high-end band
                X_wave[d], _ = librosa.load(audio_file, bp['sr'], False, dtype=np.float32, res_type=wav_resolution)
                X_spec_s[d] = spec_utils.wave_to_spectrogram(X_wave[d], bp['hl'], bp['n_fft'], self.mp, band=d, is_v51_model=False)
                    
                if not np.any(X_wave[d]) and is_mp3:
                    X_wave[d] = rerun_mp3(audio_file, bp['sr'])

                if X_wave[d].ndim == 1:
                    X_wave[d] = np.asarray([X_wave[d], X_wave[d]])
            else: # lower bands
                X_wave[d] = librosa.resample(X_wave[d+1], self.mp.param['band'][d+1]['sr'], bp['sr'], res_type=wav_resolution)
                X_spec_s[d] = spec_utils.wave_to_spectrogram(X_wave[d], bp['hl'], bp['n_fft'], self.mp, band=d, is_v51_model=False)

            if d == bands_n and self.high_end_process != 'none':
                self.input_high_end_h = (bp['n_fft']//2 - bp['crop_stop']) + (self.mp.param['pre_filter_stop'] - self.mp.param['pre_filter_start'])
                self.input_high_end = X_spec_s[d][:, bp['n_fft']//2-self.input_high_end_h:bp['n_fft']//2, :]

        X_spec = spec_utils.combine_spectrograms(X_spec_s, self.mp, is_v51_model=False)
        
        del X_wave, X_spec_s, audio_file

        return X_spec

    def inference_vr(self, X_spec, device, aggressiveness):
        def _execute(X_mag_pad, roi_size):
            X_dataset = []
            patches = (X_mag_pad.shape[2] - 2 * self.model_run.offset) // roi_size
            total_iterations = patches // self.batch_size
            for i in range(patches):
                start = i * roi_size
                X_mag_window = X_mag_pad[:, :, start:start + self.window_size]
                X_dataset.append(X_mag_window)

            X_dataset = np.asarray(X_dataset)
            self.model_run.eval()
            with torch.no_grad():
                mask = []
                for i in range(0, patches, self.batch_size):
                    self.progress_value += 1
                    if self.progress_value >= total_iterations:
                        self.progress_value = total_iterations
                    # self.set_progress_bar(0.1, 0.8/total_iterations*self.progress_value)
                    X_batch = X_dataset[i: i + self.batch_size]
                    X_batch = torch.from_numpy(X_batch).to(device)
                    pred = self.model_run.predict_mask(X_batch)

                    pred = pred.detach().cpu().numpy()
                    pred = np.concatenate(pred, axis=2)
                    mask.append(pred)

                
                mask = np.concatenate(mask, axis=2)
            return mask

        def postprocess(mask, X_mag, X_phase):
            is_non_accom_stem = False
            for stem in NON_ACCOM_STEMS:
                if stem == self.primary_stem:
                    is_non_accom_stem = True
                    
            mask = spec_utils.adjust_aggr(mask, is_non_accom_stem, aggressiveness)

            # if self.is_post_process:
            #     mask = spec_utils.merge_artifacts(mask, thres=self.post_process_threshold)

            y_spec = mask * X_mag * np.exp(1.j * X_phase)
            v_spec = (1 - mask) * X_mag * np.exp(1.j * X_phase)
        
            return y_spec, v_spec
        
        X_mag, X_phase = spec_utils.preprocess(X_spec)
        n_frame = X_mag.shape[2]
        pad_l, pad_r, roi_size = spec_utils.make_padding(n_frame, self.window_size, self.model_run.offset)
        X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
        X_mag_pad /= X_mag_pad.max()
        mask = _execute(X_mag_pad, roi_size)
        mask = mask[:, :, :n_frame]

        y_spec, v_spec = postprocess(mask, X_mag, X_phase)
        
        return y_spec, v_spec

    def spec_to_wav(self, spec):
        if self.high_end_process.startswith('mirroring') and isinstance(self.input_high_end, np.ndarray) and self.input_high_end_h:        
            input_high_end_ = spec_utils.mirroring(self.high_end_process, spec, self.input_high_end, self.mp)
            wav = spec_utils.cmb_spectrogram_to_wave(spec, self.mp, self.input_high_end_h, input_high_end_, is_v51_model=False)       
        else:
            wav = spec_utils.cmb_spectrogram_to_wave(spec, self.mp, is_v51_model=False)
            
        return wav
    
    def process_secondary_stem(self, stem_source, secondary_model_source=None, model_scale=None):
        if not self.is_secondary_model:
            if self.is_secondary_model_activated and isinstance(secondary_model_source, np.ndarray):
                secondary_model_scale = model_scale if model_scale else self.secondary_model_scale
                stem_source = spec_utils.average_dual_sources(stem_source, secondary_model_source, secondary_model_scale)

        return stem_source
    
    def final_process(self, stem_path, source, secondary_source, stem_name, samplerate):
        # source = self.process_secondary_stem(source, secondary_source)
        self.write_audio(stem_path, source, samplerate, stem_name=stem_name)
        
        return {stem_name: source}
    
    def write_audio(self, stem_path: str, stem_source, samplerate, stem_name=None):
        
        def save_audio_file(path, source):
            source = spec_utils.normalize(source, False)
            sf.write(path, source, samplerate, subtype=self.wav_type_set)

            save_format(path, self.save_format, self.mp3_bit_set)

        def save_with_message(stem_path, stem_name, stem_source):
            save_audio_file(stem_path, stem_source)
            
        save_with_message(stem_path, stem_name, stem_source)

        if stem_name == VOCAL_STEM:
            self.master_vocal_path = stem_path
    
    
def gather_sources(primary_stem_name, secondary_stem_name, secondary_sources: dict):
    
    source_primary = False
    source_secondary = False

    for key, value in secondary_sources.items():
        if key in primary_stem_name:
            source_primary = value
        if key in secondary_stem_name:
            source_secondary = value

    return source_primary, source_secondary
        
def prepare_mix(mix):
    
    audio_path = mix

    if not isinstance(mix, np.ndarray):
        mix, sr = librosa.load(mix, mono=False, sr=44100)
    else:
        mix = mix.T

    if isinstance(audio_path, str):
        if not np.any(mix) and audio_path.endswith('.mp3'):
            mix = rerun_mp3(audio_path)

    if mix.ndim == 1:
        mix = np.asfortranarray([mix,mix])

    return mix

def rerun_mp3(audio_file, sample_rate=44100):

    with audioread.audio_open(audio_file) as f:
        track_length = int(f.duration)

    return librosa.load(audio_file, duration=track_length, mono=False, sr=sample_rate)[0]

def save_format(audio_path, save_format, mp3_bit_set):
    
    if not save_format == 'wav':
        musfile = pydub.AudioSegment.from_wav(audio_path)
        
        if save_format == 'flac':
            audio_path_flac = audio_path.replace(".wav", ".flac")
            musfile.export(audio_path_flac, format="flac")  
        
        if save_format == 'mp3':
            audio_path_mp3 = audio_path.replace(".wav", ".mp3")
            try:
                musfile.export(audio_path_mp3, format="mp3", bitrate=mp3_bit_set, codec="libmp3lame")
            except Exception as e:
                print(e)
                musfile.export(audio_path_mp3, format="mp3", bitrate=mp3_bit_set)
        try:
            os.remove(audio_path)
        except Exception as e:
            print(e)
            
def pitch_shift(mix):
    new_sr = 31183

    # Resample audio file
    resampled_audio = signal.resample_poly(mix, new_sr, 44100)
    
    return resampled_audio

def list_to_dictionary(lst):
    dictionary = {item: index for index, item in enumerate(lst)}
    return dictionary

def loading_mix(X, mp):

    X_wave, X_spec_s = {}, {}
    
    bands_n = len(mp.param['band'])
    
    for d in range(bands_n, 0, -1):        
        bp = mp.param['band'][d]
        wav_resolution = 'polyphase'#bp['res_type']
    
        if d == bands_n: # high-end band
            X_wave[d] = X

        else: # lower bands
            X_wave[d] = librosa.resample(X_wave[d+1], mp.param['band'][d+1]['sr'], bp['sr'], res_type=wav_resolution)
            
        X_spec_s[d] = spec_utils.wave_to_spectrogram(X_wave[d], bp['hl'], bp['n_fft'], mp, band=d, is_v51_model=True)
        
        # if d == bands_n and is_high_end_process:
        #     input_high_end_h = (bp['n_fft']//2 - bp['crop_stop']) + (mp.param['pre_filter_stop'] - mp.param['pre_filter_start'])
        #     input_high_end = X_spec_s[d][:, bp['n_fft']//2-input_high_end_h:bp['n_fft']//2, :]

    X_spec = spec_utils.combine_spectrograms(X_spec_s, mp)
    
    del X_wave, X_spec_s

    return X_spec
