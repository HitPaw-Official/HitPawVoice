import os, subprocess
import torch
import argparse
import time
import numpy as np
import librosa

from script.vc_infer_pipeline import VC
from lib.infer_pack.models import SynthesizerTrnMs768NSFsid, SynthesizerTrnMs768NSFsid_nono
from lib.vocal_remover.separate import SeperateVR
from demo_se import SE
from script.utils import load_audio
from script.config import Config
from fairseq import checkpoint_utils
from scipy.io import wavfile


def vocal_remove(input_path, output_path=None):
    vr = SeperateVR("cuda:1")
    if os.path.isdir(input_path):
        for f in os.listdir(input_path):
            spath = os.path.join(input_path, f)
            for audio in  os.listdir(spath):
                audio_path = os.path.join(spath, audio)
                output_path = os.path.dirname(audio_path).replace('1_base', '2_vocals')
                os.makedirs(output_path, exist_ok=True)
                primary_stem_path, secondary_stem_path = vr.seperate(audio_path, output_path)
    else:
        if os.path.exists(input_path):
            if not output_path:
                output_path = os.path.dirname(input_path)
            primary_stem_path, secondary_stem_path = vr.seperate(input_path, output_path)
    
    return primary_stem_path, secondary_stem_path


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, default="data/bug/4.mp3", help="input dir or audio path")
    parser.add_argument("--output", type=str, default="data/outputs/bug", help="output dir")
    parser.add_argument("--checkpoint", type=str, default="checkpoint_svc_single_1.0", help="model dir")
    parser.add_argument("--name", type=str, default='AmyLee', help="preson name")
    parser.add_argument("--sid", type=int, default=0, help="speak id")
    parser.add_argument("--filter_radius", type=int, default=3, help="[0, 7]")
    parser.add_argument("--rms_mix_rate", type=float, default=0.25, help="[0, 1]")
    parser.add_argument("--protect", type=float, default=0.3, help="[0, 0.5]")
    parser.add_argument("--f0_up_key", type=int, default=6, help="f0 up key value.male—>female:12,female—>male:-12")
    parser.add_argument("--index_rate", type=int, default=0.7, help="feature merge")
    parser.add_argument("--f0_method", type=str, default='pm', help="f0 up key value, 'pm', 'harvest', 'rmvpe'")
    parser.add_argument("--resample", type=int, default=48000, help="resample sample rate")
    parser.add_argument("--device", type=str, default="cuda:1", help="devide")
    parser.add_argument("--half", type=bool, default=True, help="model type")

    return parser.parse_args()


class VCInfer(object):
    def __init__(self, arg, checkpoint, name, device='cuda:0') -> None:
        self.arg = arg
        self.version = "v2"
        self.config = Config()
        self.device = device
        self.name = name
        self.name_dir = os.path.join(checkpoint, self.name)
        self.load_huber()
        self.vc = VC(48000, self.device, self.config)
        self.vr = SeperateVR(self.device)
        self.se = SE()

    def load_huber(self):
        models, _, _ = checkpoint_utils.load_model_ensemble_and_task(["weights/hubert_base.pt"], suffix="")
        hubert_model = models[0]
        hubert_model = hubert_model.to(self.device)
        if self.arg.half:
            hubert_model = hubert_model.half()
        else:
            hubert_model = hubert_model.float()
        self.hubert = hubert_model.eval()

    def load_vc(self):
        model_path = os.path.join(self.name_dir, self.name + '.pth')
        print("loading %s" % model_path)
        cpt = torch.load(model_path, map_location="cpu")
        self.tgt_sr = cpt["config"][-1]  # 采样率
        cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
        if_f0 = cpt.get("f0", 1)
        self.if_f0 = if_f0
        if self.if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=self.arg.half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
        del net_g.enc_q  # 不加这一行清不干净，真奇葩
        net_g.load_state_dict(cpt["weight"], strict=False)
        net_g.eval().to(self.device)
        if self.arg.half:
            self.net_g = net_g.half()
        else:
            self.net_g = net_g.float()

        del cpt
        torch.cuda.empty_cache()


    def vc_single(self, input_audio, f0_up_key, index_rate):
        audios = []
        os.makedirs(self.arg.output, exist_ok=True)
        if os.path.isdir(input_audio):
            for audio in [os.path.join(input_audio, i) for i in os.listdir(input_audio)]:
                audios.append(audio)
        else:
            audios.append(input_audio)
        for i_audio in audios:
            primary_stem_path, secondary_stem_path = self.vr.seperate(i_audio, self.arg.output)
            primary_stem_path, secondary_stem_path = vocal_remove(i_audio, self.arg.output)
            file_index = [os.path.join(self.name_dir, i) for i in os.listdir(self.name_dir) if i.endswith('index')]
            print('load index: ', file_index[0])
            ori_vocals = librosa.load(secondary_stem_path, sr=44100)[0]
            vocals = librosa.resample(ori_vocals, 44100, 16000)
            ori_instrumental = librosa.load(primary_stem_path, sr=44100)[0]
            instrumental = librosa.resample(ori_instrumental, 44100, 48000)

            times = [0, 0, 0]
            vocals = self.vc.pipeline_svc(self.hubert, self.net_g, self.se, self.arg.sid, vocals, secondary_stem_path, times, f0_up_key, self.arg.f0_method,
                                         file_index[0], index_rate, self.if_f0, self.arg.filter_radius, self.tgt_sr,
                                         self.arg.resample, self.arg.rms_mix_rate, self.version, self.arg.protect)
            print(times)
            
            # output audio path
            out_path = os.path.join(self.arg.output, i_audio.split('/')[-1].split('.')[0]+'.wav')
            if self.arg.resample is None:
                self.arg.resample = self.tgt_sr
            wavfile.write(secondary_stem_path, self.arg.resample, vocals)
            wavfile.write(primary_stem_path, self.arg.resample, instrumental)
            # ffmpeg_cmd = "ffmpeg -i %s -i %s -filter_complex amix=inputs=2:duration=first:dropout_transition=3 %s -y" % (primary_stem_path, secondary_stem_path, out_path)
            ffmpeg_cmd = "ffmpeg -i %s -i %s -filter_complex '[0:a]volume=1[a];[1:a]volume=2[v];[a][v]amix=inputs=2:duration=longest' %s -y" % (primary_stem_path, secondary_stem_path, out_path)
            subprocess.run(ffmpeg_cmd, shell=True)
            print('vc done!')
            os.remove(primary_stem_path)
            os.remove(secondary_stem_path)


def run_multi():
    args = parse_args()
    t1 = time.time()
    args.checkpoint = "checkpoint_svc_single_1.0"
    for n in [i for i in os.listdir(args.checkpoint) if i != 'mute']:
        sid = 0
        args.input = 'data/svc/test_women'
        args.output = os.path.join("data/svc/outputs/2023_11_27", n)
        args.f0_up_key = -6
        args.sid = sid

        run = VCInfer(args, args.checkpoint, n, args.device)
        t2 = time.time()
        print('model load time:', t2-t1)
        run.load_vc()
        run.vc_single(args.input, args.f0_up_key, args.index_rate)
        t3 = time.time()
        print('infer time:', t3-t2)
    print('finish in ', t3-t1)


def run_single():
    args = parse_args()
    t1 = time.time()
    run = VCInfer(args, args.checkpoint, args.name, args.device)
    t2 = time.time()
    print('model load time:', t2-t1)
    run.load_vc()
    run.vc_single(args.input, args.f0_up_key, args.index_rate)
    t3 = time.time()
    print('infer time:', t3-t2)
    print('finish in:', t3-t1)


if __name__ == "__main__":
    # run_multi()
    run_single()
    # vocal_remove('data/bug/2.wav')
