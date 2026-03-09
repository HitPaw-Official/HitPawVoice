import os, sys, traceback

import torch
import torch.nn.functional as F
import soundfile as sf
import numpy as np
from fairseq import checkpoint_utils


# wave must be 16k, hop_size=320
def readwave(wav_path, normalize=False):
    wav, sr = sf.read(wav_path)
    assert sr == 16000
    feats = torch.from_numpy(wav).float()
    if feats.dim() == 2:  # double channels
        feats = feats.mean(-1)
    assert feats.dim() == 1, feats.dim()
    if normalize:
        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
    feats = feats.view(1, -1)
    return feats


def extract_feature(n_part, i_part, exp_dir, version, device):
    model_path = "weights/hubert_base.pt"
    wavPath = "%s/1_16k_wavs" % exp_dir
    outPath = ("%s/3_feature256" % exp_dir if version == "v1" else "%s/3_feature768" % exp_dir)
    os.makedirs(outPath, exist_ok=True)

    # HuBERT model
    print("load model(s) from {}".format(model_path))
    # if hubert model is exist
    if os.access(model_path, os.F_OK) == False:
        print(
            "Error: Extracting is shut down because %s does not exist, you may download it from https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main"
            % model_path
        )
        exit(0)
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [model_path],
        suffix="",
    )
    model = models[0]
    model = model.to(device)
    model = model.half()
    model.eval()

    todo = sorted(list(os.listdir(wavPath)))[i_part::n_part]
    n = max(1, len(todo) // 10)  # 最多打印十条
    if len(todo) == 0:
        print("no-feature-todo")
    else:
        print("all-feature-%s" % len(todo))
        for idx, file in enumerate(todo):
            try:
                if file.endswith(".wav"):
                    wav_path = "%s/%s" % (wavPath, file)
                    out_path = "%s/%s" % (outPath, file.replace("wav", "npy"))

                    if os.path.exists(out_path):
                        continue

                    feats = readwave(wav_path, normalize=saved_cfg.task.normalize)
                    padding_mask = torch.BoolTensor(feats.shape).fill_(False)
                    inputs = {
                        "source": feats.half().to(device)
                        if device not in ["mps", "cpu"]
                        else feats.to(device),
                        "padding_mask": padding_mask.to(device),
                        "output_layer": 9 if version == "v1" else 12,  # layer 9
                    }
                    with torch.no_grad():
                        logits = model.extract_features(**inputs)
                        feats = (
                            model.final_proj(logits[0]) if version == "v1" else logits[0]
                        )

                    feats = feats.squeeze(0).float().cpu().numpy()
                    if np.isnan(feats).sum() == 0:
                        np.save(out_path, feats, allow_pickle=False)
                    else:
                        print("%s-contains nan" % file)
                    if idx % n == 0:
                        print("now-%s,all-%s,%s,%s" % (len(todo), idx, file, feats.shape))
            except:
                print(traceback.format_exc())
        print("all-feature-done")
