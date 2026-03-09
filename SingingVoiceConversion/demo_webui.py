import os
import subprocess
import torch
import librosa
import numpy as np
import gradio as gr
from scipy.io import wavfile
from script.i18n import I18nAuto
from fairseq import checkpoint_utils
from lib.audio import load_audio
from script.config import Config
from script.vc_infer_pipeline import VC
from lib.infer_pack.models import SynthesizerTrnMs768NSFsid, SynthesizerTrnMs768NSFsid_nono
from lib.vocal_remover.separate import SeperateVR

ckpt_path = "checkpoint_svc_single_1.0"
sr_dict = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}
names = [n for n in os.listdir(ckpt_path) if n != 'mute']
hubert_model = None
device = torch.device("cuda:0")


def load_hubert():
    global hubert_model
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["weights/hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(device)
    hubert_model = hubert_model.half()
    hubert_model.eval()


def get_vc(name):
    global n_spk, tgt_sr, net_g, vc, cpt, vr
    vr = SeperateVR()
    config = Config()
    person = "%s/%s/%s.pth" % (ckpt_path, name, name)
    person_audio = "%s/%s/%s.wav" % (ckpt_path, name, name)
    print("loading %s" % person)

    cpt = torch.load(person, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    if_f0 = cpt.get("f0", 1)
    if if_f0 == 1:
        net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=True)
    else:
        net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))
    net_g.eval().to(device)
    net_g = net_g.half()
    vc = VC(tgt_sr, device, config)
    n_spk = cpt["config"][-3]

    return person_audio



def vc_infer(
    sid,
    input_audio_path,
    f0_up_key,
    f0_method,
    file_index,
    index_rate,
    filter_radius,
    resample_sr,
    rms_mix_rate,
    protect,
):
    global tgt_sr, net_g, vc, hubert_model, cpt
    if input_audio_path is None:
        return "You need to upload an audio", None
    f0_up_key = int(f0_up_key)
    audio = load_audio(input_audio_path, 16000)
    audio_max = np.abs(audio).max() / 0.95
    if audio_max > 1:
        audio /= audio_max
    times = [0, 0, 0]
    if not hubert_model:
        load_hubert()
    if_f0 = cpt.get("f0", 1)
    audio_opt = vc.pipeline(
        hubert_model,
        net_g,
        sid,
        audio,
        input_audio_path,
        times,
        f0_up_key,
        f0_method,
        file_index,
        index_rate,
        if_f0,
        filter_radius,
        tgt_sr,
        resample_sr,
        rms_mix_rate,
        version="v2",
        protect=protect
    )

    return tgt_sr, audio_opt


def svc_infer(
    sid,
    input_audio_path,
    f0_up_key,
    f0_method,
    file_index,
    index_rate,
    filter_radius,
    resample_sr,
    rms_mix_rate,
    protect,
):
    global tgt_sr, net_g, vc, hubert_model, cpt
    if input_audio_path is None:
        return "You need to upload an audio", None
    f0_up_key = int(f0_up_key)

    # vr
    primary_stem_path, secondary_stem_path = vr.seperate(input_audio_path, os.path.dirname(input_audio_path))
    instrumental = load_audio(primary_stem_path, 48000)
    audio = load_audio(secondary_stem_path, 16000)
    audio_max = np.abs(audio).max() / 0.95
    if audio_max > 1:
        audio /= audio_max
    times = [0, 0, 0]
    if not hubert_model:
        load_hubert()
    if_f0 = cpt.get("f0", 1)
    audio_opt = vc.pipeline(
        hubert_model,
        net_g,
        sid,
        audio,
        secondary_stem_path,
        times,
        f0_up_key,
        f0_method,
        file_index,
        index_rate,
        if_f0,
        filter_radius,
        tgt_sr,
        resample_sr,
        rms_mix_rate,
        version="v2",
        protect=protect
    )

    wavfile.write(secondary_stem_path, 48000, audio_opt)
    wavfile.write(primary_stem_path, 48000, instrumental)
    out_path = os.path.join(os.path.dirname(input_audio_path), input_audio_path.split('/')[-1].split('.')[0]+'_res.wav')
    ffmpeg_cmd = "ffmpeg -i %s -i %s -filter_complex amix=inputs=2:duration=first:dropout_transition=3 -y %s" % (primary_stem_path, secondary_stem_path, out_path)
    subprocess.run(ffmpeg_cmd, shell=True)
    print('vc done!')
    os.remove(primary_stem_path)
    os.remove(secondary_stem_path)
    res = librosa.load(out_path, 48000)[0]

    return tgt_sr, res


with gr.Blocks() as app:
    with gr.Tabs():
        with gr.TabItem("AI音乐变声demo(AI孙燕姿)"):
            with gr.Row():
                sid0 = gr.Dropdown(label="变声模板", choices=sorted(names))
                person_audio = gr.Audio(label="模板人物音频")
                refresh_button = gr.Button("加载模型", variant="primary")
                spk_item = gr.Slider(
                    minimum=0,
                    maximum=2333,
                    step=1,
                    label="请选择说话人id",
                    value=0,
                    visible=False,
                    interactive=True,
                )
            with gr.Row():
                # with gr.Column():
                input_audio0 = gr.Audio(label="上传音频", type='filepath')
                with gr.Column():
                    vc_transform0 = gr.Slider(
                        label="变调(男转女推荐+12, 女转男推荐-12,男转男或女转女默认0)",
                        value=0,
                        minimum=-12,
                        maximum=12,
                    )
                    index_rate1 = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="检索特征占比(调低则音频质量高，声音相似度低，调高则声音相似度高，音频质量低)",
                    value=0.7,
                    interactive=True,
                )
                f0method0 = gr.Radio(
                    label=
                        "选择音高提取算法,输入歌声可用pm提速,harvest低音好但巨慢无比,crepe效果好但吃GPU,rmvpe效果最好且微吃GPU"
                    ,
                    choices=["rmvpe"],
                    value="rmvpe",
                    interactive=True,
                    visible=False
                )
                filter_radius0 = gr.Slider(
                    minimum=0,
                    maximum=7,
                    label=">=3则使用对harvest音高识别的结果使用中值滤波，数值为滤波半径，使用可以削弱哑音",
                    value=3,
                    step=1,
                    interactive=True,
                    visible=False
                )
                # with gr.Column():
                file_index1 = gr.Textbox(
                    label="特征检索库文件路径,为空则使用下拉的选择结果",
                    value="",
                    interactive=True,
                    visible=False
                )
                refresh_button.click(
                    fn=get_vc,
                    inputs=[sid0],
                    outputs=[person_audio],
                    api_name="infer_refresh",
                )
                
                # with gr.Column():
                resample_sr0 = gr.Slider(
                    minimum=0,
                    maximum=48000,
                    label="后处理重采样至最终采样率，0为不进行重采样",
                    value=0,
                    step=1,
                    interactive=True,
                    visible=False
                )
                rms_mix_rate0 = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络",
                    value=0.25,
                    interactive=True,
                    visible=False
                )
                protect0 = gr.Slider(
                    minimum=0,
                    maximum=0.5,
                    label=
                        "保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果"
                    ,
                    value=0.3,
                    step=0.01,
                    interactive=True,
                    visible=False
                )
                but0 = gr.Button("启动变声", variant="primary", size='lg')
                vc_output = gr.Audio(label="输出音频")
                but0.click(
                    svc_infer,
                    [
                        spk_item,
                        input_audio0,
                        vc_transform0,
                        f0method0,
                        file_index1,
                        index_rate1,
                        filter_radius0,
                        resample_sr0,
                        rms_mix_rate0,
                        protect0,
                    ],
                    [vc_output],
                    api_name="infer_convert",
                )
    app.queue(concurrency_count=511, max_size=1022).launch(
            server_name="0.0.0.0",
            inbrowser=False,
            server_port=7861,
            quiet=True,
        )
