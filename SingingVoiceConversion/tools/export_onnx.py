import sys, os
now_dir = os.path.curdir
sys.path.append(now_dir)
# from lib.infer_pack.models_onnx import SynthesizerTrnMsNSFsidM
from lib.infer_pack.models_dml import SynthesizerTrnMs768NSFsid
import torch
import onnx
from onnxsim import simplify


def export(model_path):
    MoeVS = True  # 模型是否为MoeVoiceStudio（原MoeSS）使用
    output_Path = model_path.replace('pth', 'onnx')  # 输出路径
    hidden_channels = 768  # hidden_channels，为768Vec做准备
    cpt = torch.load(model_path, map_location="cpu")
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    print(*cpt["config"])

    test_phone = torch.rand(1, 200, hidden_channels).half()  # hidden unit
    test_phone_lengths = torch.tensor([200]).long()  # hidden unit 长度（貌似没啥用）
    test_pitch = torch.randint(size=(1, 200), low=5, high=255)  # 基频（单位赫兹）
    test_pitchf = torch.rand(1, 200).half()  # nsf基频
    # test_ds = torch.LongTensor([0])  # 说话人ID
    test_ds = torch.LongTensor([0])#.unsqueeze(0)  # 说话人ID
    test_rnd = torch.rand(1, 192, 200).half()  # 噪声（加入随机因子）

    device = "cuda:0"  # 导出时设备（不影响使用模型）

    net_g = SynthesizerTrnMs768NSFsid(
        *cpt["config"], version='v2', is_half=True
    )  # fp32导出（C++要支持fp16必须手动将内存重新排列所以暂时不用fp16
    net_g.load_state_dict(cpt["weight"], strict=False)
    input_names = ["phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd"]
    output_names = [
        "audio",
    ]
    net_g.half()
    net_g.to(device)
    # net_g.construct_spkmixmap(19) # 多角色混合轨道导出
    torch.onnx.export(
        net_g,
        (
            test_phone.to(device),
            test_phone_lengths.to(device),
            test_pitch.to(device),
            test_pitchf.to(device),
            test_ds.to(device),
            test_rnd.to(device),
        ),
        output_Path,
        dynamic_axes={
            "phone": [1],
            "pitch": [1],
            "pitchf": [1],
            "rnd": [2],
        },
        do_constant_folding=False,
        opset_version=15,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
    )

    onnx_model = onnx.load(output_Path)  # load onnx model
    model_simp, check = simplify(onnx_model)
    onnx.save(model_simp, output_Path)
    print('finished exporting onnx')


if __name__ == "__main__":
    # models = [m for m in os.listdir("checkpoint") if m != 'mute']
    # for index, m in enumerate(models):
    #     export('checkpoint/%s/%s.pth'% (m, m))
    #     print('model %s export success! count %s' % (m, index+1))

    export("checkpoint/One/One.pth")
    print('model export success!')
