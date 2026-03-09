import os, sys, json, time
import torch
import faiss
import argparse
import numpy as np
from random import shuffle
from subprocess import Popen
from sklearn.cluster import MiniBatchKMeans
from lib.train.utils import HParams

now_dir = os.getcwd()
sys.path.append(os.path.join(now_dir))
sys.path.append(os.path.join('script'))

sr_dict = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}

def get_hps(args):
    with open(args.config, 'r') as f:
        data = f.read()
    configs = json.loads(data)
    hparams = HParams(**configs)

    hparams.model_dir = hparams.experiment_dir = os.path.join('checkpoint', args.experiment_dir)
    hparams.save_every_epoch = args.save_every_epoch
    hparams.name = args.experiment_dir
    hparams.total_epoch = args.total_epoch
    hparams.pretrainG = args.pretrainG
    hparams.pretrainD = args.pretrainD
    hparams.version = args.version
    hparams.gpus = args.gpus
    hparams.train.batch_size = args.batch_size
    hparams.sample_rate = args.sample_rate
    hparams.if_f0 = args.if_f0
    hparams.if_latest = True
    hparams.save_every_weights = "10"
    hparams.if_cache_data_in_gpu = True
    hparams.data.training_files = "%s/filelist.txt" % os.path.join('checkpoint', args.experiment_dir)

    return hparams


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--save_every_epoch", type=int, default=10, help="checkpoint save frequency (epoch)")
    parser.add_argument("--total_epoch", type=int, default=40, help="total_epoch")
    parser.add_argument("--pretrainG", type=str, default="pretrained/f0G48k.pth", help="Pretrained Discriminator path")
    parser.add_argument("--pretrainD", type=str, default="pretrained/f0D48k.pth", help="Pretrained Generator path")
    parser.add_argument("--gpus", type=str, default="0-1", help="split by -")
    parser.add_argument("--num_cpu", type=int, default=8, help="")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--experiment_dir", type=str, default='KanyeWest', help="experiment dir")
    parser.add_argument("--train_dir", type=str, default='data/svc/train', help="train data dir")
    parser.add_argument("--version", type=str, default='v2', help="version, ['v1', 'v2']")
    parser.add_argument("--speaker_id", type=int, default=0, help="speaker id")
    parser.add_argument("--sample_rate", type=str, default='48k', help="sample rate, 32k/40k/48k")
    parser.add_argument("--config", type=str, default='configs/48k_v2.json', help="sample rate, 32k/40k/48k")
    parser.add_argument("--f0_method", type=str, default='rmvpe', help="f0 method, pm/harvest/dio/rmvpe")
    parser.add_argument("--if_f0", type=bool, default=True, help="use f0 as one of the inputs of the model, 1 or 0")

    return parser.parse_args()


def train1key(
    hps,
    exp_dir1,
    sr2,
    if_f0_3,
    trainset_dir,
    spk_id5,
    np7,
    f0method8,
    gpus16,
    version19,
):
    # step1:处理数据
    base_dir = 'checkpoint'
    exp_dir = os.path.join(base_dir, exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    from script.trainset_preprocess_pipeline_print import preprocess_trainset
    trainset_dir4 = os.path.join(trainset_dir, exp_dir1)
    preprocess_trainset(spk_id5, trainset_dir4, sr_dict[sr2], np7, exp_dir)
    print("step1:数据处理完成")

    gt_wavs_dir = "%s/0_gt_wavs" % exp_dir
    feature_dir = ("%s/3_feature256" % exp_dir if version19 == "v1" else "%s/3_feature768" % exp_dir)

    #step2a:提取音高
    if if_f0_3:
        from script.extract_f0_print import FeatureInput
        from multiprocessing import Process
        extractor_f0 = FeatureInput()
        print("step2a:正在提取音高")
        if f0method8 == "rmvpe":
            paths = []
            inp_root = "%s/1_16k_wavs" % exp_dir
            opt_root1 = "%s/2a_f0" % exp_dir
            opt_root2 = "%s/2b-f0nsf" % exp_dir

            os.makedirs(opt_root1, exist_ok=True)
            os.makedirs(opt_root2, exist_ok=True)
            for name in sorted(list(os.listdir(inp_root))):
                inp_path = "%s/%s" % (inp_root, name)
                if "spec" in inp_path:
                    continue
                opt_path1 = "%s/%s" % (opt_root1, name)
                opt_path2 = "%s/%s" % (opt_root2, name)
                paths.append([inp_path, opt_path1, opt_path2])

            ps = []
            for i in range(np7):
                p = Process(
                    target=extractor_f0.go,
                    args=(
                        paths[i::np7],
                        f0method8,
                    ),
                )
                p.start()
                ps.append(p)
            for p in ps:
                p.join()

    #step2b:提取特征
    print("step2b:正在提取特征")
    from script.extract_feature_print import extract_feature
    gpus = gpus16.split("-")
    leng = len(gpus)
    for idx, n_g in enumerate(gpus):
        extract_feature(leng, idx, exp_dir, version19, 'cuda:' + gpus[0])
    
    #step3a:训练模型
    print("step3a:正在训练模型")
    # 生成filelist
    if if_f0_3:
        f0_dir = "%s/2a_f0" % exp_dir
        f0nsf_dir = "%s/2b-f0nsf" % exp_dir
        names = (
            set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
            & set([name.split(".")[0] for name in os.listdir(feature_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
        )
    else:
        names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
            [name.split(".")[0] for name in os.listdir(feature_dir)]
        )
    opt = []
    for name in names:
        if if_f0_3:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    f0_dir.replace("\\", "\\\\"),
                    name,
                    f0nsf_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
        else:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
    fea_dim = 256 if version19 == "v1" else 768
    if if_f0_3:
        for _ in range(2):
            opt.append(
                "%s/mute/0_gt_wavs/mute%s.wav|%s/mute/3_feature%s/mute.npy|%s/mute/2a_f0/mute.wav.npy|%s/mute/2b-f0nsf/mute.wav.npy|%s"
                % (base_dir, sr2, base_dir, fea_dim, base_dir, base_dir, spk_id5)
            )
    else:
        for _ in range(2):
            opt.append(
                "%s/mute/0_gt_wavs/mute%s.wav|%s/mute/3_feature%s/mute.npy|%s"
                % (base_dir, sr2, base_dir, fea_dim, spk_id5)
            )
    shuffle(opt)
    with open("%s/filelist.txt" % exp_dir, "a+") as f:
        f.write("\n".join(opt))
    
    from script.train_nsf_sim_cache_sid_load_pretrain_old import main
    main(hps)
    print("训练结束, 您可查看控制台训练日志或实验文件夹下的train.log")

    #step3b:训练索引
    npys = []
    listdir_res = list(os.listdir(feature_dir))
    for name in sorted(listdir_res):
        phone = np.load("%s/%s" % (feature_dir, name))
        npys.append(phone)
    big_npy = np.concatenate(npys, 0)

    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]

    if big_npy.shape[0] > 2e5:
        info = "Trying doing kmeans %s shape to 10k centers." % big_npy.shape[0]
        print(info)

        try:
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    verbose=True,
                    batch_size=256 * 4,
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )
        except:
            print(info)


    np.save("%s/total_fea.npy" % exp_dir, big_npy)

    # n_ivf =  big_npy.shape[0] // 39
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    print("%s,%s" % (big_npy.shape, n_ivf))
    index = faiss.index_factory(256 if version19 == "v1" else 768, "IVF%s,Flat" % n_ivf)
    print("training index")
    index_ivf = faiss.extract_index_ivf(index)  #
    index_ivf.nprobe = 1
    index.train(big_npy)
    faiss.write_index(
        index,
        "%s/trained_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )
    print("adding index")
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])
    faiss.write_index(
        index,
        "%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )
    print(
        "成功构建索引, added_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (n_ivf, index_ivf.nprobe, exp_dir1, version19)
    )
    print("全流程结束！")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    t1 = time.time()
    args = parse_args()
    hps = get_hps(args)
    train1key(
        hps,
        args.experiment_dir,
        args.sample_rate,
        args.if_f0,
        args.train_dir,
        args.speaker_id,
        args.num_cpu,
        args.f0_method,
        args.gpus,
        args.version,
    )
    t2 = time.time()
    print('Training time: %s' % ((t2-t1) / 60))
