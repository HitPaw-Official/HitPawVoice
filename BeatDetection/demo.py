import librosa, os, argparse, time
from BeatNet import BeatNet


def beat_detect(audio_path, model_path):
    ori_wav, ori_sr = librosa.load(audio_path)
    new_wav = librosa.resample(y=ori_wav, orig_sr=ori_sr, target_sr=22050)

    estimator = BeatNet(model_path)
    beats = estimator.process(new_wav)
    file = beats.tolist()

    return file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/test/1_3beats.mp3", help="input dir or audio path")
    parser.add_argument("--model", type=str, default="models/model_1_weights.onnx", help="model path")


    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    t1 = time.time()
    if os.path.isdir(args.input):
        for audio in os.listdir(args.input):
            audio_in = os.path.join(args.input, audio)
            file = beat_detect(args.input, args.model)
            print(file)
    else:
        if os.path.exists(args.input):
            file = beat_detect(args.input, args.model)
            print(file)
    t2 = time.time()
    print('\n%s Done in %s s' %(args.input, t2-t1))
