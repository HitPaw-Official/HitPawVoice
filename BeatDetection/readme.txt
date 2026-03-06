1、安装环境
pip install -r requirements.txt
cd madmom
python setup.py develop

2、测试脚本
python demo.py --input data\test\1_3beats.mp3 --model models/model_1_weights.onnx

3、测试文件在data/test，基准结果在data/outputs