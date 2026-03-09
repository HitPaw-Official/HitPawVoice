# AI Voice Conversion code


启动文件：demo_svc.py

主要参数：
- sid：说话人id，int [0-18]
- f0_up_key：升调降调，int [-12,12].男声变男声或女声变女声0，男声变女声+12，女声变男声-12.
- index_rate:声音相似度，默认0.7，float [0, 1]

环境包详情见requirements.txt