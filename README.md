# dog_breed
https://www.kaggle.com/c/dog-breed-identification
kaggle 120种狗识别比赛， 目前得分0.03665.
使用inception_v3 网络， dataset使用Stanford dogs dataset。

我发现目前gluon不能导入自定义训练模型， 本项目把inception_V3模型从gluon中提取出来。使其可以重复训练。

新增特征提取后代码
得分0.00872
test_features.py 提取特征
train.py 训练
