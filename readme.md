## USAGE:
* model.py定义了RRAM的PGM模型，train.py读入数据集进行训练， test.py读入训练好的参数输出模型推断的A和V。
* python train.py [-o](overwrite) [-r](result path)
* python test.py [-m](model path) [-r](result path)

## TODO:
* 整理输出结果（ELBO, A, V）
* ELBO训练后期有时会有小正数，原因不明
* 训练不稳定（不收敛？），调超参