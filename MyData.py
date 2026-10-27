import time
from CL_Train import train, eval, select_best
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main(model_config=None):
    # 针对您1024x1024x16数据集的配置
    modelConfig = {
        "state": "eval",  # 首次运行请先设为"train"进行训练，训练完成后可设为"select_best"选择最佳模型，最后设为"eval"进行评估
        "epoch": 40,      # 训练轮数，根据数据量较大可考虑适当减少
        "band": 1200,        # 光谱波段数，您的数据为16
        "batch_size": 40,  # 批大小，根据GPU内存可调整
        "seed": 1,         # 随机种子
        "channel": 16,     # 嵌入维度N，论文中设为16
        "lr": 1e-4,        # 学习率
        "multiplier": 2.,  # 学习率预热参数
        "epision": 10,     # AUC值小数位数
        "grad_clip": 1.,   # 梯度裁剪
        "device": "cuda:0",  # 使用GPU
        "training_load_weight": None,  # 训练时加载的权重，None表示从头训练
        "save_dir": "./models",       # 模型保存目录
        "test_load_weight": "ckpt_27_.pt",  # 测试时加载的权重，训练完成后会自动生成
        "patch_size": 3,   # 论文参数分析表明，对于目标较小的场景，较小的patch size更合适
        "m": 5,            # 光谱分组长度，由于您只有16个波段，设置为2以获得合适的序列长度
        "state_size": 16,  # SSM隐藏状态大小
        "layer": 1,        # 网络深度，论文中设为1以保持轻量
        "delta": 0.1,      # 背景抑制参数
        "dataset": "MyData",  # 您的数据集名称
        "path": "datasets/testdata.mat"  # 您的数据路径，请确保.mat文件包含'data'和'map'变量
    }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train(modelConfig)
    elif modelConfig["state"] == "select_best":
        select_best(modelConfig)
    else:
        eval(modelConfig)


if __name__ == '__main__':
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print('Total execution time: %s seconds' % (end - start))