import sys
import os
# 强制将当前目录加入路径，防止找不到 src
sys.path.append(os.getcwd())

import torch
from torchviz import make_dot
from src.model import EfficiencyPredictor

def main():
    # 1. 初始化模型
    # 确保这里的参数与您训练时一致
    INPUT_DIM = 19
    model = EfficiencyPredictor(input_dim=INPUT_DIM)
    
    # 2. 创建一个虚拟输入数据
    # 假设一个轨迹有 100 帧 (Batch Size = 1)
    dummy_input = torch.randn(1, 100, INPUT_DIM)
    
    # 3. 前向传播 (Forward Pass)
    # 模型返回: (macro_pred, scores, attn_weights)
    # 我们主要想看 macro_pred (最终预测值) 是怎么算出来的，因为通过它能回溯整个网络
    output = model(dummy_input)
    macro_pred = output[0]
    
    # 4. 生成计算图
    # params=dict(model.named_parameters()): 显示参数名称
    # show_attrs=True: 显示层的属性 (如 input/output shape)
    # show_saved=True: 显示反向传播需要的中间张量
    graph = make_dot(macro_pred, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
    
    # 5. 保存文件
    # filename='model_viz' 会生成 model_viz.png (如果 format='png')
    # view=False 确保不尝试打开窗口
    output_filename = "model_architecture"
    graph.format = "png"
    graph.render(filename=output_filename, view=False)
    
    print(f"成功！架构图已保存为: {output_filename}.png")
    print(f"同时生成了源文件: {output_filename}")

if __name__ == "__main__":
    main()