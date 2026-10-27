import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import scipy.io

def merge_hyperspectral_data():
    """
    合并高光谱数据和标注掩码，输出为MATLAB v5格式的.mat文件
    特别显示第300通道的数据
    """
    
    # =========================
    # 1. 直接设置输入输出路径
    # =========================
    data_source_file = r"D:\first-floor\crops\crop_256x256_21.mat"
    map_source_file = r"D:\first-floor\2raw\crops\crop_256x256_21_band300_2026-07-21T06-32-02.mat"
    output_file = r"D:\first-floor\reconstruction\data.mat"
    
    print('='*60)
    print('高光谱数据合并工具 (Python -> MATLAB .mat格式)')
    print('='*60)
    print(f'处理开始...')
    print(f'data文件: {data_source_file}')
    print(f'map文件: {map_source_file}')
    print(f'输出文件: {output_file}')
    print('-'*60)
    
    # =========================
    # 2. 加载data数据
    # =========================
    print(f'\n正在加载data数据...')
    try:
        with h5py.File(data_source_file, 'r') as f:
            data_vars = list(f.keys())
            print(f'  文件中的变量: {data_vars}')
            
            if len(data_vars) == 0:
                raise ValueError('data文件中没有找到变量')
            
            data_var = data_vars[0]
            data = f[data_var][:]
            print(f'  使用变量: {data_var}, 维度: {data.shape}')
            
    except Exception as e:
        print(f'  加载data失败: {e}')
        return
    
    # =========================
    # 3. 加载map数据
    # =========================
    print(f'\n正在加载map数据...')
    try:
        with h5py.File(map_source_file, 'r') as f:
            map_vars = list(f.keys())
            print(f'  文件中的变量: {map_vars}')
            
            if len(map_vars) == 0:
                raise ValueError('map文件中没有找到变量')
            
            # 优先选择annotationMask变量
            if 'annotationMask' in map_vars:
                map_var = 'annotationMask'
            else:
                map_var = map_vars[0]
            
            map_data = f[map_var][:]
            print(f'  使用变量: {map_var}, 维度: {map_data.shape}')
            
    except Exception as e:
        print(f'  加载map失败: {e}')
        return
    
    # =========================
    # 4. 检查并修正维度
    # =========================
    print(f'\n检查数据维度...')
    
    # 检查data维度 (1200, 256, 256)
    data_shape = data.shape
    print(f'  data原始维度: {data_shape}')
    
    # 检查map维度 (256, 256)
    map_shape = map_data.shape
    print(f'  map原始维度: {map_shape}')
    
    # 确保data是(H, W, B)格式，即(256, 256, 1200)
    if len(data_shape) == 3:
        if data_shape[0] == 1200 and data_shape[1] == 256 and data_shape[2] == 256:
            print(f'  转置data维度: {data_shape} -> (256, 256, 1200)')
            data = np.transpose(data, (1, 2, 0))  # (1200, 256, 256) -> (256, 256, 1200)
        elif data_shape == (256, 256, 1200):
            print(f'  data维度正确: {data_shape}')
        else:
            print(f'  data维度异常: {data_shape}，尝试继续处理...')
    
    # 确保map是二维的(256, 256)
    if len(map_shape) == 2:
        if map_shape == (256, 256):
            print(f'  map维度正确: {map_shape}')
        else:
            print(f'  map维度异常: {map_shape}，尝试继续处理...')
    elif len(map_shape) == 3:
        # 如果是三维的，取第一个通道
        print(f'  map是三维的，取第一个通道')
        map_data = map_data[:, :, 0]
        print(f'  新map维度: {map_data.shape}')
    
    print(f'  处理后data维度: {data.shape}')
    print(f'  处理后map维度: {map_data.shape}')
    
    # =========================
    # 5. 合并并保存为MATLAB v5格式的.mat文件
    # =========================
    print(f'\n合并并保存为MATLAB v5格式文件...')
    
    # 创建输出目录
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f'  创建输出目录: {output_dir}')
    
    try:
        # 准备保存的数据字典
        save_dict = {
            'data': data,      # 高光谱数据 (256, 256, 1200)
            'map': map_data    # 标注掩码 (256, 256)
        }
        
        # 保存为MATLAB v5格式（最兼容的格式）
        scipy.io.savemat(output_file, save_dict, format='5')
        
        print(f'  已保存到: {output_file}')
        print(f'  文件格式: MATLAB v5 (.mat)')
        print(f'  包含变量: data (维度:{data.shape}), map (维度:{map_data.shape})')
        
    except Exception as e:
        print(f'  保存失败: {e}')
        return
    
    # =========================
    # 6. 显示第300通道的数据（重点修改部分）
    # =========================
    print(f'\n显示第300通道的数据...')
    
    # 检查是否有第300通道（索引为299，因为Python从0开始计数）
    if data.shape[2] >= 300:
        channel_idx = 299  # 第300通道的索引
        
        # 提取第300通道
        channel_300 = data[:, :, channel_idx]
        
        print(f'  第300通道维度: {channel_300.shape}')
        print(f'  第300通道数据范围: [{channel_300.min():.4f}, {channel_300.max():.4f}]')
        print(f'  第300通道均值: {channel_300.mean():.4f}')
        print(f'  第300通道标准差: {channel_300.std():.4f}')
        
        # 归一化显示
        channel_norm = (channel_300 - channel_300.min()) / \
                      (channel_300.max() - channel_300.min() + 1e-8)
        
        # 创建多子图显示
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 第300通道灰度图像
        axes[0, 0].imshow(channel_norm, cmap='gray')
        axes[0, 0].set_title(f'第300通道灰度图像')
        axes[0, 0].axis('off')
        
        # 2. 第300通道直方图
        axes[0, 1].hist(channel_300.flatten(), bins=100, edgecolor='black')
        axes[0, 1].set_title(f'第300通道直方图')
        axes[0, 1].set_xlabel('像素值')
        axes[0, 1].set_ylabel('频率')
        
        # 3. 第300通道与map的叠加显示
        overlay = np.ma.masked_where(map_data == 0, map_data)
        axes[1, 0].imshow(channel_norm, cmap='gray')
        axes[1, 0].imshow(overlay, cmap='hot', alpha=0.5)
        axes[1, 0].set_title(f'第300通道与标注掩码叠加')
        axes[1, 0].axis('off')
        
        # 4. 第300通道的伪彩色显示
        axes[1, 1].imshow(channel_norm, cmap='jet')
        axes[1, 1].set_title(f'第300通道伪彩色显示')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # 保存第300通道的图像
        save_channel_path = os.path.join(os.path.dirname(output_file), 'channel_300.png')
        plt.savefig(save_channel_path, dpi=150, bbox_inches='tight')
        print(f'  第300通道图像已保存: {save_channel_path}')
        
    else:
        print(f'  数据只有{data.shape[2]}个通道，无法显示第300通道')
        # 显示最接近的第299通道（如果存在）
        if data.shape[2] >= 299:
            channel_idx = 298  # 第299通道的索引
            channel_299 = data[:, :, channel_idx]
            channel_norm = (channel_299 - channel_299.min()) / \
                          (channel_299.max() - channel_299.min() + 1e-8)
            
            plt.figure(figsize=(8, 6))
            plt.imshow(channel_norm, cmap='gray')
            plt.title(f'第299通道灰度图像（第300通道不存在）')
            plt.colorbar()
            plt.axis('off')
            plt.tight_layout()
            plt.show()
    
    # =========================
    # 7. 显示map黑白图像
    # =========================
    print(f'显示map黑白图像...')
    
    plt.figure(figsize=(8, 6))
    plt.imshow(map_data, cmap='gray', vmin=0, vmax=1)
    plt.title('map黑白图像')
    plt.colorbar()
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # =========================
    # 8. 信息统计
    # =========================
    print('\n' + '='*60)
    print('信息统计')
    print('='*60)
    print(f'data维度: {data.shape}')
    print(f'map维度: {map_data.shape}')
    
    # 计算map中0和1的数量
    map_binary = map_data > 0
    num_ones = np.sum(map_binary)
    num_zeros = map_binary.size - num_ones
    percentage = 100 * num_ones / map_binary.size
    
    print(f'map中1的数量: {num_ones}')
    print(f'map中0的数量: {num_zeros}')
    print(f'1的百分比: {percentage:.2f}%')
    
    # 第300通道统计信息
    if data.shape[2] >= 300:
        channel_300 = data[:, :, 299]
        print(f'\n第300通道统计:')
        print(f'  最小值: {channel_300.min():.4f}')
        print(f'  最大值: {channel_300.max():.4f}')
        print(f'  均值: {channel_300.mean():.4f}')
        print(f'  标准差: {channel_300.std():.4f}')
    
    print('='*60)
    print('\n处理完成！')
    
    # =========================
    # 9. 验证输出文件
    # =========================
    print('\n验证输出文件...')
    try:
        mat_data = scipy.io.loadmat(output_file)
        print(f'  输出文件中的变量: {list(mat_data.keys())}')
        
        # 检查必要的变量
        required_vars = ['data', 'map']
        for var in required_vars:
            if var in mat_data:
                var_data = mat_data[var]
                print(f'  {var}变量维度: {var_data.shape}, 数据类型: {var_data.dtype}')
            else:
                print(f'  警告: {var}变量不存在!')
                
    except Exception as e:
        print(f'  验证失败: {e}')
    
    return output_file

def visualize_comparison(data, map_data, channel_idx=299):
    """
    可视化对比data的指定通道和map
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 显示data指定通道
    channel = data[:, :, channel_idx]
    channel_norm = (channel - channel.min()) / \
                   (channel.max() - channel.min() + 1e-8)
    
    axes[0].imshow(channel_norm, cmap='gray')
    axes[0].set_title(f'data第{channel_idx+1}通道')
    axes[0].axis('off')
    
    # 显示map
    axes[1].imshow(map_data, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('map标注掩码')
    axes[1].axis('off')
    
    # 显示叠加效果
    overlay = np.ma.masked_where(map_data == 0, map_data)
    axes[2].imshow(channel_norm, cmap='gray')
    axes[2].imshow(overlay, cmap='hot', alpha=0.5)
    axes[2].set_title('叠加显示')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # 运行主函数
    output_path = merge_hyperspectral_data()
    
    # 可选：可视化对比
    if output_path and os.path.exists(output_path):
        print('\n是否显示详细对比？(y/n)')
        choice = input().lower()
        if choice == 'y':
            try:
                mat_data = scipy.io.loadmat(output_path)
                data = mat_data['data']
                map_data = mat_data['map']
                # 显示第300通道的对比（索引为299）
                visualize_comparison(data, map_data, channel_idx=299)
            except Exception as e:
                print(f'可视化失败: {e}')