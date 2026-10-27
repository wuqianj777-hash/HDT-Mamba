import numpy as np
import h5py
from spectral.io import envi
import os
import gc

# =========================
# 1. 路径配置
# =========================
hdr_path = r"D:\first-floor\2_FSHT25C009.hdr"
spe_path = r"D:\first-floor\2_FSHT25C009.spe"
out_mat_path = r"D:\first-floor\2.mat"

# 文件存在性检查
for path, name in [(hdr_path, "HDR文件"), (spe_path, "SPE文件")]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到{name}：{path}")
print("✅ 所有输入文件都存在")

# =========================
# 2. 读取头文件
# =========================
print("\nOpening HDR file...")
img = envi.open(hdr_path, spe_path)
H, W, B = img.shape
print(f"数据维度: 高度={H}, 宽度={W}, 波段数={B}")
print(f"数据类型: {img.dtype}")

# =========================
# 3. 读取原始数据范围检查
# =========================
print("\n检查原始数据范围...")
for b in [0, 100, 300, 600, 900, 1199]:
    if b < B:
        band = img.read_band(b)
        print(f"  波段 {b}: 最小值={band.min()}, 最大值={band.max()}, 均值={band.mean():.2f}, 类型={band.dtype}")

# =========================
# 4. 计算全局最大值
# =========================
print("\n计算全局最大值（用于保持波段间相对关系）...")
sample_bands = list(range(0, B, max(1, B//50)))
print(f"  采样 {len(sample_bands)} 个波段进行计算...")

band_maxes = []
for i, b in enumerate(sample_bands):
    band = img.read_band(b)
    band_max = band.max()
    band_maxes.append(band_max)
    if (i + 1) % 10 == 0:
        print(f"    已采样 {i+1}/{len(sample_bands)} 个波段，当前最大值: {band_max}")

global_max = np.percentile(band_maxes, 99.9)
print(f"✅ 99.9%分位数全局最大值: {global_max}")
print(f"✅ 采样波段最大值列表: 最小={min(band_maxes)}, 最大={max(band_maxes)}")

if global_max == 0:
    raise ValueError("所有波段的最大值为0！请检查相机采集")

# =========================
# 5. 创建8位内存映射
# =========================
temp_file = r"D:\first-floor\hypercube_temp.dat"
print(f"\n创建临时内存映射文件: {temp_file}")
hypercube_disk = np.memmap(
    temp_file,
    dtype=np.uint8,
    mode='w+',
    shape=(H, W, B)
)

# =========================
# 6. 逐波段写入
# =========================
print("\n开始逐波段处理...")
print("  处理进度: ", end="", flush=True)

for b in range(B):
    band = img.read_band(b)
    band_f = band.astype(np.float32)
    band_f = np.clip(band_f, 0, None)
    
    if global_max > 0:
        band_normalized = band_f / global_max
        if b % 200 == 0:
            norm_min = band_normalized.min()
            norm_max = band_normalized.max()
            print(f"\n    波段 {b}: 归一化后范围 [{norm_min:.6f}, {norm_max:.6f}]")
    else:
        band_normalized = band_f
    
    band_scaled = band_normalized * 255.0
    band_scaled = np.clip(band_scaled, 0, 255)
    hypercube_disk[:, :, b] = band_scaled.astype(np.uint8)
    
    del band, band_f, band_normalized, band_scaled
    
    if (b + 1) % 50 == 0:
        print(f"■", end="", flush=True)
        if (b + 1) % 500 == 0:
            print(f" {b+1}/{B}", end="", flush=True)
    
    if (b + 1) % 100 == 0:
        gc.collect()

print(f"\n✅ 所有波段处理完成")

print("\n验证内存映射文件中的数据...")
test_band = hypercube_disk[:, :, 300]
print(f"  测试波段300: 最小值={test_band.min()}, 最大值={test_band.max()}, 均值={test_band.mean():.2f}")

hypercube_disk.flush()

# =========================
# 7. 读取波长信息
# =========================
wavelengths = img.metadata.get("wavelength")
if wavelengths is not None:
    wavelengths = np.array(wavelengths, dtype=np.float32)
    print(f"\n✅ 波长信息读取成功，共{B}个波段")
    print(f"  波长范围: {wavelengths[0]:.2f} - {wavelengths[-1]:.2f} nm")
else:
    wavelengths = np.arange(B, dtype=np.float32)
    print(f"\n⚠️ 未找到波长信息，使用波段索引代替")

# =========================
# 8. 保存为MATLAB v7.3格式
# =========================
print(f"\n保存为MATLAB v7.3格式: {out_mat_path}")
with h5py.File(out_mat_path, 'w') as f:
    ds = f.create_dataset(
        "data",
        shape=(B, H, W),
        dtype=np.uint8,
        chunks=(1, 256, 256),
        compression="gzip",
        compression_opts=4
    )
    
    print("  写入数据到HDF5文件...")
    for b in range(B):
        ds[b, :, :] = hypercube_disk[:, :, b]
        if (b + 1) % 200 == 0:
            print(f"    已写入 {b+1}/{B} 个波段")
    
    f.create_dataset("wavelength", data=wavelengths)
    f.attrs["original_shape"] = (H, W, B)
    f.attrs["global_max"] = float(global_max)
    f.attrs["data_type"] = "uint8"
    f.attrs["normalization"] = f"global_max_normalized_to_0-255"

print("✅ HDF5文件保存完成")

# =========================
# 9. 验证保存的文件
# =========================
print("\n验证保存的文件...")
with h5py.File(out_mat_path, 'r') as f:
    saved_data = f["data"]
    print(f"  保存的数据维度: {saved_data.shape}")
    print(f"  保存的数据类型: {saved_data.dtype}")
    for b in [0, 300, 600, 900, 1199]:
        if b < saved_data.shape[0]:
            band_data = saved_data[b, :, :]
            print(f"  波段 {b}: 最小值={band_data.min()}, 最大值={band_data.max()}, 均值={band_data.mean():.2f}")

# =========================
# 10. 清理临时文件（修复Windows文件锁问题）
# =========================
print("\n清理临时文件...")
try:
    # 显式删除memmap对象，强制释放文件句柄
    del hypercube_disk
    # 强制垃圾回收，确保Windows释放文件锁
    gc.collect()
    
    if os.path.exists(temp_file):
        os.remove(temp_file)
        print(f"✅ 临时文件已删除: {temp_file}")
except PermissionError:
    print(f"⚠️ 临时文件被系统占用，无法自动删除，可手动清理：{temp_file}")
    print("  提示：下次运行程序时会自动覆盖该文件，不影响使用")
except Exception as e:
    print(f"⚠️ 删除临时文件时出现错误: {e}")

print("\n🎉 数据生成完成！")
print(f"输出文件路径: {out_mat_path}")
print(f"数据范围: 原始0~{int(global_max)} → 归一化后0~255 (uint8)")
print("\n📌 后续使用建议:")
print("  1. 用裁剪工具打开 data.mat，按B键切换到波段300")
print("  2. 放大预览图，能看到清晰的亮斑/物体轮廓")
print("  3. 按空格保存裁剪区域，直接用于HTD-Mamba训练")