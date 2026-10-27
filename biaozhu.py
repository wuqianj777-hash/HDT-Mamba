import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import os
import sys

# =========================
# 1. 完整文件路径配置
# =========================
FILE_PATH = r"D:\first-floor\2.mat"
OUTPUT_DIR = r"D:\first-floor\2raw\crops"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# 2. 基础配置（避免乱码）
# =========================
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# =========================
# 3. 加载数据
# =========================
print(f"Loading: {os.path.basename(FILE_PATH)}")
try:
    f = h5py.File(FILE_PATH, 'r')
    dset = f['data']
    print(f"Data shape: {dset.shape}")
    B, H, W = dset.shape
except Exception as e:
    print(f"Failed to load data: {e}")
    sys.exit(1)

# =========================
# 4. 加载300波段并优化显示
# =========================
preview_band = 300
print(f"Loading band {preview_band} for preview...")
band = dset[preview_band, :, :][:].astype(np.float32)

vmin = np.percentile(band, 2)
vmax = np.percentile(band, 98)
if vmax <= vmin:
    vmax = vmin + 1

band_norm = np.clip((band - vmin) / (vmax - vmin + 1e-8), 0, 1)
preview_img = (band_norm * 255).astype(np.uint8)
print(f"Preview range: [{preview_img.min()}, {preview_img.max()}]")

# =========================
# 5. 像素标注工具（修复十字显示问题）
# =========================
class PixelAnnotator:
    def __init__(self, img, dset, output_dir, band_idx):
        self.img = img
        self.dset = dset
        self.output_dir = output_dir
        self.band_idx = band_idx
        self.B, self.H, self.W = dset.shape

        # 标注状态
        self.annotation_mask = np.zeros((self.H, self.W), dtype=np.uint8)
        self.selected_points = []  # 存储全局坐标 (y, x)
        self.stage = 1  # 1: 全图选区域, 2: 放大后标注

        # 选区坐标（全局）
        self.x_start = 0
        self.y_start = 0
        self.x_end = self.W - 1
        self.y_end = self.H - 1

        # ---------- 画布 ----------
        self.fig, (self.ax_main, self.ax_mask) = plt.subplots(1, 2, figsize=(13, 7))
        self.fig.canvas.manager.set_window_title(f'Pixel Annotator - Band {band_idx}')

        # 主图：全图预览
        self.ax_main.imshow(
            self.img, cmap='gray', vmin=0, vmax=255,
            interpolation='nearest', aspect='equal'
        )
        self.ax_main.set_title('Stage1: Drag a box → SPACE zoom in | ESC quit')
        self.ax_main.axis('off')

        # 掩码图
        self.ax_mask.imshow(
            self.annotation_mask, cmap='gray', vmin=0, vmax=1,
            interpolation='nearest', aspect='equal'
        )
        self.ax_mask.set_title('Annotation Mask (0 pixels)')
        self.ax_mask.axis('off')

        # ---------- 十字标记（红色十字，无连线）----------
        # 使用plot的空数据集初始化，后续动态更新
        self.marker_pts, = self.ax_main.plot(
            [], [], 'r+',
            markersize=8,
            markeredgewidth=2,
            linestyle='none',
            visible=False
        )

        # ---------- 矩形选择器（无填充，细线）----------
        self.rect_selector = RectangleSelector(
            self.ax_main, self.on_rect_select,
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords='pixels',
            interactive=True,
            props={
                'edgecolor': '#FF3333',
                'linewidth': 1.5,
                'linestyle': '--',
                'fill': False   # ✅ 不要红色填充
            }
        )

        # 事件绑定
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        plt.tight_layout()
        plt.show()

    # ----------------------------------------------------
    # 选区回调
    # ----------------------------------------------------
    def on_rect_select(self, eclick, erelease):
        x1, y1 = map(lambda v: int(round(v)), (eclick.xdata, eclick.ydata))
        x2, y2 = map(lambda v: int(round(v)), (erelease.xdata, erelease.ydata))

        self.x_start = max(0, min(self.W - 1, min(x1, x2)))
        self.y_start = max(0, min(self.H - 1, min(y1, y2)))
        self.x_end   = max(0, min(self.W - 1, max(x1, x2)))
        self.y_end   = max(0, min(self.H - 1, max(y1, y2)))

        print(f'Selected: ({self.x_start},{self.y_start}) → ({self.x_end},{self.y_end})')

    # ----------------------------------------------------
    # 键盘事件
    # ----------------------------------------------------
    def on_key(self, event):
        # ===== SPACE =====
        if event.key == ' ':
            # ---- Stage1 → Stage2：放大选区 ----
            if self.stage == 1:
                self.stage = 2
                self.rect_selector.set_active(False)

                # 裁剪放大区域
                patch = self.img[
                    self.y_start:self.y_end + 1,
                    self.x_start:self.x_end + 1
                ]

                # ✅ 清除旧图，重画放大区域
                self.ax_main.cla()
                self.ax_main.imshow(
                    patch, cmap='gray', vmin=0, vmax=255,
                    interpolation='nearest', aspect='equal'
                )
                self.ax_main.set_title(
                    f'Stage2: Click to annotate | SPACE save | ESC back'
                )
                self.ax_main.axis('off')

                # ✅ 重新创建十字标记（关键修复）
                self.marker_pts, = self.ax_main.plot(
                    [], [], 'r+',
                    markersize=8,
                    markeredgewidth=2,
                    linestyle='none',
                    visible=False
                )

                # 更新掩码图显示（只显示选区部分）
                self.update_mask_display()
                
                print(f'\nZoomed into region '
                      f'({self.x_start},{self.y_start}) → ({self.x_end},{self.y_end})')
                self.fig.canvas.draw_idle()

            # ---- Stage2 → 保存 ----
            elif self.stage == 2:
                self.save_annotations()
                plt.close(self.fig)

        # ===== ESC =====
        elif event.key == 'escape':
            # Stage2 → Stage1：回到全图
            if self.stage == 2:
                self.stage = 1
                self.rect_selector.set_active(True)

                self.ax_main.cla()
                self.ax_main.imshow(
                    self.img, cmap='gray', vmin=0, vmax=255,
                    interpolation='nearest', aspect='equal'
                )
                self.ax_main.set_title('Stage1: Drag a box → SPACE zoom in | ESC quit')
                self.ax_main.axis('off')

                # ✅ 重新创建十字标记（关键修复）
                self.marker_pts, = self.ax_main.plot(
                    [], [], 'r+',
                    markersize=8,
                    markeredgewidth=2,
                    linestyle='none',
                    visible=False
                )

                # 更新掩码图显示（显示完整掩码）
                self.ax_mask.imshow(
                    self.annotation_mask, cmap='gray', vmin=0, vmax=1,
                    interpolation='nearest', aspect='equal'
                )
                self.ax_mask.set_title(f'Annotation Mask ({len(self.selected_points)})')
                self.ax_mask.axis('off')

                print('Back to Stage1 (full image)')
                self.fig.canvas.draw_idle()

            # Stage1 → 退出
            else:
                plt.close(self.fig)

    # ----------------------------------------------------
    # 鼠标点击（标注）- 修复十字显示
    # ----------------------------------------------------
    def on_click(self, event):
        if self.stage != 2 or event.inaxes != self.ax_main or event.button != 1:
            return

        # 局部坐标（放大图上的坐标）
        lx = int(round(event.xdata))
        ly = int(round(event.ydata))

        # 转回全局坐标
        gx = lx + self.x_start
        gy = ly + self.y_start

        # 安全校验
        if not (self.x_start <= gx <= self.x_end and
                self.y_start <= gy <= self.y_end):
            return

        pt = (gy, gx)

        # 切换标注状态
        if pt in self.selected_points:
            self.selected_points.remove(pt)
            self.annotation_mask[gy, gx] = 0
            print(f'Removed: ({gx},{gy})')
        else:
            self.selected_points.append(pt)
            self.annotation_mask[gy, gx] = 1
            print(f'Added: ({gx},{gy})')

        # ✅ 关键修复：更新十字标记显示
        if self.selected_points:
            # 转换为局部坐标用于显示
            local_points = [(x - self.x_start, y - self.y_start) 
                          for y, x in self.selected_points]
            xs, ys = zip(*local_points)
            self.marker_pts.set_data(xs, ys)
            self.marker_pts.set_visible(True)
        else:
            self.marker_pts.set_visible(False)

        # 更新掩码图
        self.update_mask_display()
        self.fig.canvas.draw_idle()

    # ----------------------------------------------------
    # 更新掩码显示
    # ----------------------------------------------------
    def update_mask_display(self):
        """更新掩码图的显示"""
        if self.stage == 1:
            # 显示完整掩码
            self.ax_mask.imshow(
                self.annotation_mask, cmap='gray', vmin=0, vmax=1,
                interpolation='nearest', aspect='equal'
            )
        else:
            # 只显示选区部分的掩码
            mask_patch = self.annotation_mask[
                self.y_start:self.y_end + 1,
                self.x_start:self.x_end + 1
            ]
            self.ax_mask.imshow(
                mask_patch, cmap='gray', vmin=0, vmax=1,
                interpolation='nearest', aspect='equal'
            )
        
        self.ax_mask.set_title(f'Annotation Mask ({len(self.selected_points)})')
        self.ax_mask.axis('off')

    # ----------------------------------------------------
    # 保存标注（h5py，兼容 MATLAB v7.3）
    # ----------------------------------------------------
    def save_annotations(self):
        if len(self.selected_points) == 0:
            print('\nNo annotations to save!')
            return

        print(f'\nSaving {len(self.selected_points)} annotations...')

        base_name = os.path.splitext(os.path.basename(FILE_PATH))[0]
        ts = np.datetime64('now').astype(str).replace(':', '-').replace('.', '-')
        out_path = os.path.join(
            self.output_dir,
            f'{base_name}_band{self.band_idx}_{ts}.mat'
        )

        with h5py.File(out_path, 'w') as hf:
            hf.create_dataset(
                'annotationMask',
                data=self.annotation_mask,
                dtype=np.uint8
            )
            hf.create_dataset(
                'selectedPoints',
                data=np.array(self.selected_points, dtype=np.int32)
            )
            hf.attrs['numAnnotations'] = len(self.selected_points)
            hf.attrs['imageShape'] = np.array([self.H, self.W], dtype=np.int32)
            hf.attrs['bandIndex'] = self.band_idx
            hf.attrs['selectedRegion'] = np.array(
                [self.x_start, self.y_start, self.x_end, self.y_end],
                dtype=np.int32
            )
            hf.attrs['creationTime'] = ts

        print(f'Saved: {out_path}')
        print(f'Annotations: {len(self.selected_points)}')
        print(f'Region: ({self.x_start},{self.y_start}) → ({self.x_end},{self.y_end})')

# =========================
# 6. 启动
# =========================
if __name__ == '__main__':
    try:
        PixelAnnotator(preview_img, dset, OUTPUT_DIR, preview_band)
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
    finally:
        if 'f' in locals():
            f.close()
            print('Data file closed.')