import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

# ================= 配置 =================
FILE_PATH = r"D:\first-floor\2.mat"
CROP_W, CROP_H = 256, 256
PREVIEW_BAND = 300
OUTPUT_DIR = r"D:\first-floor\crops"
# =======================================

# ====== 全局状态（关键修复）======
x = y = drag_offset = None
dset = preview = None
B = H = W = 0
# ================================

def on_press(e):
    global drag_offset
    if e.inaxes != ax or not rect.contains(e)[0]:
        return
    drag_offset = (x - e.xdata, y - e.ydata)

def on_motion(e):
    global x, y, drag_offset
    if drag_offset is None or e.inaxes != ax:
        return
    dx, dy = drag_offset
    x = max(0, min(W - CROP_W, dx + e.xdata))
    y = max(0, min(H - CROP_H, dy + e.ydata))
    rect.set_xy((x, y))
    fig.canvas.draw_idle()

def on_release(_):
    global drag_offset
    drag_offset = None

def on_key(e):
    global x, y
    if e.key == 'escape':
        plt.close(fig)
        return

    if e.key == ' ':
        crop = dset[:, int(y):int(y+CROP_H), int(x):int(x+CROP_W)]
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out = f"{OUTPUT_DIR}/crop_{CROP_W}x{CROP_H}_{np.random.randint(1000)}.mat"
        with h5py.File(out, 'w') as hf:
            hf.create_dataset('data', data=crop)
        print(f"Saved: {out}")
        print(f"Shape: {crop.shape}")
        plt.close(fig)
        return

    step = 10
    if e.key == 'up':    y = max(0, y - step)
    elif e.key == 'down': y = min(H - CROP_H, y + step)
    elif e.key == 'left': x = max(0, x - step)
    elif e.key == 'right': x = min(W - CROP_W, x + step)

    rect.set_xy((x, y))
    fig.canvas.draw_idle()

# ================= 主程序 =================
with h5py.File(FILE_PATH, 'r') as f:
    dset = f['data']
    B, H, W = dset.shape

    band = dset[PREVIEW_BAND, :, :][:].astype(np.float32)
    vmin, vmax = np.percentile(band, [2, 98])
    preview = (np.clip((band - vmin) / (vmax - vmin + 1e-8), 0, 1) * 255).astype(np.uint8)

    x = (W - CROP_W) // 2
    y = (H - CROP_H) // 2
    drag_offset = None

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(preview, cmap='gray', vmin=0, vmax=255)
    ax.set_title("Band 300 | SPACE:Save | ESC:Quit | Arrows:Move")
    ax.axis('off')

    rect = plt.Rectangle((x, y), CROP_W, CROP_H, edgecolor='red', facecolor='none', lw=2)
    ax.add_patch(rect)

    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.tight_layout()
    plt.show()