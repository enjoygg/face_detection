# YOLOv11 人脸检测 + ByteTrack 去重 & 预览

一个用于视频里**批量抽帧检测人脸**并**按轨迹去重保存**的单文件脚本，支持三种保存策略，并可**导出可视化预览视频**（叠加 ID/置信度/保存标记）。

> 代码文件：**Yolov11 人脸检测 + Byte Track 去重 & 可视化预览（单文件版）.py**（见右侧）

---

## ✨ 功能一览

- **两种运行模式**
  - `detect`：纯检测，按采样间隔批量推理并保存所有人脸裁剪。
  - `track`：基于 **ByteTrack** 跟踪去重，三种保存策略：
    - `first`：每个轨迹 **首帧** 保存一次。
    - `best`：运行过程中 **持续覆盖** 为最佳帧（置信度最高）。
    - `best_end`：**仅在轨迹结束时** 保存该轨迹的最佳帧（最省 I/O）。
- **可视化预览**：输出 `preview.mp4`，在原视频上绘制人脸框、Track ID、置信度；当本帧发生保存/最佳更新时以 **★** 标注。
- **稳健性**：
  - 自动回退 `fps=25`（当容器读不到 FPS）。
  - 防越界裁剪，过滤空/极小图。
  - GPU 自动使用半精度推理（`half=True`），并启用 cuDNN benchmark。
  - 记录文件分批落盘（防止异常丢数据）。
- **权重自适应**：优先 `yolov11n-face.pt`，若不可用则回退 `yolov8n-face.pt`。

---

## 🧰 环境要求

- Python **3.8+**（建议 3.9/3.10）
- 依赖：
  - `ultralytics`（YOLOv8/v11 框架，内置 ByteTrack/OC-SORT）
  - `torch`（建议 GPU+Cuda，支持 CPU）
  - `opencv-python`, `tqdm`, `numpy`

### 安装示例

```bash
# 建议在虚拟环境中
pip install ultralytics opencv-python tqdm numpy
# 安装 PyTorch（根据你的 CUDA 版本选择）
# 参考 https://pytorch.org/get-started/locally/
```

> 首次运行会自动从网络下载模型权重；无网络时将尝试回退到本地已存在的 `yolov8n-face.pt`。

---

## 🚀 快速开始

把你的输入视频（例如 `1.mp4`）放在脚本同目录，然后执行：

### 1）去重保存：**首帧** + 预览

```bash
python your_script.py --video 1.mp4 --mode track --save-strategy first --conf 0.5 --sample-interval 1 --preview-out preview.mp4
```

### 2）去重保存：**最佳帧（持续覆盖）** + 预览

```bash
python your_script.py --video 1.mp4 --mode track --save-strategy best --conf 0.5 --sample-interval 1
```

### 3）去重保存：**轨迹结束时的最佳帧** + 预览（推荐大批量处理）

```bash
python your_script.py --video 1.mp4 --mode track --save-strategy best_end --conf 0.5 --sample-interval 1
```

### 4）纯检测（不做跟踪去重）

```bash
python your_script.py --video 1.mp4 --mode detect --conf 0.5 --sample-interval 6 --batch-size 8
```

---

## ⚙️ 命令行参数说明

| 参数                  | 说明                                           | 默认            |
| ------------------- | -------------------------------------------- | ------------- |
| `--video`           | 输入视频路径                                       | `1.mp4`       |
| `--mode`            | 运行模式：`detect` / `track`                      | `track`       |
| `--conf`            | 置信度阈值                                        | `0.5`         |
| `--sample-interval` | 采样间隔（每隔多少帧处理一次）。`track` 模式建议 `1`。            | `1`           |
| `--batch-size`      | `detect` 模式下的批量大小                            | `8`           |
| `--flush-every`     | 累积多少条记录再落盘到 CSV                              | `200`         |
| `--save-strategy`   | `track` 去重保存策略：`first` / `best` / `best_end` | `first`       |
| `--preview-out`     | 预览视频输出路径                                     | `preview.mp4` |

---

## 📦 输出产物

- 目录 ``：裁剪后的人脸图像
  - `face_00001_track12_t_00-00-05_f123_c0.91.jpg`（首帧策略）
  - `face_00003_track7_BEST.jpg`（最佳帧持续覆盖策略）
  - `face_00005_track4_BEST_END.jpg`（轨迹结束时最佳帧策略）
- 记录文件（CSV）：
  - `face_records.txt`（检测模式）
  - `face_records_track_first.txt`（track-首帧）
  - `face_records_track_best.txt`（track-最佳覆盖）
  - `face_records_track_best_on_end.txt`（track-轨迹结束时最佳）
- 预览视频：`preview.mp4`

> 记录文件字段：`face_id, track_id, time(HH:MM:SS), timestamp(sec), confidence, image_path, frame_count`

---

## 🧠 工作原理（简述）

- **Detect 模式**：按 `sample-interval` 从视频抽帧，批量送入 YOLO 推理；对每个框裁剪人脸并直接保存。
- **Track 模式（ByteTrack）**：
  - 每帧调用 `model.track(..., tracker='bytetrack.yaml', persist=True)` 获取带 `id` 的追踪框。
  - **去重策略**：
    - `first`：首次出现时保存一次，随后不再保存。
    - `best`：全程比较置信度，始终覆盖同名文件为“当前最佳”。
    - `best_end`：仅维护“当前最佳”的内存缓存；当 **该 Track ID 从当前帧消失** 或在 **视频结束** 时，才落盘一次。
  - 预览视频在每帧绘制框、Track ID、置信度；当本帧发生保存/最佳更新时，用 **★** 标注该框。

---

## ⚡ 性能与精度建议

- **GPU** 优先，已自动开启半精度推理（`half=True`）。
- **固定分辨率视频** 建议保留默认的 cuDNN benchmark（已自动开启）。
- `track` 模式请尽量设置 `--sample-interval 1`，利于稳定关联；若性能不足再逐步增大。
- 纯检测批处理靠 `--batch-size` 与 `--sample-interval` 控制吞吐；I/O 压力大时把 `detected_faces` 放到 **SSD**。
- 过多小脸/模糊场景可适当 **调低 **``，反之调高；必要时对输入视频先做去噪/放大。

---

## 🧩 常见问题（FAQ）

1. **打不开视频 / FPS 为 0？**\
   请检查路径、编码与权限。脚本已在读不到 FPS 时回退为 `25` 并给出提示。
2. **GPU 显存不足（CUDA OOM）？**\
   先将输入视频分辨率降采样，或提高 `--sample-interval`，或在纯检测模式下减小 `--batch-size`。
3. **没有 **``**？**\
   首次会自动下载；若网络不可用，将尝试回退到本地 `yolov8n-face.pt`。
4. **Track ID 不出现/一直为 None？**\
   请确认 Ultralytics 版本较新且 `tracker='bytetrack.yaml'` 可用；必要时更新 `ultralytics`。
5. **预览太快/太慢？**\
   预览帧率按 `fps / sample-interval` 自动设置；需要更慢的预览可把 `--sample-interval` 调大或用视频播放器控制播放速率。

---

## 🔧 开发者提示

- 记录落盘采用“积攒再写”（`--flush-every`），防止频繁 I/O；异常中断时数据丢失最小化。
- 文件名包含 `trackId / 时间 / 帧号 / 置信度`，便于回溯与对账。
- 如需导出 **框坐标 / 原图可视化帧** 等扩展字段，可在 `append_records` 与 `draw_preview` 处加列/加样式。

---

## 📜 免责声明与合规

请确保在 **合法合规** 的前提下采集与处理视频数据，并遵守相关隐私政策与场地/平台规则。

---

## 🗺️ 路线图（可选）

- 轨迹尾迹（轨迹线）可视化。
- 按人/轨迹导出拼图 report。
- 质量过滤（最小人脸尺寸 / 人脸质量评分 / 模糊度阈值）。
- 多进程 I/O 与离线批整合。

---

