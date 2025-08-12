import cv2
import os
import time
import torch
from ultralytics import YOLO
from datetime import timedelta
import numpy as np
from tqdm import tqdm
import argparse

# =============================
# 配置与常量
# =============================
OUTPUT_DIR = 'detected_faces'
RECORDS_DET_PATH = 'face_records.txt'              # 纯检测模式记录
RECORDS_TRACK_FIRST_PATH = 'face_records_track_first.txt'  # Track-去重(首帧)记录
RECORDS_TRACK_BEST_PATH = 'face_records_track_best.txt'    # Track-去重(最佳帧)记录
RECORDS_TRACK_BEST_END_PATH = 'face_records_track_best_on_end.txt'  # Track-去重(轨迹结束时最佳帧)记录
PREVIEW_OUT = 'preview.mp4'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================
# 工具函数
# =============================

def hms_str(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    td = timedelta(seconds=int(seconds))
    h, remainder = divmod(td.seconds, 3600)
    m, s = divmod(remainder, 60)
    h += td.days * 24
    return f"{h:02d}:{m:02d}:{s:02d}"


def safe_clip_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def draw_preview(frame, xyxy_list, ids_list, confs_list, saved_flags, time_text):
    """在帧上绘制检测框、id、置信度与是否保存标记。"""
    out = frame.copy()
    H, W = out.shape[:2]
    # 时间戳角标
    cv2.rectangle(out, (10, 10), (200, 45), (0, 0, 0), -1)
    cv2.putText(out, time_text, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    for i, (xyxy, tid, conf) in enumerate(zip(xyxy_list, ids_list, confs_list)):
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"ID:{tid} | {conf:.2f}"
        if saved_flags and saved_flags[i]:
            label += " ★"
        # 文本背景
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        bx1, by1 = x1, max(0, y1 - th - 8)
        bx2, by2 = x1 + tw + 8, y1
        cv2.rectangle(out, (bx1, by1), (bx2, by2), (0, 255, 0), -1)
        cv2.putText(out, label, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return out


def append_records_header_if_needed(path, header_line):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(header_line + "\n")


def append_records(path, face_records):
    if not face_records:
        return
    with open(path, 'a', encoding='utf-8') as f:
        for r in face_records:
            f.write(
                f"{r['face_id']},{r['track_id']},{r['time']},{r['timestamp']:.2f},{r['confidence']:.4f},{r['image_path']},{r['frame_count']}\n"
            )


def save_image(face_img, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    return cv2.imwrite(file_path, face_img)


# =============================
# 模型加载
# =============================

def load_model():
    # 优先 v11，再回退 v8
    weights_try = ['yolov11n-face.pt', 'yolov8n-face.pt']
    last_error = None
    model = None
    for w in weights_try:
        try:
            model = YOLO(w)
            break
        except Exception as e:
            last_error = e
            model = None
    if model is None:
        raise RuntimeError(f"无法加载人脸模型: {last_error}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    model.to(device)
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
    return model, device


# =============================
# 纯检测（保留以兼容原逻辑，可选）
# =============================

def process_batch_detect(frames_data, model, device, confidence_threshold=0.5):
    batch_frames = [d['frame'] for d in frames_data]
    use_half = (device == 'cuda')
    results = model(batch_frames, verbose=False, device=device, half=use_half)
    face_data = []

    for i, result in enumerate(results):
        frame_info = frames_data[i]
        frame = frame_info['frame']
        timestamp = frame_info['timestamp']
        frame_count = frame_info['frame_count']
        H, W = frame.shape[:2]
        time_str = hms_str(timestamp)

        boxes = getattr(result, 'boxes', None)
        if boxes is None or boxes.data is None or len(boxes) == 0:
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy().reshape(-1)
        for j in range(xyxy.shape[0]):
            conf = float(confs[j])
            if conf < confidence_threshold:
                continue
            x1, y1, x2, y2 = xyxy[j]
            clipped = safe_clip_box(x1, y1, x2, y2, W, H)
            if clipped is None:
                continue
            cx1, cy1, cx2, cy2 = clipped
            face_img = frame[cy1:cy2, cx1:cx2]
            if face_img.size == 0 or face_img.shape[0] < 5 or face_img.shape[1] < 5:
                continue
            face_data.append({
                'face_img': face_img,
                'time_str': time_str,
                'timestamp': timestamp,
                'confidence': conf,
                'frame_count': frame_count
            })
    return face_data


def process_video_detect(video_path, model, device,
                         batch_size=8, sample_interval=None,
                         confidence_threshold=0.5, flush_every=200):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not fps or fps <= 1e-6:
        fps = 25.0
        print("警告: 无法读取 FPS，使用回退值 25.0")

    if sample_interval is None:
        sample_interval = max(int(round(fps / 4.0)), 1)

    print(f"视频FPS: {fps}")
    print(f"总帧数: {total_frames}")
    print(f"采样间隔: 每 {sample_interval} 帧取 1 帧")

    # 记录文件
    append_records_header_if_needed(
        RECORDS_DET_PATH,
        "人脸ID,TrackID(无),时间(HH:MM:SS),时间戳(秒),置信度,图片路径,帧号"
    )

    all_face_count = 0
    frame_count = 0
    pbar = tqdm(total=total_frames if total_frames > 0 else None, desc="检测模式: 处理视频")

    try:
        pending_records = []
        while True:
            frames_batch = []
            batch_cnt = 0
            while batch_cnt < batch_size:
                ret, frame = cap.read()
                if not ret:
                    break
                if total_frames > 0:
                    pbar.update(1)
                if frame_count % sample_interval == 0:
                    timestamp = frame_count / fps
                    frames_batch.append({
                        'frame': frame,
                        'timestamp': timestamp,
                        'frame_count': frame_count
                    })
                    batch_cnt += 1
                frame_count += 1
            if not frames_batch:
                break
            face_data = process_batch_detect(
                frames_batch, model, device, confidence_threshold
            )
            if face_data:
                # 保存图像 + 写记录
                for data in face_data:
                    face_img = data['face_img']
                    time_str = data['time_str']
                    timestamp = data['timestamp']
                    confidence = data['confidence']
                    fcnt = data['frame_count']

                    face_filename = os.path.join(
                        OUTPUT_DIR,
                        f"face_{all_face_count:05d}_t_{time_str.replace(':','-')}_f{fcnt}_c{confidence:.2f}.jpg"
                    )
                    if save_image(face_img, face_filename):
                        pending_records.append({
                            'face_id': all_face_count,
                            'track_id': -1,
                            'time': time_str,
                            'timestamp': timestamp,
                            'confidence': confidence,
                            'image_path': face_filename,
                            'frame_count': fcnt
                        })
                        all_face_count += 1
                if len(pending_records) >= flush_every:
                    append_records(RECORDS_DET_PATH, pending_records)
                    pending_records.clear()
        if pending_records:
            append_records(RECORDS_DET_PATH, pending_records)
    finally:
        pbar.close()
        cap.release()

    print(f"检测模式完成! 共保存 {all_face_count} 张人脸")
    return all_face_count


# =============================
# ByteTrack 去重 + 可视化预览
# =============================

def process_video_track(video_path, model, device,
                        confidence_threshold=0.5,
                        sample_interval=1,
                        save_strategy='first',  # 'first' 或 'best'
                        preview_out=PREVIEW_OUT,
                        flush_every=200):
    assert save_strategy in ['first', 'best']

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not fps or fps <= 1e-6:
        fps = 25.0
        print("警告: 无法读取 FPS，使用回退值 25.0")

    # 预览视频写出（按照采样速率调整 FPS，确保时序感不至于过快）
    ret, frame0 = cap.read()
    if not ret:
        print("视频为空或无法读取第一帧")
        cap.release()
        return 0
    H, W = frame0.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 回到开头

    preview_fps = max(fps / max(sample_interval, 1), 1)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(preview_out, fourcc, preview_fps, (W, H))

    # 记录表头
    if save_strategy == 'first':
                    # 第一次出现立刻保存
                    if tid not in best_cache:
                        face_filename = os.path.join(
                            OUTPUT_DIR,
                            f"face_{face_id:05d}_track{tid}_t_{time_str.replace(':','-')}_f{frame_count}_c{conf:.2f}.jpg"
                        )
                        if save_image(face_img, face_filename):
                            saved_flags[j] = True
                            saved_count += 1
                            rec = {
                                'face_id': face_id,
                                'track_id': tid,
                                'time': time_str,
                                'timestamp': timestamp,
                                'confidence': conf,
                                'image_path': face_filename,
                                'frame_count': frame_count
                            }
                            pending_records.append(rec)
                            best_cache[tid] = {'saved': True}

                elif save_strategy == 'best':
                    # 仅保留最佳帧：以固定文件名覆盖写入
                    entry = best_cache.get(tid)
                    if (entry is None) or (conf > entry['confidence']):
                        face_filename = os.path.join(OUTPUT_DIR, f"face_{face_id:05d}_track{tid}_BEST.jpg")
                        if save_image(face_img, face_filename):
                            saved_flags[j] = True
                            best_cache[tid] = {
                                'face_id': face_id,
                                'track_id': tid,
                                'time': time_str,
                                'timestamp': timestamp,
                                'confidence': conf,
                                'image_path': face_filename,
                                'frame_count': frame_count
                            }
                else:  # 'best_end'
                    # 仅更新最佳候选，不立即落盘；在轨迹结束时统一保存
                    entry = best_end_cache.get(tid)
                    if (entry is None) or (conf > entry['confidence']):
                        best_end_cache[tid] = {
                            'face_id': face_id,
                            'track_id': tid,
                            'time': time_str,
                            'timestamp': timestamp,
                            'confidence': conf,
                            'frame_count': frame_count,
                            'face_img': face_img
                        }
                        saved_flags[j] = True

                # 仅保留最佳帧：以固定文件名覆盖写入
                    entry = best_cache.get(tid)
                    if (entry is None) or (conf > entry['confidence']):
                        face_filename = os.path.join(OUTPUT_DIR, f"face_{face_id:05d}_track{tid}_BEST.jpg")
                        if save_image(face_img, face_filename):
                            saved_flags[j] = True  # 标注在预览上
                            best_cache[tid] = {
                                'face_id': face_id,
                                'track_id': tid,
                                'time': time_str,
                                'timestamp': timestamp,
                                'confidence': conf,
                                'image_path': face_filename,
                                'frame_count': frame_count
                            }

            # 处理轨迹结束（与上一帧对比）
            current_active_ids = set(map(int, ids.tolist() if hasattr(ids, 'tolist') else ids))
            if save_strategy == 'best_end':
                ended_ids = previous_active_ids - current_active_ids
                end_commit_records = []
                for tid in list(ended_ids):
                    entry = best_end_cache.get(tid)
                    if entry:
                        face_filename = os.path.join(OUTPUT_DIR, f"face_{entry['face_id']:05d}_track{tid}_BEST_END.jpg")
                        if save_image(entry['face_img'], face_filename):
                            end_commit_records.append({
                                'face_id': entry['face_id'],
                                'track_id': tid,
                                'time': entry['time'],
                                'timestamp': entry['timestamp'],
                                'confidence': entry['confidence'],
                                'image_path': face_filename,
                                'frame_count': entry['frame_count']
                            })
                            saved_count += 1
                        best_end_cache.pop(tid, None)
                if end_commit_records:
                    pending_records_end.extend(end_commit_records)
                    if len(pending_records_end) >= flush_every:
                        append_records(RECORDS_TRACK_BEST_END_PATH, pending_records_end)
                        pending_records_end.clear()
                previous_active_ids = current_active_ids

            # 写入预览帧
            preview = draw_preview(frame, xyxy, ids, confs, saved_flags, time_str)
            vw.write(preview)

            # 定期将 first 策略的记录落盘
            if save_strategy == 'first' and len(pending_records) >= flush_every:
                append_records(RECORDS_TRACK_FIRST_PATH, pending_records)
                pending_records.clear()
            # 定期将 best_end 策略的记录落盘
            if save_strategy == 'best_end' and len(pending_records_end) >= flush_every:
                append_records(RECORDS_TRACK_BEST_END_PATH, pending_records_end)
                pending_records_end.clear()

            frame_count += 1

        # 尾声：写入 remaining 记录
        if save_strategy == 'first' and pending_records:
            append_records(RECORDS_TRACK_FIRST_PATH, pending_records)

        if save_strategy == 'best':
            if best_cache:
                best_records = list(best_cache.values())
                with open(RECORDS_TRACK_BEST_PATH, 'a', encoding='utf-8') as f:
                    for r in best_records:
                        f.write(
                            f"{r['face_id']},{r['track_id']},{r['time']},{r['timestamp']:.2f},{r['confidence']:.4f},{r['image_path']},{r['frame_count']}
"
                        )
                saved_count = len(best_records)

        if save_strategy == 'best_end':
            # 将仍在进行中的轨迹作为“结束”处理一次
            end_commit_records = []
            for tid, entry in list(best_end_cache.items()):
                face_filename = os.path.join(OUTPUT_DIR, f"face_{entry['face_id']:05d}_track{tid}_BEST_END.jpg")
                if save_image(entry['face_img'], face_filename):
                    end_commit_records.append({
                        'face_id': entry['face_id'],
                        'track_id': tid,
                        'time': entry['time'],
                        'timestamp': entry['timestamp'],
                        'confidence': entry['confidence'],
                        'image_path': face_filename,
                        'frame_count': entry['frame_count']
                    })
                    saved_count += 1
                best_end_cache.pop(tid, None)
            if end_commit_records:
                append_records(RECORDS_TRACK_BEST_END_PATH, end_commit_records)

    finally:
        pbar.close()
        cap.release()
        vw.release()

    print(f"Track 模式完成! 策略: {save_strategy}，共保存 {saved_count} 张人脸；预览已导出到 {preview_out}")
    return saved_count


# =============================
# 主函数与命令行
# =============================

def main():
    parser = argparse.ArgumentParser(description='YOLOv11 人脸检测 + ByteTrack 去重与预览')
    parser.add_argument('--video', type=str, default='1.mp4', help='输入视频路径')
    parser.add_argument('--mode', type=str, choices=['detect', 'track'], default='track', help='运行模式: detect 或 track')
    parser.add_argument('--conf', type=float, default=0.5, help='置信度阈值')
    parser.add_argument('--sample-interval', type=int, default=1, help='采样间隔（每隔多少帧处理一次；track 模式建议 1）')
    parser.add_argument('--batch-size', type=int, default=8, help='detect 模式批量大小')
    parser.add_argument('--flush-every', type=int, default=200, help='积攒多少条记录再写入文件（first/检测模式）')
    parser.add_argument('--save-strategy', type=str, choices=['first', 'best', 'best_end'], default='first', help='track 去重保存策略：first=首次出现保存；best=全程覆盖保存最佳；best_end=仅在轨迹结束时保存最佳')
    parser.add_argument('--preview-out', type=str, default=PREVIEW_OUT, help='可视化预览视频输出路径')

    args = parser.parse_args()

    print("开始加载模型...")
    model, device = load_model()
    print("模型加载完成!")

    start = time.time()
    if device == 'cpu':
        torch.set_num_threads(min(4, os.cpu_count() or 4))

    if args.mode == 'detect':
        total_faces = process_video_detect(
            args.video, model, device,
            batch_size=args.batch_size,
            sample_interval=args.sample_interval if args.sample_interval > 0 else 1,
            confidence_threshold=args.conf,
            flush_every=args.flush_every
        )
    else:
        total_faces = process_video_track(
            args.video, model, device,
            confidence_threshold=args.conf,
            sample_interval=args.sample_interval if args.sample_interval > 0 else 1,
            save_strategy=args.save_strategy,
            preview_out=args.preview_out,
            flush_every=args.flush_every
        )

    end = time.time()
    print(f"处理时间: {end - start:.2f} 秒 | 总保存数: {total_faces}")


if __name__ == "__main__":
    main()
