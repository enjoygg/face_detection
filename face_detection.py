import cv2
import os
import time
import torch
from ultralytics import YOLO
from datetime import datetime
import numpy as np
import concurrent.futures
from tqdm import tqdm

# 创建保存人脸图片的目录
OUTPUT_DIR = 'detected_faces'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载YOLOv11模型
def load_model():
    # 使用预训练的YOLOv11模型，专注于人脸检测
    model = YOLO('yolov11n-face.pt')
    # 确保使用GPU（如果可用）
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    model.to(device)
    return model

# 批量处理帧并检测人脸
def process_batch(frames_data, model, confidence_threshold=0.5):
    batch_frames = [data['frame'] for data in frames_data]
    
    # 批量推理
    results = model(batch_frames, verbose=False)
    
    face_data = []
    for i, result in enumerate(results):
        frame_info = frames_data[i]
        frame = frame_info['frame']
        timestamp = frame_info['timestamp']
        frame_count = frame_info['frame_count']
        
        # 计算时间字符串
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)
        time_str = f"{minutes:02d}:{seconds:02d}"
        
        # 处理检测结果
        boxes = result.boxes
        for box in boxes:
            # 获取置信度
            confidence = float(box.conf[0])
            
            # 如果置信度足够高
            if confidence > confidence_threshold:
                # 获取边界框坐标
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # 裁剪人脸区域
                face_img = frame[y1:y2, x1:x2]
                
                face_data.append({
                    'face_img': face_img,
                    'time_str': time_str,
                    'timestamp': timestamp,
                    'confidence': confidence,
                    'frame_count': frame_count
                })
    
    return face_data

# 保存检测到的人脸图片
def save_face_images(face_data, start_face_count=0):
    face_records = []
    face_count = start_face_count
    
    for data in face_data:
        face_img = data['face_img']
        time_str = data['time_str']
        timestamp = data['timestamp']
        confidence = data['confidence']
        
        # 保存人脸图片
        face_filename = f"{OUTPUT_DIR}/face_{face_count:04d}_time_{time_str.replace(':', '_')}.jpg"
        cv2.imwrite(face_filename, face_img)
        
        # 记录人脸出现的时间
        face_records.append({
            'face_id': face_count,
            'time': time_str,
            'timestamp': timestamp,
            'confidence': confidence,
            'image_path': face_filename
        })
        
        face_count += 1
    
    return face_records, face_count

# 处理视频并检测人脸
def process_video(video_path, model):
    # 打开视频文件
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"无法打开视频: {video_path}")
    except Exception as e:
        print(f"错误: {str(e)}")
        return None
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"视频FPS: {fps}")
    print(f"总帧数: {total_frames}")
    
    # 设置批处理大小和采样间隔
    batch_size = 8  # 批处理大小
    sample_interval = 8  # 每隔多少帧采样一次
    
    # 用于记录人脸出现的时间
    all_face_records = []
    face_count = 0
    
    # 创建进度条
    pbar = tqdm(total=total_frames, desc="处理视频")
    
    # 处理视频帧
    frame_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        frames_batch = []
        batch_count = 0
        
        # 收集一批帧
        while batch_count < batch_size and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 更新进度条
            pbar.update(1)
            
            # 每隔几帧处理一次，提高效率
            if frame_count % sample_interval == 0:
                # 计算当前帧的时间戳
                timestamp = frame_count / fps
                
                frames_batch.append({
                    'frame': frame,
                    'timestamp': timestamp,
                    'frame_count': frame_count
                })
                batch_count += 1
            
            frame_count += 1
        
        # 如果没有帧可处理，退出循环
        if not frames_batch:
            break
        
        # 批量处理帧
        face_data = process_batch(frames_batch, model)
        
        # 保存检测到的人脸
        if face_data:
            face_records, face_count = save_face_images(face_data, face_count)
            all_face_records.extend(face_records)
    
    # 关闭进度条
    pbar.close()
    
    # 释放资源
    cap.release()
    
    # 保存人脸记录到文件
    save_records(all_face_records)
    
    print(f"处理完成! 共检测到 {face_count} 个人脸")
    return all_face_records

# 保存人脸记录到文件
def save_records(face_records):
    try:
        with open('face_records.txt', 'w', encoding='utf-8') as f:
            f.write("人脸ID,时间(MM:SS),时间戳(秒),置信度,图片路径\n")
            for record in face_records:
                f.write(f"{record['face_id']},{record['time']},{record['timestamp']:.2f},{record['confidence']:.4f},{record['image_path']}\n")
    except Exception as e:
        print(f"保存记录时出错: {str(e)}")

# 主函数
def main():
    print("开始加载YOLOv11模型...")
    model = load_model()
    print("模型加载完成!")
    
    video_path = '1.mp4'
    print(f"开始处理视频: {video_path}")
    
    # 设置PyTorch以使用更多线程
    if torch.cuda.is_available():
        torch.set_num_threads(4)
    
    start_time = time.time()
    face_records = process_video(video_path, model)
    end_time = time.time()
    
    print(f"处理时间: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    main()