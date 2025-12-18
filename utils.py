import cv2
import os
import re
from paddleocr import PaddleOCR
from difflib import SequenceMatcher


##每秒抽一次帧
def extract_frames(video_path, output_dir, interval_sec):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_interval = int(fps * interval_sec)
    frame_id = 0
    results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if current_frame_index % frame_interval == 0:
            timestamp = current_frame_index / fps
            frame_path = os.path.join(output_dir, f"frame_{frame_id}.jpg")
            cv2.imwrite(frame_path, frame)

            results.append({
                "frame_id": frame_id,
                "timestamp": round(timestamp, 2),
                "image_path": frame_path
            })
            frame_id += 1

    cap.release()
    return results

def resize_frame(frame, max_width=800):
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(frame, (new_w, new_h))
    return frame

def run_ocr(frames):
    ocr_results = []
    # ocr = TextRecognition()

    ocr = PaddleOCR(
    use_doc_orientation_classify=False, 
    use_doc_unwarping=False, 
    use_textline_orientation=False) # 文本检测+文本识别

    # for frame in frames:
    #     resized_frame = resize_frame(frame, max_width=720)
    #     result = ocr.predict(frame["image_path"])
    # 在 run_ocr 中：
    for frame_info in frames:
        image = cv2.imread(frame_info["image_path"])
        if image is None:
            continue
        resized = resize_frame(image, max_width=540)
        result = ocr.predict(resized)  # ← 传图像

        blocks = []

        for line in result:
            texts = line['rec_texts']          # list of str
            scores = line['rec_scores']        # list of float
            bboxes = line['rec_polys']         # list of arrays (each is Nx2)

            # 确保三者长度一致
            assert len(texts) == len(scores) == len(bboxes), "Mismatch in OCR result lengths"

            for i in range(len(texts)):
                text = texts[i]
                confidence = scores[i]
                bbox = bboxes[i]  # shape: (N, 2), e.g., [[x1,y1], [x2,y2], ...]

                # 提取 x, y 坐标
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]

                blocks.append({
                    "text": text.strip(),
                    "confidence": float(confidence),
                    "bbox": [
                        int(min(x_coords)),
                        int(min(y_coords)),
                        int(max(x_coords)),
                        int(max(y_coords))
                    ]
                })

                ocr_results.append({
                    "frame_id": frame_info["frame_id"],
                    "timestamp": frame_info["timestamp"],
                    "ocr_blocks": blocks
                })

    return ocr_results

#OCR 去噪
def is_valid_text(text):
    # 规则 1：长度 < 2
    if len(text) < 2:
        return False

    # 规则 2：非中英文字符占比 > 50%
    non_zh_en = re.findall(r"[^\u4e00-\u9fa5A-Za-z]", text)
    if len(non_zh_en) / len(text) > 0.5:
        return False

    # 规则 3：全是标点或数字
    if re.fullmatch(r"[\d\W]+", text):
        return False

    return True

def denoise_ocr(ocr_results, conf_threshold=0.75):
    cleaned = []

    for frame in ocr_results:
        valid_blocks = []

        for block in frame["ocr_blocks"]:
            if block["confidence"] < conf_threshold:
                continue

            if not is_valid_text(block["text"]):
                continue

            valid_blocks.append(block)

        if valid_blocks:
            cleaned.append({
                "frame_id": frame["frame_id"],
                "timestamp": frame["timestamp"],
                "ocr_blocks": valid_blocks
            })

    return cleaned

#去重复
def text_similarity(a, b):
    return SequenceMatcher(None, a.strip(), b.strip()).ratio()

def merge_text_across_frames_for_understanding(
    cleaned_ocr,
    sim_threshold,      # 相似度阈值（可调）
    time_gap_merge       # 时间间隔 ≤2秒视为连续
):
    """
    为视频理解优化的OCR文本合并:
    - 保留所有文本（包括水印）
    - 将相似文本聚类
    - 合并其出现的时间段（支持非连续出现）
    """
    clusters = {}  # key: representative_text, value: list of timestamps

    # Step 1: 聚类所有文本块（按相似度）
    for frame in cleaned_ocr:
        t = frame["timestamp"]
        for block in frame["ocr_blocks"]:
            text = block["text"]
            if not text.strip():
                continue

            # 查找是否与已有聚类相似
            matched_key = None
            for key in clusters:
                if text_similarity(text, key) >= sim_threshold:
                    matched_key = key
                    break

            if matched_key is not None:
                clusters[matched_key].append(t)
            else:
                clusters[text] = [t]  # 以首次出现的文本为key

    # Step 2: 对每个聚类，合并时间段
    result = []
    for text, times in clusters.items():
        unique_times = sorted(set(times))  # 去重 + 排序
        if not unique_times:
            continue

        segments = []
        start = end = unique_times[0]
        for t in unique_times[1:]:
            if t - end <= time_gap_merge:
                end = t
            else:
                segments.append((start, end))
                start = end = t
        segments.append((start, end))

        # 每个连续段作为一条记录
        for s, e in segments:
            result.append({
                "text": text,
                "start_time": round(s, 2),
                "end_time": round(e, 2)
            })

    # 按开始时间排序，便于阅读
    result.sort(key=lambda x: (x["start_time"], -len(x["text"])))  # 长文本优先
    return result

def build_timeline(segments):
    timeline_lines = []
    for seg in segments:
        line = f"[{seg['start_time']}s - {seg['end_time']}s] {seg['text']}"
        timeline_lines.append(line)
    
    timeline_text = "\n".join(timeline_lines)
    return timeline_text


def build_prompt(default_prompt: str, **kwargs) -> str:

    if not isinstance(default_prompt, str):
        raise TypeError("模板必须是字符串")
    
    try:
        # 先对所有传入的值做基本清理（可选）
        cleaned_kwargs = {
            k: (str(v).strip() if v is not None else "")
            for k, v in kwargs.items()
        }
        rendered = default_prompt.format(**cleaned_kwargs)
        return rendered.strip()
    
    except KeyError as e:
        raise ValueError(f"模板中包含未提供的变量占位符: {{{e.args[0]}}}")
    
    except ValueError as e:
        # 通常是 timeline_text 中包含未转义的 '{' 或 '}' 导致
        raise ValueError(
            f"变量内容包含非法格式字符（如未配对的 '{{' 或 '}}'），请检查输入内容。错误详情: {e}"
        )
    
    except Exception as e:
        raise ValueError(f"渲染 prompt 模板时发生未知错误: {e}")

