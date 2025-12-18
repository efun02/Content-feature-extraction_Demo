import json
import os
import time
from utils import (
    extract_frames,
    run_ocr,
    denoise_ocr,
    merge_text_across_frames_for_understanding,
    build_timeline,
    build_prompt
)
from llm_client import LLMClient

# 最终实现
if __name__ == "__main__":
    start = time.perf_counter()
    video_path = "sample_videos/体育新闻热点.mp4"
    frame_dir = "frames"
    #OCR 提取并保存为指定格式

    interval_sec=5#抽帧间隔
    frames = extract_frames(video_path, frame_dir, interval_sec)
    ocr_raw = run_ocr(frames)
    ocr_cleaned = denoise_ocr(ocr_raw, conf_threshold=0.75)
    final_segments = merge_text_across_frames_for_understanding(ocr_cleaned,sim_threshold=0.92,time_gap_merge=interval_sec + 1 )
    timeline_text = build_timeline(final_segments)
        
    default_prompt = """你是一个专业的短视频内容理解与审核模型。
    你将基于视频中通过 OCR 提取的文字内容，对视频进行多维度分析。

    【背景说明】
    - 以下文字按时间顺序提取自视频画面（包括字幕、标题、水印等）。
    - 若某文本在连续时间段重复出现，通常表示其为核心信息或固定标识。
    - 文本可能包含口语化表达、营销话术或不完整句子，请结合整体语境理解。

    【视频文字时间轴】
    {timeline_text}

    【分析任务】
    请严格按以下 8 项完成分析，并以 JSON 格式输出，不要任何额外说明：

    1. summary: 用一句话概括视频主要内容
    2. tags: 给出 3~5 个内容标签（字符串列表）
    3. category: 内容类型（如 新闻、体育、娱乐、广告 等）
    4. genre: 内容体裁（如 赛事报道、人物特写、快讯 等）
    5. tone: 整体调性（如 客观、煽情、幽默、严肃 等）
    6. sentiment: 情感倾向（如 积极、消极、中性）
    7. is_low_quality: 是否为低质内容（是/否）
    8. has_risk: 是否存在潜在违规风险（是/否）

    【输出格式】
    只输出一个合法 JSON 对象，字段名必须为上述英文名。"""

    prompt = build_prompt(default_prompt,timeline_text=timeline_text)

    llm = LLMClient(
        api_key=os.getenv('DASHSCOPE_API_KEY'),
        api_url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        model_name="qwen-plus"  # 示例
    )
    try:
        analysis_result = llm.analyze(prompt)
        print(json.dumps(analysis_result, ensure_ascii=False, indent=2))
        
        
    except Exception as e:
        print("分析失败:", str(e))
    elapsed = time.perf_counter() - start
    print(f"\n🕒 整个流程耗时: {elapsed:.2f} 秒")
