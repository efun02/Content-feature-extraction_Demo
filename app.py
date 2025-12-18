# app.py
import streamlit as st
import os
import time
import tempfile
from pathlib import Path
import logging
import json
import os
from datetime import datetime

# ====== 导入你的真实模块(请根据实际路径调整)======
from utils import (
    extract_frames,
    run_ocr,
    denoise_ocr,
    merge_text_across_frames_for_understanding,
    build_timeline,
    build_prompt
)
from llm_client import LLMClient



default_prompt = """你是一个专业的短视频内容理解与审核模型。
你将基于视频中通过 OCR 提取的文字内容，对视频进行多维度分析。

【背景说明】
- 以下文字按时间顺序提取自视频画面（包括字幕、标题、水印等）。
- 若某文本在连续时间段重复出现，通常表示其为核心信息或固定标识。
- 文本可能包含口语化表达、营销话术或不完整句子，请结合整体语境理解。

【视频文字时间轴】
{timeline_text}

【分析任务】
请严格按以下各项完成分析，并以 JSON 格式输出，不要任何额外说明：

1. summary: 用一句话概括视频主要内容
2. summary_confidence:给出摘要的置信度(0-1)
3. tags: 给出 3~5 个内容标签（字符串列表）
4. category: 内容类型（如 新闻、体育、娱乐、广告 等）
5. genre: 内容体裁（如 赛事报道、人物特写、快讯 等）
6. tone: 整体调性（如 客观、煽情、幽默、严肃 等）
7. sentiment: 情感倾向（如 积极、消极、中性）
8. is_low_quality: 是否为低质内容（是/否）若是请描述原因
9. has_risk: 是否存在潜在违规风险（是/否）若是请描述原因

【输出格式】
只输出一个合法 JSON 对象，字段名必须为上述英文名。"""
# ====== 页面配置 ======
st.set_page_config(
    page_title="🎥 AI 视频理解系统",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# 创建 logs 目录（如果不存在）
os.makedirs("logs", exist_ok=True)

# 配置日志格式和文件
log_filename = f"logs/analysis_{datetime.now().strftime('%Y%m%d')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()  # 同时输出到控制台（可选）
    ]
)
logger = logging.getLogger(__name__)


def get_analysis_mode_config(mode: str):

    MODE_CONFIGS = {
        "快速摘要": {
            "sim_threshold": 0.92,      # 高相似才合并,保留关键信息
            "time_gap_merge": 6,        # 较长间隔,减少片段数量
            "interval_sec": 5,

        },
        "全面分析": {
            "sim_threshold": 0.85,      # 中等相似度,平衡细节与冗余
            "time_gap_merge": 3,        # 适中合并窗口
            "interval_sec": 3,

        },
        "审核模式": {
            "sim_threshold": 0.78,      # 更敏感,保留更多原文细节(防漏检)
            "time_gap_merge": 2,        # 短间隔,避免跨镜头误合
            "interval_sec": 1,
        },
        "自定义": {
            "sim_threshold": 0.90,      # 默认值,实际由前端传参覆盖(此处仅兜底)
            "time_gap_merge": 6,
            "interval_sec": 1,
        }
    }

    if mode not in MODE_CONFIGS:
        raise ValueError(f"不支持的分析模式: {mode}。可选值: {list(MODE_CONFIGS.keys())}")

    return MODE_CONFIGS[mode]
# ====== 主界面 ======
st.title("🎥 AI 视频内容理解系统")
st.caption("支持多模态分析 · 动态 Prompt 配置 · 实时结构化输出")

# ==============================
# 三栏布局
# ==============================
col_left, col_middle, col_right = st.columns([5, 3.2, 4.3], gap="medium")
# --- 左侧:输入控制 ---
with col_left:
    with st.container(border=True, height=800):
        st.subheader("🛠️ 输入与配置")
        
        use_sample = st.checkbox("📱 使用示例视频", value=True)
        uploaded_file = None
        if not use_sample:
            uploaded_file = st.file_uploader("📤 上传 MP4 视频", type=["mp4"])
        
        st.markdown("### 🧠 Prompt ")
        with st.container(border=True, height=400):
            if "current_prompt" not in st.session_state:
                st.session_state.current_prompt = default_prompt

            edited_prompt = st.text_area(
                '',
                value=st.session_state.current_prompt,  # 显示当前已确认的 prompt
                height=300,
                help="必须包含 {timeline_text} 占位符，否则无法使用"
            )
            if st.button("✅ 确认更新 Prompt"):
                if "{timeline_text}" not in edited_prompt:
                    # 校验失败：显示错误提示，不更新
                    st.error("❌ 修改失败：Prompt 中必须包含 `{timeline_text}` 占位符！")
                else:
                    # 校验成功：保存并提示
                    st.session_state.current_prompt = edited_prompt
                    st.toast("✅ Prompt 已成功更新！", icon="🎉")
            
        analysis_mode = st.selectbox(
            "🔍 分析模式",
            options=["快速摘要", "全面分析", "审核模式", "自定义"],
            index=1,
            help="快速摘要:低延迟；全面分析:平衡效果；审核模式:高精度+安全检测"
        )
        # 👇 仅在自定义模式
        if analysis_mode == "自定义":
            st.markdown("⚙️ 自定义参数(仅 UI 展示,实际由后端使用)")
            user_sim_threshold = st.slider("文本相似度阈值", 0.7, 1.0, 0.92, 0.01)    
            user_time_gap_merge = st.slider("合并时间间隔(秒)", 3, 10, 6, 1)
            user_interval_sec = st.slider("抽帧间隔", 1, 10, 5, 1)

        st.markdown("🧠 大模型选择")
        model_options = {
                "Qwen-Turbo（快速/低成本）": "qwen-turbo",
                "Qwen-Plus（均衡）": "qwen-plus",
                "Qwen-Max（最强/高精度）": "qwen-max",
        }
        selected_model_label = st.selectbox(
                "选择推理模型",
                options=list(model_options.keys()),
                index=1,
                help="自定义模式下可自由选择底层大模型"
            )
        selected_model = model_options[selected_model_label]

# --- 中间:视频预览 ---
with col_middle:
    with st.container(border=True, height=800):
        st.subheader("📺 视频预览")
        
        video_source = None
        if use_sample:
            sample_path = "sample_videos/体育新闻热点.mp4"
            if Path(sample_path).exists():
                video_source = sample_path
            else:
                st.warning("❌ 示例视频未找到,请上传或检查 sample_videos/ 目录")
        elif uploaded_file:
            video_source = uploaded_file
        
        if video_source:
            st.video(video_source, format="video/mp4")
        else:
            st.info("请选择视频源")


# --- 右侧:分析结果区(一体化流式展示)---
with col_right:
    result_container = st.container(height=800, border=True)
    
    with result_container:
        st.subheader("📊 分析结果")
        
        if st.button("🚀 开始分析", type="primary", use_container_width=True):
            mode_config = get_analysis_mode_config(analysis_mode)
            if analysis_mode == "自定义":
                actual_sim = user_sim_threshold
                actual_gap = user_time_gap_merge
                interval_sec = user_interval_sec
            else:
                actual_sim = mode_config["sim_threshold"]
                actual_gap = mode_config["time_gap_merge"]
                interval_sec = mode_config["interval_sec"]
            
            status_text = st.empty()
            progress_bar = st.progress(0)
            
            try:
                # === 阶段 1: 准备视频 ===
                status_text.text("⏳ 准备视频...")
                progress_bar.progress(10)
                
                if use_sample:
                    video_path = "sample_videos/体育新闻热点.mp4"
                    if not Path(video_path).exists():
                        raise FileNotFoundError("示例视频不存在,请检查 sample_videos/ 目录")
                else:
                    if not uploaded_file:
                        raise ValueError("请上传视频文件")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                        tmp.write(uploaded_file.read())
                        video_path = tmp.name
                
                # === 阶段 2: 抽帧 & OCR ===
                status_text.text("📸 抽帧中...")
                progress_bar.progress(30)
                frame_dir = tempfile.mkdtemp()
                frames = extract_frames(video_path, frame_dir, interval_sec)
               
                
                status_text.text("🔤 OCR 识别中...")
                progress_bar.progress(50)
                ocr_raw = run_ocr(frames)
                ocr_cleaned = denoise_ocr(ocr_raw, conf_threshold=0.75)
                
                # === 阶段 3: 合并文本 ===
                status_text.text("🧩 合并文本片段...")
                progress_bar.progress(70)

                final_segments = merge_text_across_frames_for_understanding(
                    ocr_cleaned,
                    sim_threshold=actual_sim,
                    time_gap_merge=actual_gap
                )
                # === 构造最终 Prompt(关键:使用用户输入的 prompt)===

                timeline_text = build_timeline(final_segments)

                final_prompt = build_prompt(st.session_state.current_prompt,timeline_text=timeline_text)

                
                # === 阶段 4: 调用 LLM ===
                status_text.text("🧠 调用大模型生成报告...")
                progress_bar.progress(90)
                actual_model = selected_model  # ← 用户选择的模型

                llm = LLMClient(
                    api_key=os.getenv('DASHSCOPE_API_KEY'),
                    api_url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                    model_name=actual_model
                )
                result = llm.analyze(final_prompt)  # ← 真实调用
                # 在你的分析代码中替换日志部分
                try:
                    # 构造纯字符串日志（安全！）
                    log_msg = (
                        f"\n-------------------------------- [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] --------------------------\n"
                        f"----------------Full Prompt:---------------\n{st.session_state.current_prompt}\n"
                        f"----------------TimelineText:-------------\n{timeline_text}\n"
                        f"Model: {actual_model}\n"
           
                        f"Result preview: {str(result)[:500]}\n"
                        f"{'-'*50}\n"
                    )
                    
                    # 直接写入文件（不依赖 logging 模块）
                    with open(f"logs/analysis_{datetime.now().strftime('%Y%m%d')}.log", "a", encoding="utf-8") as f:
                        f.write(log_msg)
                    
                    st.toast("✅ 分析日志已保存！", icon="📝")

                except Exception as e:
                    st.error(f"❌ 日志保存失败: {e}")
                    # 强制记录错误
                    with open("logs/error.log", "a") as f:
                        f.write(f"{datetime.now()}: {e}\n")
                progress_bar.progress(100)#进度条
                time.sleep(0.2)
                status_text.empty()
                progress_bar.empty()
                
                # ==============================
                # 可折叠显示原始 result 内容
                # ==============================
                with st.expander("🔍 查看LLM分析结果"):
                    st.json(result)  # 以格式化 JSON 显示，美观且可读
                    # 或者用 st.write(result) 也可以，但 st.json 更适合字典结构
                # 摘要 + 置信度
                st.markdown("##### 📝 自动摘要")
                st.write(result.get("summary", "未返回摘要"))
                conf = result.get("summary_confidence", 0.85)
                st.markdown(
                    f'<div style="height:6px; background:#e2e8f0; border-radius:3px; margin:8px 0;">'
                    f'<div style="height:100%; width:{conf*100}%; background:#3b82f6; border-radius:3px;"></div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                st.caption(f"置信度:{conf:.0%}")
                st.divider()
                
                # 标签
                tags = result.get("tags", [])
                if tags:
                    tag_badges = "".join([
                        f'<span style="display:inline-block; background:#dbeafe; color:#1d4ed8; '
                        f'padding:4px 12px; border-radius:20px; margin:0 6px 6px 0; font-size:0.85em;">'
                        f'{tag}</span>'
                        for tag in tags
                    ])
                    st.markdown("##### 🏷️ 关键词标签")
                    st.markdown(tag_badges, unsafe_allow_html=True)
                    st.divider()
                
                feature_labels = {
                    "category": "分类",
                    "genre": "体裁",
                    "tone": "调性",
                    "sentiment": "情感倾向",
                    "is_low_quality": "是否为低质内容",
                    "has_risk": "是否潜在违规风险",
                }

                # 用于存储最终要显示的 (标签, 值) 对
                display_items = []

                # 1. 处理已知字段（按 feature_labels 顺序，保证 UI 稳定）
                for key, label in feature_labels.items():
                    if key in result and result[key] not in (None, ""):
                        value = result[key]
                        # 如果是布尔值，转为“是/否”
                        if isinstance(value, bool):
                            value = "是" if value else "否"
                        else:
                            value = str(value)
                        display_items.append((label, value))

                # 2. 处理未知字段（不在 feature_labels 中的）
                for key, value in result.items():
                    if key not in feature_labels and key not in ("summary","summary_confidence", "tags"):  # 排除 summary/tags 等主字段
                        if value is not None and value != "":
                            # 简单美化字段名：如 "extra_field" → "Extra Field"
                            pretty_key = key.replace("_", " ").capitalize()
                            if isinstance(value, bool):
                                value = "是" if value else "否"
                            else:
                                value = str(value)
                            display_items.append((pretty_key, value))

                # 3. 渲染
                if display_items:
                    st.markdown("##### 🎯 内容特征")
                    for label, value in display_items:
                        st.markdown(f"**{label}**：{value}")
                    st.divider()
                # 安全检测
                st.markdown("##### ⚠️ 安全检测")
                risk_level = result.get("risk_level", "低")
                risk_color = {"低": "#4ade80", "中": "#fbbf24", "高": "#ef4444"}.get(risk_level, "#9ca3af")
                st.markdown(
                    f'<span style="display:inline-block; background:{risk_color}20; color:{risk_color}; '
                    f'padding:4px 12px; border-radius:20px; font-weight:500;">'
                    f'违规风险:{risk_level}</span>',
                    unsafe_allow_html=True
                )
                sensitive_words = result.get("sensitive_words", [])
                if sensitive_words:
                    st.write("敏感词:" + ", ".join(sensitive_words))
                else:
                    st.info("未检测到敏感内容")
                st.divider()

                
                # # 用户反馈
                # st.markdown("💬 **结果反馈**")
                # cols = st.columns(3)
                # with cols[0]:
                #     if st.button("摘要不准", key="fb1"):
                #         st.toast("感谢反馈！我们将优化摘要模型。")
                # with cols[1]:
                #     if st.button("标签不相关", key="fb2"):
                #         st.toast("感谢反馈！标签系统将进行迭代。")
                # with cols[2]:
                #     if st.button("其他问题", key="fb3"):
                #         st.toast("请通过内部渠道提交详细反馈。")
            
            except Exception as e:
                status_text.empty()
                progress_bar.empty()
                st.error(f"❌ 分析失败: {str(e)}")
                # 生产环境建议记录日志,而非显示 traceback
                # logger.error("Analysis failed", exc_info=True)


# ====== 页脚 ======
st.markdown("---")
st.caption("© 2025 视频智能分析平台 | 基于 PaddleOCR + Qwen | 数据不出域 · 安全合规")