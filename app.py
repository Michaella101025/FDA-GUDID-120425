# GUDID 資料品質分析多代理系統 (Streamlit / Hugging Face Space)
# - Multi-agent FDA GUDID data quality analysis & visualization
# - WOW UI (light/dark, 20 flower themes via Jackslot, EN/繁中)
# - 31 agents from agents.yaml
# - Multi-provider LLMs: Gemini, OpenAI, Anthropic, Grok (xAI)
# - AI Note Keeper with multiple AI tools & two AI Magic features

import os
import json
import random
import textwrap
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import streamlit as st
import altair as alt
import yaml

# --- Optional AI SDK imports (handled gracefully) -----------------------------
try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import anthropic
except ImportError:
    anthropic = None


# === CONSTANTS ===============================================================

APP_TITLE = "GUDID 資料品質分析多代理系統 (Multi-Agent GUDID Quality Lab)"

# Supported "global" models for quick selection
GLOBAL_MODEL_OPTIONS = [
    "gpt-4o-mini",
    "gpt-4.1-mini",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "claude-3-5-sonnet-20241022",
    "claude-3-haiku-20240307",
    "grok-4-fast-reasoning",
    "grok-3-mini",
]

DEFAULT_MAX_TOKENS = 12000

# --- Language pack -----------------------------------------------------------
TEXT = {
    "en": {
        "app_title": APP_TITLE,
        "analysis_tab": "Analysis Workspace",
        "dashboard_tab": "Dashboard",
        "logs_tab": "Agent Logs",
        "notes_tab": "AI Note Keeper",
        "language": "Language",
        "theme": "Theme",
        "theme_light": "Light",
        "theme_dark": "Dark",
        "flower_theme": "Flower Style (Jackslot)",
        "spin": "Spin Jackslot",
        "dataset_section": "Dataset Management",
        "use_mock": "Load mock GUDID sample",
        "upload_csv": "Upload GUDID CSV",
        "download_csv": "Download edited CSV",
        "data_editor": "Edit dataset (up to 300 rows for performance)",
        "agents_section": "Agents Configuration & Execution",
        "agents_status": "Agents Overview",
        "run_selected": "Run selected agents (sequential)",
        "clear_results": "Clear results",
        "api_keys": "API Keys (stored only in this browser session)",
        "api_from_env": "Loaded from environment",
        "api_enter_key": "Enter API key",
        "overall_risk": "Overall Risk Indicator",
        "risk_low": "Low",
        "risk_medium": "Medium",
        "risk_high": "High",
        "issues_total": "Total issues detected",
        "issues_by_severity": "Issues by severity",
        "agent_performance": "Agent performance (avg risk score)",
        "no_results": "No agent results yet. Run analysis first.",
        "note_source": "Original Note (paste or type)",
        "transform_md": "Transform to Markdown",
        "note_md": "Transformed Markdown (editable)",
        "note_preview": "Markdown Preview",
        "ai_tools": "AI Note Tools",
        "ai_formatting": "AI Formatting",
        "ai_keywords": "AI Keywords",
        "ai_summary": "AI Summary",
        "ai_entities": "AI Entities",
        "ai_chat": "AI Chat",
        "ai_magic": "AI Magic",
        "ai_magic_1": "AI Compliance Checklist",
        "ai_magic_2": "AI Insight Cards",
        "summary_prompt": "Custom summary prompt",
        "max_tokens": "Max tokens",
        "select_model": "Select model",
        "chat_question": "Ask about this note",
        "run": "Run",
        "logs": "Execution Logs",
        "no_dataset": "No dataset loaded. Please load mock data or upload a CSV.",
        "agent_input_pipeline": "Pipeline input (from previous agent or custom)",
        "agent_custom_prompt": "Custom user prompt (optional)",
        "agent_system_prompt": "System prompt (from agents.yaml, editable for this run)",
        "agent_model": "Model for this agent",
        "agent_max_tokens": "Max tokens for this agent",
        "agent_run": "Run this agent",
        "agent_output_editable": "Editable output (used as pipeline for next agent if you choose)",
        "agent_set_pipeline": "Use as pipeline input for next agent",
        "view_mode": "View mode",
        "view_text": "Text",
        "view_markdown": "Markdown",
        "status_badge": "WOW Status Indicators",
        "agents_loaded": "Agents loaded",
        "theme_active": "Active flower theme",
        "api_missing": "Some API keys missing. You can still use providers with keys set.",
        "ai_error": "AI call failed",
    },
    "zh": {
        "app_title": APP_TITLE,
        "analysis_tab": "分析工作區",
        "dashboard_tab": "儀表板",
        "logs_tab": "代理人日誌",
        "notes_tab": "AI 筆記小幫手",
        "language": "語言",
        "theme": "主題",
        "theme_light": "亮色",
        "theme_dark": "暗色",
        "flower_theme": "花卉風格 (Jackslot 拉霸)",
        "spin": "拉霸抽花卉主題",
        "dataset_section": "資料集管理",
        "use_mock": "載入內建 GUDID 範例資料",
        "upload_csv": "上傳 GUDID CSV 檔案",
        "download_csv": "下載編輯後 CSV",
        "data_editor": "線上編輯資料 (為效能考量最多顯示 300 筆)",
        "agents_section": "代理人設定與執行",
        "agents_status": "代理人總覽",
        "run_selected": "依序執行勾選的代理人",
        "clear_results": "清除分析結果",
        "api_keys": "API 金鑰 (僅儲存於此瀏覽器 Session)",
        "api_from_env": "已由環境變數載入",
        "api_enter_key": "請輸入 API Key",
        "overall_risk": "整體風險指標",
        "risk_low": "低風險",
        "risk_medium": "中風險",
        "risk_high": "高風險",
        "issues_total": "偵測到的問題總數",
        "issues_by_severity": "依嚴重程度分佈",
        "agent_performance": "代理人績效 (平均風險分數)",
        "no_results": "尚無代理人結果，請先執行分析。",
        "note_source": "原始筆記內容 (貼上或輸入)",
        "transform_md": "轉換成 Markdown",
        "note_md": "轉換後的 Markdown (可編輯)",
        "note_preview": "Markdown 預覽",
        "ai_tools": "AI 筆記工具",
        "ai_formatting": "AI 排版優化",
        "ai_keywords": "AI 關鍵字",
        "ai_summary": "AI 摘要",
        "ai_entities": "AI 實體抽取",
        "ai_chat": "AI 對話",
        "ai_magic": "AI 魔法",
        "ai_magic_1": "AI 法規檢查清單",
        "ai_magic_2": "AI 洞見卡片",
        "summary_prompt": "自訂摘要提示詞",
        "max_tokens": "最大 Token 數",
        "select_model": "選擇模型",
        "chat_question": "針對此筆記提問",
        "run": "執行",
        "logs": "執行日誌",
        "no_dataset": "尚未載入資料集，請載入範例或上傳 CSV。",
        "agent_input_pipeline": "管線輸入 (來自上一個代理人或自行輸入)",
        "agent_custom_prompt": "自訂使用者提示詞 (選填)",
        "agent_system_prompt": "系統提示詞 (來自 agents.yaml，本次可編輯)",
        "agent_model": "此代理人使用的模型",
        "agent_max_tokens": "此代理人最大 Token 數",
        "agent_run": "執行此代理人",
        "agent_output_editable": "可編輯輸出內容 (若選擇可作為下一代理人的輸入)",
        "agent_set_pipeline": "作為下一個代理人的管線輸入",
        "view_mode": "檢視模式",
        "view_text": "純文字",
        "view_markdown": "Markdown",
        "status_badge": "WOW 狀態指示器",
        "agents_loaded": "已載入代理人數",
        "theme_active": "目前花卉主題",
        "api_missing": "部分 API Key 尚未設定，僅可使用已設定廠商。",
        "ai_error": "AI 呼叫失敗",
    },
}

# --- 20 Flower themes for Jackslot -------------------------------------------
FLOWER_THEMES = [
    {"id": 0, "name": "Sakura Breeze", "emoji": "🌸", "bg": "#fff5f8", "accent": "#ff7aa2"},
    {"id": 1, "name": "Lavender Field", "emoji": "💜", "bg": "#f5f0ff", "accent": "#8b5cf6"},
    {"id": 2, "name": "Sunflower Glow", "emoji": "🌻", "bg": "#fffbea", "accent": "#f59e0b"},
    {"id": 3, "name": "Rose Garden", "emoji": "🌹", "bg": "#fff1f2", "accent": "#e11d48"},
    {"id": 4, "name": "Lotus Calm", "emoji": "🪷", "bg": "#ecfeff", "accent": "#06b6d4"},
    {"id": 5, "name": "Orchid Mist", "emoji": "🪻", "bg": "#f9f5ff", "accent": "#a855f7"},
    {"id": 6, "name": "Peony Blush", "emoji": "🌺", "bg": "#fff1f7", "accent": "#db2777"},
    {"id": 7, "name": "Daisy Meadow", "emoji": "🌼", "bg": "#fefce8", "accent": "#eab308"},
    {"id": 8, "name": "Tulip Spring", "emoji": "🌷", "bg": "#fff7ed", "accent": "#f97316"},
    {"id": 9, "name": "Hydrangea Sky", "emoji": "🩵", "bg": "#eff6ff", "accent": "#3b82f6"},
    {"id": 10, "name": "Camellia Silk", "emoji": "🌺", "bg": "#fdf2f8", "accent": "#ec4899"},
    {"id": 11, "name": "Magnolia Dawn", "emoji": "🌸", "bg": "#faf5ff", "accent": "#7c3aed"},
    {"id": 12, "name": "Iris Twilight", "emoji": "🦋", "bg": "#eef2ff", "accent": "#6366f1"},
    {"id": 13, "name": "Poppy Fire", "emoji": "🌺", "bg": "#fff1f2", "accent": "#f97373"},
    {"id": 14, "name": "Marigold Sunset", "emoji": "🌼", "bg": "#fffbeb", "accent": "#f59e0b"},
    {"id": 15, "name": "Bluebell Forest", "emoji": "🔔", "bg": "#eff6ff", "accent": "#2563eb"},
    {"id": 16, "name": "Cherry Blossom Night", "emoji": "🌸", "bg": "#020617", "accent": "#fb7185"},
    {"id": 17, "name": "Moonlight Lily", "emoji": "🤍", "bg": "#020617", "accent": "#e5e7eb"},
    {"id": 18, "name": "Wildflower Mix", "emoji": "💐", "bg": "#f9fafb", "accent": "#10b981"},
    {"id": 19, "name": "Desert Bloom", "emoji": "🌵", "bg": "#fffbf5", "accent": "#f97316"},
]


# === SESSION STATE INIT ======================================================

def init_session_state():
    ss = st.session_state
    if "lang" not in ss:
        ss.lang = "zh"  # default Traditional Chinese per spec
    if "ui_theme" not in ss:
        ss.ui_theme = "light"
    if "flower_theme_id" not in ss:
        ss.flower_theme_id = 0
    if "agents" not in ss:
        ss.agents = []
    if "dataset_df" not in ss:
        ss.dataset_df = None
    if "agent_results" not in ss:
        ss.agent_results = {}  # agent_id -> result dict
    if "logs" not in ss:
        ss.logs = []
    if "selected_agent_ids" not in ss:
        ss.selected_agent_ids = set()
    if "pipeline_input" not in ss:
        ss.pipeline_input = ""
    # API keys input (from UI, not env)
    if "api_keys_ui" not in ss:
        ss.api_keys_ui = {
            "openai": "",
            "gemini": "",
            "anthropic": "",
            "grok": "",
        }
    # Note keeper
    if "note_raw" not in ss:
        ss.note_raw = ""
    if "note_md" not in ss:
        ss.note_md = ""
    if "note_view_mode" not in ss:
        ss.note_view_mode = "markdown"
    if "note_ai_summary_prompt" not in ss:
        ss.note_ai_summary_prompt = (
            "You are an expert FDA medical device data quality analyst. "
            "Summarize the note in well-structured markdown. Highlight key findings, "
            "data-quality concerns, and regulatory relevance. Include a short bullet list "
            "of action items. Render important keywords in coral color using "
            "<span style=\"color: coral\">keyword</span>."
        )
    if "note_ai_model" not in ss:
        ss.note_ai_model = "gpt-4o-mini"
    if "note_ai_max_tokens" not in ss:
        ss.note_ai_max_tokens = 2000


# === UTILITY FUNCTIONS =======================================================

def t(key: str) -> str:
    """Translate using current language."""
    lang = st.session_state.get("lang", "zh")
    return TEXT.get(lang, TEXT["zh"]).get(key, key)


def get_active_theme() -> Dict[str, Any]:
    theme_id = st.session_state.get("flower_theme_id", 0)
    return FLOWER_THEMES[theme_id % len(FLOWER_THEMES)]


def apply_custom_css():
    """Inject WOW UI CSS based on theme + flower style."""
    theme = get_active_theme()
    ui_theme = st.session_state.get("ui_theme", "light")
    bg = theme["bg"]
    accent = theme["accent"]
    text_color = "#0f172a" if ui_theme == "light" else "#e5e7eb"
    card_bg = "rgba(255,255,255,0.85)" if ui_theme == "light" else "rgba(15,23,42,0.9)"
    border_color = accent

    css = f"""
    <style>
    .stApp {{
        background: {bg};
        color: {text_color};
    }}
    .theme-card {{
        background: {card_bg};
        border-radius: 12px;
        border: 1px solid {border_color};
        padding: 1rem;
        box-shadow: 0 12px 30px rgba(15,23,42,0.12);
    }}
    .wow-badge {{
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.3rem 0.7rem;
        border-radius: 999px;
        background: linear-gradient(135deg, {accent}33, {accent}11);
        border: 1px solid {accent}55;
        font-size: 0.8rem;
        font-weight: 600;
        color: {text_color};
    }}
    .wow-risk-low {{
        color: #16a34a;
    }}
    .wow-risk-medium {{
        color: #f59e0b;
    }}
    .wow-risk-high {{
        color: #dc2626;
    }}
    .jackslot-display {{
        font-size: 1.2rem;
        font-weight: 600;
        padding: 0.4rem 0.7rem;
        border-radius: 999px;
        border: 1px dashed {accent};
        background: rgba(15,23,42,0.02);
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def log(message: str):
    """Append message to execution logs."""
    st.session_state.logs.append(message)


def load_agents_from_yaml(path: str = "agents.yaml") -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        # Fallback minimal placeholder agents so UI still works
        return [
            {
                "id": "rule_validator",
                "name": "Rule Validator",
                "description": "Validates GUDID records against basic structural rules.",
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "system_prompt": "You are a rule-based validator for GUDID device records.",
                "enabled": True,
            },
            {
                "id": "risk_assessor",
                "name": "Risk Assessor",
                "description": "Estimates data-quality risk per record.",
                "provider": "openai",
                "model": "gpt-4o-mini",
                "system_prompt": "You assess data quality risk of GUDID records.",
                "enabled": True,
            },
        ]
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or []
    # Expect a list of 31 agents, but do not enforce count
    agents = []
    for a in data:
        agent = {
            "id": a.get("id"),
            "name": a.get("name", a.get("id", "unnamed-agent")),
            "description": a.get("description", ""),
            "provider": a.get("provider", "openai"),
            "model": a.get("model", "gpt-4o-mini"),
            "system_prompt": a.get("system_prompt", ""),
            "enabled": bool(a.get("enabled", True)),
        }
        agents.append(agent)
    return agents


def build_mock_dataset() -> pd.DataFrame:
    """Simple built-in GUDID-like sample."""
    data = [
        {
            "primary_di": "12345678901234",
            "brand_name": "CardioSense Monitor",
            "device_description": "Non-invasive cardiac monitoring device.",
            "mri_safety_status": "MR Safe",
            "catalog_number": "CS-100",
            "version_model_number": "V1.0",
        },
        {
            "primary_di": "98765432109876",
            "brand_name": "NeuroScan EEG",
            "device_description": "Electroencephalograph for neurological diagnostics.",
            "mri_safety_status": "MR Conditional",
            "catalog_number": "NS-EEG",
            "version_model_number": "NS-2.1",
        },
        {
            "primary_di": "55555555555555",
            "brand_name": "SpineFix Implant",
            "device_description": "Implantable device for spinal fusion.",
            "mri_safety_status": "",
            "catalog_number": "SF-IMP",
            "version_model_number": "SF-3D",
        },
    ]
    return pd.DataFrame(data)


def get_env_api_keys() -> Dict[str, Optional[str]]:
    return {
        "openai": os.getenv("OPENAI_API_KEY"),
        "gemini": os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "grok": os.getenv("XAI_API_KEY"),
    }


def effective_api_key(provider: str) -> Optional[str]:
    """Return API key from env or UI (env takes precedence)."""
    env_keys = get_env_api_keys()
    if env_keys.get(provider):
        return env_keys[provider]
    return st.session_state.api_keys_ui.get(provider, "") or None


def detect_provider_from_model(model: str) -> str:
    m = model.lower()
    if m.startswith("gemini"):
        return "gemini"
    if m.startswith("gpt-4"):
        return "openai"
    if m.startswith("claude") or "anthropic" in m:
        return "anthropic"
    if "grok" in m:
        return "grok"
    # Fallback
    return "openai"


# === LLM CALL WRAPPER ========================================================

def call_llm(
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.2,
) -> str:
    """
    Unified LLM call across providers. Returns text content.
    """
    provider = detect_provider_from_model(model)
    api_key = effective_api_key(provider)
    if not api_key:
        raise RuntimeError(f"No API key configured for provider '{provider}'.")

    if provider == "gemini":
        if genai is None:
            raise RuntimeError("google-generativeai library not installed.")
        genai.configure(api_key=api_key)
        gmodel = genai.GenerativeModel(model)
        response = gmodel.generate_content(
            [{"role": "user", "parts": [system_prompt + "\n\n" + user_prompt]}],
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        )
        return response.text or ""

    elif provider == "openai":
        if OpenAI is None:
            raise RuntimeError("openai library not installed.")
        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return completion.choices[0].message.content

    elif provider == "anthropic":
        if anthropic is None:
            raise RuntimeError("anthropic library not installed.")
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(block.text for block in message.content if hasattr(block, "text"))

    elif provider == "grok":
        # xAI Grok via OpenAI-compatible client
        if OpenAI is None:
            raise RuntimeError("openai library not installed.")
        client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return completion.choices[0].message.content

    else:
        raise RuntimeError(f"Unsupported provider: {provider}")


# === AGENT EXECUTION LOGIC ===================================================

def build_agent_user_prompt(
    base_prompt: str,
    dataset_df: Optional[pd.DataFrame],
    pipeline_input: str,
) -> str:
    """Construct user prompt combining sample data + pipeline text."""
    parts = []
    if base_prompt.strip():
        parts.append(base_prompt.strip())
    if dataset_df is not None and not dataset_df.empty:
        sample = dataset_df.head(10).to_dict(orient="records")
        parts.append(
            "Here is a JSON sample of the current GUDID dataset (up to 10 records):\n"
            + json.dumps(sample, ensure_ascii=False, indent=2)
        )
    if pipeline_input.strip():
        parts.append(
            "Additional context / previous agent output to consider:\n"
            + pipeline_input.strip()
        )
    # Instruct JSON schema for dashboard
    schema_instructions = textwrap.dedent(
        """
        Respond in **strict JSON** with this schema:
        {
          "summary": "markdown string summarizing your findings",
          "issues": [
            {
              "id": "string",
              "title": "string",
              "description": "string",
              "severity": "low|medium|high",
              "record_ids": ["primary_di1", "primary_di2", "..."]
            }
          ],
          "metrics": {
            "risk_score": 0-100 integer
          }
        }
        """
    )
    parts.append(schema_instructions)
    return "\n\n".join(parts)


def run_single_agent(
    agent: Dict[str, Any],
    model_override: Optional[str],
    max_tokens: int,
    custom_system_prompt: str,
    custom_user_prompt: str,
    dataset_df: Optional[pd.DataFrame],
    pipeline_input: str,
) -> Dict[str, Any]:
    """Execute one agent and parse result."""
    model = model_override or agent["model"]
    system_prompt = custom_system_prompt or agent.get("system_prompt", "")
    user_prompt = build_agent_user_prompt(custom_user_prompt, dataset_df, pipeline_input)

    log(f"Running agent '{agent['name']}' with model {model}...")
    raw = call_llm(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=max_tokens,
    )

    parsed = None
    error = None
    try:
        parsed = json.loads(raw)
    except Exception as e:
        error = f"JSON parse error: {e}"

    result = {
        "raw": raw,
        "parsed": parsed,
        "error": error,
    }
    return result


def aggregate_dashboard_metrics(agent_results: Dict[str, Any]) -> Tuple[int, Dict[str, int], Dict[str, float]]:
    """
    Compute total issues, severity counts, and average risk scores per agent.
    """
    total_issues = 0
    severity_counts = {"low": 0, "medium": 0, "high": 0}
    agent_risk = {}  # agent_id -> avg risk score

    for agent_id, res in agent_results.items():
        parsed = res.get("parsed")
        if not parsed:
            continue
        issues = parsed.get("issues") or []
        total_issues += len(issues)
        for issue in issues:
            sv = str(issue.get("severity", "")).lower()
            if sv in severity_counts:
                severity_counts[sv] += 1
        metrics = parsed.get("metrics") or {}
        risk = metrics.get("risk_score")
        if isinstance(risk, (int, float)):
            agent_risk[agent_id] = float(risk)

    return total_issues, severity_counts, agent_risk


def risk_label_from_score(score: float) -> str:
    if score < 33:
        return "low"
    if score < 66:
        return "medium"
    return "high"


# === NOTE KEEPER AI UTILS ====================================================

def simple_text_to_markdown(text: str) -> str:
    """Non-AI baseline: split into paragraphs and bullet lines."""
    lines = [l.rstrip() for l in text.splitlines()]
    md_lines = []
    for l in lines:
        if not l.strip():
            md_lines.append("")
        elif l.strip().startswith(("-", "*", "•")):
            md_lines.append("- " + l.lstrip(" -*•"))
        else:
            md_lines.append(l)
    return "\n".join(md_lines)


def run_note_ai_tool(
    tool_type: str,
    note_text: str,
    base_prompt: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    extra_user_input: str = "",
) -> str:
    """
    Generic executor for note-related AI tools.
    tool_type: 'format', 'keywords', 'summary', 'entities', 'chat', 'magic1', 'magic2'
    """
    if not note_text.strip():
        return "Note is empty."

    if model is None:
        model = st.session_state.note_ai_model
    if max_tokens is None:
        max_tokens = st.session_state.note_ai_max_tokens

    if tool_type == "summary":
        system_prompt = (
            "You are an expert FDA/GUDID medical device data-quality analyst.\n"
            "Summarize the note in well-structured markdown. "
            "Highlight sections, bullet lists, and clearly mark **data-quality issues**, "
            "**regulatory considerations**, and **recommended actions**.\n"
            "Important keywords MUST be wrapped in "
            "<span style=\"color: coral\">keyword</span> (HTML) so they render in coral."
        )
        user_prompt = base_prompt or st.session_state.note_ai_summary_prompt
        user_prompt = user_prompt + "\n\nHere is the note:\n" + note_text

    elif tool_type == "format":
        system_prompt = (
            "You are a meticulous technical editor. Reformat the note into clean markdown "
            "without changing the meaning. Use headings, bullet lists, tables when helpful."
        )
        user_prompt = "Reformat the following note into tidy markdown:\n\n" + note_text

    elif tool_type == "keywords":
        system_prompt = (
            "You extract high-value keywords from FDA / GUDID technical notes. "
            "Return a short markdown section of keywords grouped by category. "
            "Render each keyword in coral color using "
            "<span style=\"color: coral\">keyword</span>."
        )
        user_prompt = "Extract domain-specific keywords from this note:\n\n" + note_text

    elif tool_type == "entities":
        system_prompt = (
            "You are an information extraction specialist. From the note, identify **20 key entities** "
            "(e.g., device names, identifiers, manufacturers, attributes, risks, standards, processes). "
            "Return markdown with one section per entity:\n"
            "### Entity Name\n"
            "- Type: ...\n- Context: ...\n- Related risks or quality concerns: ...\n"
        )
        user_prompt = "Extract 20 entities with context from this note:\n\n" + note_text

    elif tool_type == "chat":
        system_prompt = (
            "You are an assistant helping the user interpret an FDA / GUDID-related note.\n"
            "Always base your answers on the provided note content. If something is not in the note, "
            "say that it is not specified explicitly."
        )
        user_prompt = (
            "Here is the note:\n\n" + note_text + "\n\n"
            "User question:\n" + extra_user_input
        )

    elif tool_type == "magic1":
        # AI Compliance Checklist
        system_prompt = (
            "You are an FDA regulatory specialist. From the note, derive a practical compliance "
            "and data-quality checklist related to GUDID / medical device submissions."
        )
        user_prompt = textwrap.dedent(
            f"""
            Based on the following note, create a detailed markdown checklist for:
            - Data completeness and GUDID field coverage
            - Identifier consistency and uniqueness
            - Labeling and MRI safety status consistency
            - Risk-related flags and follow-up items

            Use markdown checkboxes like:
            - [ ] Item

            Note:
            {note_text}
            """
        )

    elif tool_type == "magic2":
        # AI Insight Cards
        system_prompt = (
            "You are a senior data quality lead. Extract the most important insights "
            "from the note and present them as concise 'insight cards'."
        )
        user_prompt = textwrap.dedent(
            f"""
            From the note, produce **5–8 insight cards** in markdown with this pattern:

            ### Insight #N: Short Title
            - Category: (Data Quality | Regulatory | Risk | Process | Other)
            - Impact: Low / Medium / High
            - Description: ...
            - Recommended next action: ...

            Make the content concrete and, when relevant, connect to GUDID concepts.

            Note:
            {note_text}
            """
        )
    else:
        return "Unsupported tool."

    return call_llm(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=max_tokens,
    )


# === SIDEBAR UI ==============================================================

def render_sidebar():
    theme = get_active_theme()
    st.sidebar.markdown(
        f"### {t('app_title')}\n"
        f"<span class='wow-badge'>{theme['emoji']} {t('status_badge')}</span>",
        unsafe_allow_html=True,
    )

    # Language & theme
    st.sidebar.markdown("----")
    lang = st.sidebar.radio(
        t("language"),
        options=["zh", "en"],
        format_func=lambda v: "繁體中文" if v == "zh" else "English",
        index=0 if st.session_state.lang == "zh" else 1,
        key="lang",
    )
    st.session_state.lang = lang

    ui_theme = st.sidebar.radio(
        t("theme"),
        options=["light", "dark"],
        format_func=lambda v: t("theme_light") if v == "light" else t("theme_dark"),
        index=0 if st.session_state.ui_theme == "light" else 1,
        key="ui_theme",
    )
    st.session_state.ui_theme = ui_theme

    # Flower Jackslot
    st.sidebar.markdown("----")
    st.sidebar.markdown(f"**{t('flower_theme')}**")
    col1, col2 = st.sidebar.columns([2, 1])
    with col1:
        th = get_active_theme()
        st.markdown(
            f"<div class='jackslot-display'>{th['emoji']} {th['name']}</div>",
            unsafe_allow_html=True,
        )
    with col2:
        if st.button(t("spin"), key="spin_jackslot"):
            st.session_state.flower_theme_id = random.randint(0, len(FLOWER_THEMES) - 1)

    # API Keys
    st.sidebar.markdown("----")
    st.sidebar.markdown(f"**{t('api_keys')}**")
    env_keys = get_env_api_keys()
    any_missing = False
    for provider, label in [
        ("openai", "OpenAI"),
        ("gemini", "Gemini"),
        ("anthropic", "Anthropic"),
        ("grok", "Grok (xAI)"),
    ]:
        has_env = bool(env_keys.get(provider))
        if has_env:
            st.sidebar.markdown(f"- {label}: ✅ {t('api_from_env')}")
        else:
            any_missing = True
            st.sidebar.text_input(
                f"{label} {t('api_enter_key')}",
                type="password",
                key=f"api_key_input_{provider}",
            )
            st.session_state.api_keys_ui[provider] = st.session_state.get(
                f"api_key_input_{provider}", ""
            )
    if any_missing:
        st.sidebar.markdown(f"<span style='color:#f97316'>{t('api_missing')}</span>", unsafe_allow_html=True)

    # Dataset controls
    st.sidebar.markdown("----")
    st.sidebar.markdown(f"**{t('dataset_section')}**")
    if st.sidebar.button(t("use_mock"), key="load_mock"):
        st.session_state.dataset_df = build_mock_dataset()
        log("Loaded mock GUDID dataset.")

    uploaded = st.sidebar.file_uploader(
        t("upload_csv"), type=["csv"], key="csv_uploader"
    )
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.session_state.dataset_df = df
        log(f"Uploaded CSV with {len(df)} rows.")

    if st.session_state.dataset_df is not None:
        csv = st.session_state.dataset_df.to_csv(index=False).encode("utf-8-sig")
        st.sidebar.download_button(
            t("download_csv"), data=csv, file_name="gudid_edited.csv", mime="text/csv"
        )


# === MAIN TABS ==============================================================

def render_analysis_workspace():
    st.markdown(f"## {t('analysis_tab')}")
    theme = get_active_theme()

    # WOW status bar
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            "<div class='theme-card'><b>"
            + t("theme_active")
            + f"</b><br/>{theme['emoji']} {theme['name']}</div>",
            unsafe_allow_html=True,
        )
    with col2:
        n_agents = len(st.session_state.agents)
        st.markdown(
            "<div class='theme-card'><b>"
            + t("agents_loaded")
            + f"</b><br/>{n_agents}</div>",
            unsafe_allow_html=True,
        )
    with col3:
        issues_total, severity_counts, agent_risk = aggregate_dashboard_metrics(
            st.session_state.agent_results
        )
        if agent_risk:
            avg_risk = sum(agent_risk.values()) / len(agent_risk)
            label = risk_label_from_score(avg_risk)
            label_text = {
                "low": t("risk_low"),
                "medium": t("risk_medium"),
                "high": t("risk_high"),
            }[label]
            css_class = {
                "low": "wow-risk-low",
                "medium": "wow-risk-medium",
                "high": "wow-risk-high",
            }[label]
            st.markdown(
                f"<div class='theme-card'><b>{t('overall_risk')}</b><br/>"
                f"<span class='{css_class}'>{label_text} ({avg_risk:.1f})</span></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='theme-card'><b>{t('overall_risk')}</b><br/>–</div>",
                unsafe_allow_html=True,
            )

    st.markdown("----")

    # Dataset editor
    st.markdown(f"### {t('dataset_section')}")
    if st.session_state.dataset_df is None:
        st.info(t("no_dataset"))
    else:
        df = st.session_state.dataset_df
        show_df = df.head(300)
        edited = st.data_editor(
            show_df,
            num_rows="dynamic",
            use_container_width=True,
            key="dataset_editor",
        )
        # Update session state only for edited subset (simple strategy)
        st.session_state.dataset_df.iloc[: len(edited)] = edited.values

    st.markdown("----")

    # Agents UI
    st.markdown(f"### {t('agents_section')}")
    if not st.session_state.agents:
        st.warning("No agents loaded from agents.yaml (fallback minimal agents in use).")

    # Global pipeline input
    with st.expander(t("agent_input_pipeline"), expanded=False):
        st.session_state.pipeline_input = st.text_area(
            t("agent_input_pipeline"),
            value=st.session_state.pipeline_input,
            height=120,
            key="pipeline_input_text",
        )

    # Global controls
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.markdown(f"**{t('agents_status')}**")
    with c2:
        if st.button(t("run_selected"), key="run_selected_agents"):
            run_selected_agents()
    with c3:
        if st.button(t("clear_results"), key="clear_agent_results"):
            st.session_state.agent_results = {}
            log("Cleared agent results.")

    # Per-agent panels
    for agent in st.session_state.agents:
        agent_id = agent["id"]
        enabled = agent.get("enabled", True)
        default_selected = enabled
        checked = st.checkbox(
            f"[{agent['id']}] {agent['name']}",
            value=agent_id in st.session_state.selected_agent_ids or default_selected,
            key=f"select_agent_{agent_id}",
        )
        if checked:
            st.session_state.selected_agent_ids.add(agent_id)
        else:
            st.session_state.selected_agent_ids.discard(agent_id)

        with st.expander(f"⚙️ {agent['name']} – {agent['description']}", expanded=False):
            # Per-agent controls
            col_a, col_b = st.columns(2)
            with col_a:
                custom_system = st.text_area(
                    t("agent_system_prompt"),
                    value=agent.get("system_prompt", ""),
                    height=120,
                    key=f"sys_prompt_{agent_id}",
                )
                custom_user = st.text_area(
                    t("agent_custom_prompt"),
                    value="",
                    height=120,
                    key=f"user_prompt_{agent_id}",
                )
            with col_b:
                model_choice = st.selectbox(
                    t("agent_model"),
                    options=["(agent default)"] + GLOBAL_MODEL_OPTIONS,
                    index=0,
                    key=f"model_{agent_id}",
                )
                max_tokens = st.number_input(
                    t("agent_max_tokens"),
                    min_value=256,
                    max_value=120000,
                    value=DEFAULT_MAX_TOKENS,
                    step=256,
                    key=f"max_tokens_{agent_id}",
                )
                view_mode = st.radio(
                    t("view_mode"),
                    options=["text", "markdown"],
                    format_func=lambda x: t("view_text") if x == "text" else t("view_markdown"),
                    index=1,
                    key=f"view_mode_{agent_id}",
                )

                if st.button(t("agent_run"), key=f"run_agent_{agent_id}"):
                    try:
                        model_override = None if model_choice == "(agent default)" else model_choice
                        res = run_single_agent(
                            agent=agent,
                            model_override=model_override,
                            max_tokens=int(max_tokens),
                            custom_system_prompt=custom_system,
                            custom_user_prompt=custom_user,
                            dataset_df=st.session_state.dataset_df,
                            pipeline_input=st.session_state.pipeline_input,
                        )
                        st.session_state.agent_results[agent_id] = res
                        log(f"Agent '{agent['name']}' finished.")
                    except Exception as e:
                        st.error(f"{t('ai_error')}: {e}")
                        log(f"Error running agent '{agent['name']}': {e}")

            # Output editor
            res = st.session_state.agent_results.get(agent_id)
            if res:
                raw = res.get("raw", "")
                editable_key = f"agent_output_editable_{agent_id}"
                current_edit = st.text_area(
                    t("agent_output_editable"),
                    value=raw,
                    height=200,
                    key=editable_key,
                )
                col_x, col_y = st.columns([1, 1])
                with col_x:
                    if st.button(t("agent_set_pipeline"), key=f"set_pipeline_{agent_id}"):
                        st.session_state.pipeline_input = current_edit
                        st.success("Pipeline input updated for next agent.")
                with col_y:
                    if view_mode == "markdown":
                        st.markdown("#### Preview")
                        st.markdown(current_edit, unsafe_allow_html=True)


def run_selected_agents():
    """Sequentially execute all selected agents."""
    if not st.session_state.dataset_df is not None and not st.session_state.pipeline_input:
        st.warning(t("no_dataset"))
        return

    for agent in st.session_state.agents:
        agent_id = agent["id"]
        if agent_id not in st.session_state.selected_agent_ids:
            continue
        custom_system = st.session_state.get(f"sys_prompt_{agent_id}", agent.get("system_prompt", ""))
        custom_user = st.session_state.get(f"user_prompt_{agent_id}", "")
        model_choice = st.session_state.get(f"model_{agent_id}", "(agent default)")
        max_tokens = int(
            st.session_state.get(f"max_tokens_{agent_id}", DEFAULT_MAX_TOKENS)
        )
        model_override = None if model_choice == "(agent default)" else model_choice
        try:
            res = run_single_agent(
                agent=agent,
                model_override=model_override,
                max_tokens=max_tokens,
                custom_system_prompt=custom_system,
                custom_user_prompt=custom_user,
                dataset_df=st.session_state.dataset_df,
                pipeline_input=st.session_state.pipeline_input,
            )
            st.session_state.agent_results[agent_id] = res
            # Auto-update pipeline with this agent's raw output
            st.session_state.pipeline_input = res.get("raw", "")
            log(f"Sequential run: agent '{agent['name']}' finished.")
        except Exception as e:
            log(f"Sequential run: error in agent '{agent['name']}': {e}")


def render_dashboard():
    st.markdown(f"## {t('dashboard_tab')}")
    if not st.session_state.agent_results:
        st.info(t("no_results"))
        return

    total_issues, severity_counts, agent_risk = aggregate_dashboard_metrics(
        st.session_state.agent_results
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(t("issues_total"), total_issues)
    with col2:
        low = severity_counts["low"]
        med = severity_counts["medium"]
        high = severity_counts["high"]
        st.metric(t("issues_by_severity"), f"L:{low} / M:{med} / H:{high}")
    with col3:
        if agent_risk:
            avg_risk = sum(agent_risk.values()) / len(agent_risk)
            label = risk_label_from_score(avg_risk)
            label_text = {
                "low": t("risk_low"),
                "medium": t("risk_medium"),
                "high": t("risk_high"),
            }[label]
            st.metric(t("overall_risk"), f"{label_text} ({avg_risk:.1f})")
        else:
            st.metric(t("overall_risk"), "–")

    # Severity pie chart
    sev_df = pd.DataFrame(
        {
            "severity": ["low", "medium", "high"],
            "count": [
                severity_counts["low"],
                severity_counts["medium"],
                severity_counts["high"],
            ],
        }
    )
    pie = (
        alt.Chart(sev_df)
        .mark_arc()
        .encode(
            theta="count:Q",
            color=alt.Color(
                "severity:N",
                scale=alt.Scale(
                    domain=["low", "medium", "high"],
                    range=["#22c55e", "#eab308", "#ef4444"],
                ),
            ),
            tooltip=["severity", "count"],
        )
    )
    st.altair_chart(pie, use_container_width=True)

    # Agent performance bar
    if agent_risk:
        rows = []
        for agent in st.session_state.agents:
            aid = agent["id"]
            if aid in agent_risk:
                rows.append({"agent": agent["name"], "risk_score": agent_risk[aid]})
        if rows:
            df = pd.DataFrame(rows)
            bar = (
                alt.Chart(df)
                .mark_bar()
                .encode(
                    x=alt.X("agent:N", sort="-y"),
                    y=alt.Y("risk_score:Q"),
                    tooltip=["agent", "risk_score"],
                    color=alt.value(get_active_theme()["accent"]),
                )
            )
            st.markdown(f"### {t('agent_performance')}")
            st.altair_chart(bar, use_container_width=True)

    # Detailed cards
    st.markdown("----")
    st.markdown("### Agent Findings")
    for agent in st.session_state.agents:
        aid = agent["id"]
        res = st.session_state.agent_results.get(aid)
        if not res or not res.get("parsed"):
            continue
        parsed = res["parsed"]
        st.markdown(f"#### {agent['name']}")
        st.markdown(parsed.get("summary", ""))
        issues = parsed.get("issues") or []
        if issues:
            for issue in issues:
                sev = issue.get("severity", "medium").lower()
                sev_label = sev.capitalize()
                sev_color = {
                    "low": "#16a34a",
                    "medium": "#f59e0b",
                    "high": "#dc2626",
                }.get(sev, "#6b7280")
                st.markdown(
                    f"<div class='theme-card'>"
                    f"<b>{issue.get('title','(no title)')}</b><br/>"
                    f"<span style='color:{sev_color}'>Severity: {sev_label}</span><br/>"
                    f"{issue.get('description','')}"
                    f"</div>",
                    unsafe_allow_html=True,
                )


def render_logs():
    st.markdown(f"## {t('logs')}")
    if not st.session_state.logs:
        st.info("No logs yet.")
    else:
        st.text_area(
            label=t("logs"),
            value="\n".join(st.session_state.logs),
            height=300,
            key="logs_view",
        )


def render_note_keeper():
    st.markdown(f"## {t('notes_tab')}")

    col_left, col_right = st.columns([3, 2])

    # LEFT: Note & Markdown
    with col_left:
        st.markdown(f"### {t('note_source')}")
        st.session_state.note_raw = st.text_area(
            t("note_source"),
            value=st.session_state.note_raw,
            height=200,
            key="note_raw_text",
        )

        if st.button(t("transform_md"), key="btn_transform_md"):
            # Baseline non-AI transform; user can then apply AI Formatting
            st.session_state.note_md = simple_text_to_markdown(
                st.session_state.note_raw
            )

        st.markdown(f"### {t('note_md')}")
        st.session_state.note_md = st.text_area(
            t("note_md"),
            value=st.session_state.note_md,
            height=220,
            key="note_md_text",
        )

        st.session_state.note_view_mode = st.radio(
            t("view_mode"),
            options=["text", "markdown"],
            format_func=lambda x: t("view_text") if x == "text" else t("view_markdown"),
            index=1 if st.session_state.note_view_mode == "markdown" else 0,
            key="note_view_mode_radio",
        )

        if st.session_state.note_view_mode == "markdown":
            st.markdown(f"### {t('note_preview')}")
            st.markdown(st.session_state.note_md, unsafe_allow_html=True)

    # RIGHT: AI tools
    with col_right:
        st.markdown(f"### {t('ai_tools')}")

        # Shared model / tokens for note tools
        st.selectbox(
            t("select_model"),
            options=GLOBAL_MODEL_OPTIONS,
            index=GLOBAL_MODEL_OPTIONS.index(st.session_state.note_ai_model)
            if st.session_state.note_ai_model in GLOBAL_MODEL_OPTIONS
            else 0,
            key="note_ai_model",
        )
        st.session_state.note_ai_model = st.session_state.note_ai_model
        st.session_state.note_ai_max_tokens = st.number_input(
            t("max_tokens"),
            min_value=256,
            max_value=8000,
            value=st.session_state.note_ai_max_tokens,
            step=256,
            key="note_ai_max_tokens",
        )

        # Tabs for each AI tool
        tab_fmt, tab_kw, tab_sum, tab_ent, tab_chat, tab_magic = st.tabs(
            [
                t("ai_formatting"),
                t("ai_keywords"),
                t("ai_summary"),
                t("ai_entities"),
                t("ai_chat"),
                t("ai_magic"),
            ]
        )

        note_for_ai = st.session_state.note_md or st.session_state.note_raw

        with tab_fmt:
            if st.button(t("ai_formatting"), key="btn_ai_format"):
                try:
                    out = run_note_ai_tool(
                        "format",
                        note_for_ai,
                        model=st.session_state.note_ai_model,
                        max_tokens=int(st.session_state.note_ai_max_tokens),
                    )
                    st.session_state.note_md = out
                    st.success("AI Formatting applied and updated markdown.")
                except Exception as e:
                    st.error(f"{t('ai_error')}: {e}")

        with tab_kw:
            if st.button(t("ai_keywords"), key="btn_ai_keywords"):
                try:
                    out = run_note_ai_tool(
                        "keywords",
                        note_for_ai,
                        model=st.session_state.note_ai_model,
                        max_tokens=int(st.session_state.note_ai_max_tokens),
                    )
                    st.markdown(out, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"{t('ai_error')}: {e}")

        with tab_sum:
            st.text_area(
                t("summary_prompt"),
                value=st.session_state.note_ai_summary_prompt,
                key="note_ai_summary_prompt",
                height=140,
            )
            if st.button(t("ai_summary"), key="btn_ai_summary"):
                try:
                    out = run_note_ai_tool(
                        "summary",
                        note_for_ai,
                        base_prompt=st.session_state.note_ai_summary_prompt,
                        model=st.session_state.note_ai_model,
                        max_tokens=int(st.session_state.note_ai_max_tokens),
                    )
                    st.markdown(out, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"{t('ai_error')}: {e}")

        with tab_ent:
            if st.button(t("ai_entities"), key="btn_ai_entities"):
                try:
                    out = run_note_ai_tool(
                        "entities",
                        note_for_ai,
                        model=st.session_state.note_ai_model,
                        max_tokens=int(st.session_state.note_ai_max_tokens),
                    )
                    st.markdown(out, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"{t('ai_error')}: {e}")

        with tab_chat:
            question = st.text_area(
                t("chat_question"),
                height=100,
                key="note_chat_question",
            )
            if st.button(t("ai_chat"), key="btn_ai_chat"):
                try:
                    out = run_note_ai_tool(
                        "chat",
                        note_for_ai,
                        model=st.session_state.note_ai_model,
                        max_tokens=int(st.session_state.note_ai_max_tokens),
                        extra_user_input=question,
                    )
                    st.markdown(out, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"{t('ai_error')}: {e}")

        with tab_magic:
            st.markdown(f"#### {t('ai_magic_1')}")
            if st.button(t("ai_magic_1"), key="btn_ai_magic1"):
                try:
                    out = run_note_ai_tool(
                        "magic1",
                        note_for_ai,
                        model=st.session_state.note_ai_model,
                        max_tokens=int(st.session_state.note_ai_max_tokens),
                    )
                    st.markdown(out, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"{t('ai_error')}: {e}")

            st.markdown("----")
            st.markdown(f"#### {t('ai_magic_2')}")
            if st.button(t("ai_magic_2"), key="btn_ai_magic2"):
                try:
                    out = run_note_ai_tool(
                        "magic2",
                        note_for_ai,
                        model=st.session_state.note_ai_model,
                        max_tokens=int(st.session_state.note_ai_max_tokens),
                    )
                    st.markdown(out, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"{t('ai_error')}: {e}")


# === MAIN ====================================================================

def main():
    st.set_page_config(
        page_title=APP_TITLE,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_session_state()
    apply_custom_css()

    # Load agents once
    if not st.session_state.agents:
        st.session_state.agents = load_agents_from_yaml()

    render_sidebar()

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            t("analysis_tab"),
            t("dashboard_tab"),
            t("logs_tab"),
            t("notes_tab"),
        ]
    )

    with tab1:
        render_analysis_workspace()
    with tab2:
        render_dashboard()
    with tab3:
        render_logs()
    with tab4:
        render_note_keeper()


if __name__ == "__main__":
    main()
