import os
import random
import json
import requests

import yaml
import pandas as pd
import streamlit as st
import plotly.express as px

import google.generativeai as genai
from openai import OpenAI
import anthropic


# =========================
# 1. THEME & I18N UTILITIES
# =========================

FLOWER_THEMES = [
    {
        "id": "sakura_breeze",
        "name_en": "Sakura Breeze",
        "name_zh": "櫻花微風",
        "primary": "#ff8ba7",
        "accent": "#ffc6ff",
        "bg": "#fff7fb",
        "fg": "#2b2b3c",
    },
    {
        "id": "lotus_lake",
        "name_en": "Lotus Lake",
        "name_zh": "蓮花湖畔",
        "primary": "#4b9c7d",
        "accent": "#ffd6a5",
        "bg": "#f0fff7",
        "fg": "#1f2933",
    },
    {
        "id": "iris_night",
        "name_en": "Iris Night",
        "name_zh": "鳶尾夜色",
        "primary": "#4338ca",
        "accent": "#a855f7",
        "bg": "#eef2ff",
        "fg": "#111827",
    },
    {
        "id": "sunflower_field",
        "name_en": "Sunflower Field",
        "name_zh": "向日葵原野",
        "primary": "#f59e0b",
        "accent": "#fde68a",
        "bg": "#fffbeb",
        "fg": "#1f2937",
    },
    {
        "id": "orchid_mist",
        "name_en": "Orchid Mist",
        "name_zh": "蘭花薄霧",
        "primary": "#a855f7",
        "accent": "#e9d5ff",
        "bg": "#faf5ff",
        "fg": "#111827",
    },
    {
        "id": "rose_garden",
        "name_en": "Rose Garden",
        "name_zh": "玫瑰花園",
        "primary": "#f97373",
        "accent": "#fecaca",
        "bg": "#fff1f2",
        "fg": "#111827",
    },
    {
        "id": "lavender_hill",
        "name_en": "Lavender Hill",
        "name_zh": "薰衣草山丘",
        "primary": "#7c3aed",
        "accent": "#ddd6fe",
        "bg": "#f5f3ff",
        "fg": "#111827",
    },
    {
        "id": "peony_blush",
        "name_en": "Peony Blush",
        "name_zh": "牡丹粉韻",
        "primary": "#ec4899",
        "accent": "#fbcfe8",
        "bg": "#fdf2f8",
        "fg": "#111827",
    },
    {
        "id": "camellia_cloud",
        "name_en": "Camellia Cloud",
        "name_zh": "山茶雲霧",
        "primary": "#e11d48",
        "accent": "#fecdd3",
        "bg": "#fff1f2",
        "fg": "#111827",
    },
    {
        "id": "jasmine_moon",
        "name_en": "Jasmine Moon",
        "name_zh": "茉莉月光",
        "primary": "#22c55e",
        "accent": "#bbf7d0",
        "bg": "#f0fdf4",
        "fg": "#052e16",
    },
    {
        "id": "magnolia_silk",
        "name_en": "Magnolia Silk",
        "name_zh": "玉蘭絲光",
        "primary": "#0ea5e9",
        "accent": "#bae6fd",
        "bg": "#eff6ff",
        "fg": "#0f172a",
    },
    {
        "id": "chrysanthemum_gold",
        "name_en": "Chrysanthemum Gold",
        "name_zh": "菊花金輝",
        "primary": "#facc15",
        "accent": "#fef9c3",
        "bg": "#fefce8",
        "fg": "#422006",
    },
    {
        "id": "violet_dawn",
        "name_en": "Violet Dawn",
        "name_zh": "紫羅蘭黎明",
        "primary": "#6366f1",
        "accent": "#e0e7ff",
        "bg": "#eef2ff",
        "fg": "#020617",
    },
    {
        "id": "plum_blossom",
        "name_en": "Plum Blossom",
        "name_zh": "梅花霜雪",
        "primary": "#be123c",
        "accent": "#fecdd3",
        "bg": "#fff1f2",
        "fg": "#111827",
    },
    {
        "id": "dahlia_flame",
        "name_en": "Dahlia Flame",
        "name_zh": "大理花焰",
        "primary": "#ea580c",
        "accent": "#fed7aa",
        "bg": "#fff7ed",
        "fg": "#111827",
    },
    {
        "id": "poppy_horizon",
        "name_en": "Poppy Horizon",
        "name_zh": "罌粟天際",
        "primary": "#ef4444",
        "accent": "#fecaca",
        "bg": "#fef2f2",
        "fg": "#111827",
    },
    {
        "id": "hydrangea_marine",
        "name_en": "Hydrangea Marine",
        "name_zh": "繡球海藍",
        "primary": "#3b82f6",
        "accent": "#bfdbfe",
        "bg": "#eff6ff",
        "fg": "#0f172a",
    },
    {
        "id": "lotus_twilight",
        "name_en": "Lotus Twilight",
        "name_zh": "暮色蓮影",
        "primary": "#0f766e",
        "accent": "#99f6e4",
        "bg": "#ecfeff",
        "fg": "#022c22",
    },
    {
        "id": "gardenia_sunrise",
        "name_en": "Gardenia Sunrise",
        "name_zh": "梔子朝陽",
        "primary": "#f97316",
        "accent": "#ffedd5",
        "bg": "#fff7ed",
        "fg": "#111827",
    },
    {
        "id": "wisteria_river",
        "name_en": "Wisteria River",
        "name_zh": "紫藤河畔",
        "primary": "#7e22ce",
        "accent": "#e9d5ff",
        "bg": "#faf5ff",
        "fg": "#111827",
    },
]

STRINGS = {
    "en": {
        "title": "GUDID Data Quality Multi-Agent Studio",
        "sidebar_settings": "Settings",
        "language": "Language",
        "theme": "Theme",
        "flower_style": "Flower Style (Jackslot)",
        "light": "Light",
        "dark": "Dark",
        "analysis_tab": "GUDID Multi-Agent Analysis",
        "notes_tab": "AI Note Keeping",
        "dashboard_tab": "Dashboard",
        "agents_tab": "Agents & Chains",
        "api_keys": "API Keys",
    },
    "zh": {
        "title": "GUDID 資料品質分析多代理系統",
        "sidebar_settings": "系統設定",
        "language": "介面語言",
        "theme": "主題模式",
        "flower_style": "花卉風格（Jackslot）",
        "light": "亮色",
        "dark": "暗色",
        "analysis_tab": "GUDID 多代理分析",
        "notes_tab": "AI 筆記助手",
        "dashboard_tab": "儀表板",
        "agents_tab": "代理設定與串鏈",
        "api_keys": "API 金鑰",
    },
}


def t(key: str) -> str:
    lang = st.session_state.get("lang", "zh")
    return STRINGS.get(lang, STRINGS["zh"]).get(key, key)


def init_lang():
    st.session_state.setdefault("lang", "zh")


def init_theme_state():
    st.session_state.setdefault("theme_mode", "light")  # 'light' or 'dark'
    st.session_state.setdefault("flower_theme_id", FLOWER_THEMES[0]["id"])


def get_current_theme():
    current_id = st.session_state.get("flower_theme_id", FLOWER_THEMES[0]["id"])
    return next(ft for ft in FLOWER_THEMES if ft["id"] == current_id)


def jackslot_spin():
    choice = random.choice(FLOWER_THEMES)
    st.session_state["flower_theme_id"] = choice["id"]
    return choice


def inject_theme_css():
    theme = get_current_theme()
    dark = st.session_state.get("theme_mode") == "dark"
    bg = theme["bg"] if not dark else "#020617"
    fg = theme["fg"] if not dark else "#e5e7eb"
    primary = theme["primary"]
    accent = theme["accent"]

    css = f"""
    <style>
    :root {{
      --app-bg: {bg};
      --app-fg: {fg};
      --app-primary: {primary};
      --app-accent: {accent};
      --keyword-coral: coral;
    }}
    .main {{
      background-color: var(--app-bg);
      color: var(--app-fg);
    }}
    [data-testid="stSidebar"] {{
      background: linear-gradient(180deg, {primary}22, {accent}11);
    }}
    .wow-card {{
      border-radius: 0.75rem;
      padding: 1rem 1.25rem;
      border: 1px solid rgba(0,0,0,0.06);
      background: rgba(255,255,255,0.92);
    }}
    .coral-keyword {{
      color: coral;
      font-weight: 600;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# =========================
# 2. LLM PROVIDER WRAPPER
# =========================

class LLMClients:
    def __init__(self, session_state):
        self.gemini_key = os.getenv("GEMINI_API_KEY") or session_state.get("gemini_api_key")
        self.openai_key = os.getenv("OPENAI_API_KEY") or session_state.get("openai_api_key")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY") or session_state.get("anthropic_api_key")
        self.grok_key = os.getenv("GROK_API_KEY") or session_state.get("grok_api_key")

        if self.gemini_key:
            genai.configure(api_key=self.gemini_key)
        self.openai_client = OpenAI(api_key=self.openai_key) if self.openai_key else None
        self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_key) if self.anthropic_key else None

    def _call_gemini(self, model, prompt, max_tokens):
        if not self.gemini_key:
            raise RuntimeError("Gemini API key not configured.")
        model_obj = genai.GenerativeModel(model)
        resp = model_obj.generate_content(prompt, generation_config={"max_output_tokens": max_tokens})
        return resp.text

    def _call_openai(self, model, prompt, max_tokens):
        if not self.openai_client:
            raise RuntimeError("OpenAI API key not configured.")
        resp = self.openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content

    def _call_anthropic(self, model, prompt, max_tokens):
        if not self.anthropic_client:
            raise RuntimeError("Anthropic API key not configured.")
        resp = self.anthropic_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text

    def _call_grok(self, model, prompt, max_tokens):
        if not self.grok_key:
            raise RuntimeError("Grok API key not configured.")
        # NOTE: Update URL/payload if Grok API changes
        headers = {"Authorization": f"Bearer {self.grok_key}"}
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}
        resp = requests.post("https://api.x.ai/v1/chat/completions", json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def call(self, provider, model, prompt, max_tokens=12000):
        if provider == "gemini":
            return self._call_gemini(model, prompt, max_tokens)
        elif provider == "openai":
            return self._call_openai(model, prompt, max_tokens)
        elif provider == "anthropic":
            return self._call_anthropic(model, prompt, max_tokens)
        elif provider == "grok":
            return self._call_grok(model, prompt, max_tokens)
        else:
            raise ValueError(f"Unknown provider: {provider}")


# =========================
# 3. GUDID ANALYSIS HELPERS
# =========================

def build_prompt_for_agent(agent_config, df: pd.DataFrame, previous_output: str | None, extra_instruction: str | None):
    base = agent_config.get("system_prompt", "")
    data_sample = df.head(30).to_dict(orient="records")
    prompt = base + "\n\n=== GUDID SAMPLE (JSON) ===\n" + json.dumps(data_sample, ensure_ascii=False)
    if previous_output:
        prompt += "\n\n=== PREVIOUS AGENT OUTPUT ===\n" + previous_output
    if extra_instruction:
        prompt += "\n\n=== USER EXTRA INSTRUCTION ===\n" + extra_instruction
    return prompt


def parse_agent_result(raw_text: str):
    try:
        start = raw_text.index("{")
        end = raw_text.rindex("}")
        js = json.loads(raw_text[start:end + 1])
        return js
    except Exception:
        return {"summary": raw_text, "issues": [], "metrics": {"risk_score": None}}


# =========================
# 4. AI NOTE-KEEPING TOOLS
# =========================

def ai_format_markdown(clients: LLMClients, provider, model, note_text, max_tokens, extra_prompt=""):
    base = """You are an expert technical editor.
Transform the following note into clean, well-structured Markdown.
Use headings, bullet points, numbered lists, and tables when useful.
Do NOT invent content."""
    if extra_prompt:
        base += "\n\nAdditional instruction:\n" + extra_prompt
    prompt = base + "\n\n=== NOTE BEGIN ===\n" + note_text + "\n=== NOTE END ==="
    return clients.call(provider, model, prompt, max_tokens)


def ai_extract_keywords(clients: LLMClients, provider, model, note_text, max_tokens):
    prompt = """Extract 15-25 concise, domain-relevant keywords from the note.
Return them as a comma-separated list only, no explanation."""
    out = clients.call(provider, model, prompt + "\n\n" + note_text, max_tokens)
    return [k.strip() for k in out.replace("\n", " ").split(",") if k.strip()]


def ai_summarize(clients: LLMClients, provider, model, note_text, max_tokens, extra_prompt=""):
    base = """Create a comprehensive Markdown summary of the note for an FDA RA / data analyst audience.
- Use sections, bullet points, and tables when useful.
- Highlight important keywords in <span class="coral-keyword">KEYWORD</span> with coral color.
Return valid Markdown/HTML only."""
    if extra_prompt:
        base += "\n\nAdditional instruction:\n" + extra_prompt
    prompt = base + "\n\n=== NOTE BEGIN ===\n" + note_text + "\n=== NOTE END ==="
    return clients.call(provider, model, prompt, max_tokens)


def ai_extract_entities(clients: LLMClients, provider, model, note_text, max_tokens):
    prompt = """Extract exactly 20 key entities from the note.
For each entity, provide:
- name
- type (e.g., Device, Risk, Regulation, Field, Stakeholder)
- context (1-2 sentences, Markdown)
Return as a Markdown table with columns: #, Name, Type, Context."""
    return clients.call(provider, model, prompt + "\n\n" + note_text, max_tokens)


def ai_chat_on_note(clients: LLMClients, provider, model, note_text, user_message, max_tokens):
    prompt = f"""You are an assistant answering questions about the following note.
Be specific and quote relevant phrases when useful.

=== NOTE BEGIN ===
{note_text}
=== NOTE END ===

User question: {user_message}
"""
    return clients.call(provider, model, prompt, max_tokens)


def ai_magic_insight_map(clients: LLMClients, provider, model, note_text, max_tokens):
    prompt = """Create a hierarchical 'Insight Map' of the note as Markdown:
- Top-level sections as H2 (##)
- Sub-concepts as nested bullet lists
- For any strong relationship, add inline links like [see also: Section Name].
Focus on regulatory, data quality, and risk-related structure."""
    return clients.call(provider, model, prompt + "\n\n" + note_text, max_tokens)


def ai_magic_action_plan(clients: LLMClients, provider, model, note_text, max_tokens):
    prompt = """Convert the note into a prioritized, actionable plan in Markdown for an FDA RA / data quality team:
- Use numbered sections for phases (e.g., 1. Immediate fixes, 2. Short-term, 3. Long-term).
- Under each, list tasks as checklist items: - [ ] Task description
- For each task, specify owner role and expected timeline.
Do NOT invent regulations, but infer reasonable task granularity."""
    return clients.call(provider, model, prompt + "\n\n" + note_text, max_tokens)


# =========================
# 5. LOAD AGENTS (YAML)
# =========================

@st.cache_resource
def load_agents():
    with open("agents.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("defaults", {}), data["agents"]


# =========================
# 6. APP STATE & SETUP
# =========================

st.set_page_config(
    page_title="GUDID 多代理系統",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_state():
    init_lang()
    init_theme_state()
    st.session_state.setdefault("dataset_df", None)
    st.session_state.setdefault("agent_results", {})
    st.session_state.setdefault("current_chain_input", "")
    st.session_state.setdefault("note_raw", "")
    st.session_state.setdefault("note_markdown", "")
    st.session_state.setdefault("note_chat_history", [])
    st.session_state.setdefault("gemini_api_key", None)
    st.session_state.setdefault("openai_api_key", None)
    st.session_state.setdefault("anthropic_api_key", None)
    st.session_state.setdefault("grok_api_key", None)


init_state()
defaults, base_agents = load_agents()

# Keep a mutable copy of agent configs in session state
if "agents_config" not in st.session_state:
    # Deep copy-like initialization
    st.session_state["agents_config"] = [dict(a) for a in base_agents]

agents_config = st.session_state["agents_config"]

inject_theme_css()

# Sidebar: language, theme, Jackslot, API keys
with st.sidebar:
    st.header(t("sidebar_settings"))

    # Language
    st.session_state["lang"] = st.radio(
        t("language"),
        options=["zh", "en"],
        format_func=lambda v: "繁體中文" if v == "zh" else "English",
        key="lang",
    )

    # Theme
    st.session_state["theme_mode"] = st.radio(
        t("theme"),
        options=["light", "dark"],
        format_func=lambda v: t("light") if v == "light" else t("dark"),
        horizontal=True,
        key="theme_mode",
    )

    # Jackslot
    st.subheader(t("flower_style"))
    col1, col2 = st.columns([2, 1])
    with col1:
        current_theme = get_current_theme()
        st.caption(f"{current_theme['name_zh']} / {current_theme['name_en']}")
    with col2:
        if st.button("Spin!"):
            choice = jackslot_spin()
            st.success(f"🎰 {choice['name_zh']} / {choice['name_en']}")

    st.divider()

    # API Keys (hide if env var set)
    st.subheader(t("api_keys"))
    if not os.getenv("GEMINI_API_KEY"):
        st.session_state["gemini_api_key"] = st.text_input("Gemini API Key", type="password")
    if not os.getenv("OPENAI_API_KEY"):
        st.session_state["openai_api_key"] = st.text_input("OpenAI API Key", type="password")
    if not os.getenv("ANTHROPIC_API_KEY"):
        st.session_state["anthropic_api_key"] = st.text_input("Anthropic API Key", type="password")
    if not os.getenv("GROK_API_KEY"):
        st.session_state["grok_api_key"] = st.text_input("Grok API Key", type="password")

clients = LLMClients(st.session_state)

st.title(t("title"))
tabs = st.tabs([t("analysis_tab"), t("dashboard_tab"), t("agents_tab"), t("notes_tab")])
analysis_tab, dashboard_tab, agents_tab, notes_tab = tabs


# =========================
# 7. ANALYSIS TAB
# =========================

with analysis_tab:
    st.subheader("1. Dataset")

    uploaded = st.file_uploader("Upload GUDID CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.session_state["dataset_df"] = df

    if st.session_state["dataset_df"] is None:
        st.info("Using built-in mock GUDID dataset.")
        try:
            st.session_state["dataset_df"] = pd.read_csv("sample_data/gudid_mock.csv")
        except FileNotFoundError:
            st.warning("sample_data/gudid_mock.csv not found. Please upload a CSV.")
            st.session_state["dataset_df"] = pd.DataFrame()

    df = st.session_state["dataset_df"]

    if not df.empty:
        edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True, key="data_editor")
        st.session_state["dataset_df"] = edited_df

        st.download_button(
            "Download CSV",
            edited_df.to_csv(index=False).encode("utf-8"),
            file_name="gudid_edited.csv",
            mime="text/csv",
        )
    else:
        st.info("No data loaded yet.")

    st.subheader("2. Run Agents")

    enabled_agents = [a for a in agents_config if a.get("enabled", True)]
    agent_ids = [a["id"] for a in enabled_agents]

    run_mode = st.radio("Execution Mode", ["Sequential", "Single"], horizontal=True)
    selected_agent_id = None
    if run_mode == "Single":
        if enabled_agents:
            selected_agent_id = st.selectbox("Choose Agent", agent_ids)
        else:
            st.warning("No agents enabled.")

    col1, col2, col3 = st.columns(3)
    with col1:
        global_model = st.selectbox(
            "Override Model (optional)",
            ["(agent default)", "gpt-4o-mini", "gpt-4.1-mini",
             "gemini-2.5-flash", "gemini-2.5-flash-lite",
             "claude-3-5-sonnet", "grok-4-fast-reasoning", "grok-3-mini"],
        )
    with col2:
        global_max_tokens = st.number_input(
            "Max tokens", min_value=512, max_value=120000, value=12000, step=512
        )
    with col3:
        user_prompt_suffix = st.text_area(
            "Optional global user instruction (appended to agent prompt)",
            height=80,
        )

    if st.button("Run Analysis"):
        if df.empty:
            st.error("Dataset is empty. Please upload or load data first.")
        elif not enabled_agents:
            st.error("No agents enabled.")
        else:
            with st.spinner("Running agents..."):
                results = {}
                chain_input = None

                if run_mode == "Sequential":
                    target_agents = enabled_agents
                else:
                    target_agents = [
                        next(a for a in enabled_agents if a["id"] == selected_agent_id)
                    ]

                progress = st.progress(0.0)
                total = len(target_agents)

                for idx, agent in enumerate(target_agents, start=1):
                    agent_id = agent["id"]
                    provider = agent.get("provider", defaults.get("provider", "gemini"))
                    model = agent.get("model", defaults.get("model"))
                    if global_model != "(agent default)":
                        model = global_model

                    max_tokens = int(agent.get("max_tokens", defaults.get("max_tokens", global_max_tokens)))
                    max_tokens = min(max_tokens, int(global_max_tokens))

                    prompt = build_prompt_for_agent(
                        agent_config=agent,
                        df=df,
                        previous_output=chain_input,
                        extra_instruction=user_prompt_suffix,
                    )
                    try:
                        raw = clients.call(provider, model, prompt, max_tokens=max_tokens)
                        parsed = parse_agent_result(raw)
                    except Exception as e:
                        parsed = {
                            "summary": f"Error while running agent {agent_id}: {e}",
                            "issues": [],
                            "metrics": {"risk_score": None},
                        }

                    results[agent_id] = parsed
                    chain_input = parsed.get("summary", raw if "raw" in locals() else "")

                    progress.progress(idx / total)

                st.session_state["agent_results"] = results
                st.session_state["current_chain_input"] = chain_input
            st.success("Analysis completed.")


# =========================
# 8. DASHBOARD TAB
# =========================

with dashboard_tab:
    st.subheader("WOW Dashboard & Data Quality Status")

    results = st.session_state.get("agent_results", {})
    if not results:
        st.info("Run analysis to see dashboard.")
    else:
        total_issues = 0
        per_severity = {"low": 0, "medium": 0, "high": 0}
        agent_scores = []

        for agent_id, res in results.items():
            issues = res.get("issues", [])
            total_issues += len(issues)
            for iss in issues:
                sev = str(iss.get("severity", "medium")).lower()
                if sev in per_severity:
                    per_severity[sev] += 1
            risk = res.get("metrics", {}).get("risk_score")
            if risk is not None:
                agent_scores.append({"agent": agent_id, "risk_score": risk})

        if agent_scores:
            avg_risk = sum(a["risk_score"] for a in agent_scores) / len(agent_scores)
        else:
            avg_risk = 0

        if avg_risk < 30:
            risk_label = "Low"
            risk_color = "green"
        elif avg_risk < 70:
            risk_label = "Medium"
            risk_color = "orange"
        else:
            risk_label = "High"
            risk_color = "red"

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                f'<div class="wow-card"><b>Total Issues</b><br><h2>{total_issues}</h2></div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f'<div class="wow-card"><b>Overall Risk</b><br>'
                f'<h2 style="color:{risk_color}">{risk_label} ({avg_risk:.1f})</h2></div>',
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f'<div class="wow-card"><b>Agents Run</b><br><h2>{len(results)}</h2></div>',
                unsafe_allow_html=True,
            )

        sev_df = pd.DataFrame([
            {"severity": "Low", "count": per_severity["low"]},
            {"severity": "Medium", "count": per_severity["medium"]},
            {"severity": "High", "count": per_severity["high"]},
        ])

        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.pie(sev_df, values="count", names="severity", title="Issue Severity Distribution", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            if agent_scores:
                score_df = pd.DataFrame(agent_scores)
                fig2 = px.bar(score_df, x="agent", y="risk_score", title="Per-Agent Risk Scores")
                fig2.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### Detailed Findings by Agent")
        for agent_id, res in results.items():
            with st.expander(agent_id, expanded=False):
                st.markdown(f"**Summary**:\n\n{res.get('summary', '')}")
                issues = res.get("issues", [])
                if issues:
                    for i, iss in enumerate(issues, 1):
                        st.markdown(f"- **#{i}** [{iss.get('severity','')}] {iss.get('title','')}")
                        if "details" in iss:
                            st.markdown(f"  - Details: {iss['details']}")
                else:
                    st.caption("No structured issues reported.")


# =========================
# 9. AGENTS & CHAINS TAB
# =========================

with agents_tab:
    st.subheader("Agent Configuration & Chaining")
    st.caption("Enable/disable agents, override prompts/models, and edit chain handoffs.")

    for agent in agents_config:
        with st.expander(f"{agent.get('name', agent['id'])} ({agent['id']})", expanded=False):
            agent["enabled"] = st.checkbox(
                "Enabled",
                value=agent.get("enabled", True),
                key=f"enabled_{agent['id']}",
            )

            agent["system_prompt"] = st.text_area(
                "System Prompt",
                value=agent.get("system_prompt", ""),
                key=f"prompt_{agent['id']}",
                height=150,
            )

            col1, col2 = st.columns(2)
            with col1:
                provider_default = agent.get("provider", defaults.get("provider", "gemini"))
                agent["provider"] = st.selectbox(
                    "Provider",
                    ["gemini", "openai", "anthropic", "grok"],
                    index=["gemini", "openai", "anthropic", "grok"].index(provider_default),
                    key=f"prov_{agent['id']}",
                )
            with col2:
                model_default = agent.get("model", defaults.get("model", "gemini-2.5-flash"))
                all_models = [
                    "gpt-4o-mini", "gpt-4.1-mini",
                    "gemini-2.5-flash", "gemini-2.5-flash-lite",
                    "claude-3-5-sonnet",
                    "grok-4-fast-reasoning", "grok-3-mini",
                ]
                if model_default not in all_models:
                    all_models.insert(0, model_default)
                agent["model"] = st.selectbox(
                    "Model",
                    all_models,
                    index=all_models.index(model_default),
                    key=f"model_{agent['id']}",
                )

            max_tok_key = f"max_tok_{agent['id']}"
            agent["max_tokens"] = st.number_input(
                "Max tokens for this agent",
                min_value=512,
                max_value=120000,
                value=int(agent.get("max_tokens", defaults.get("max_tokens", 12000))),
                step=512,
                key=max_tok_key,
            )

    st.divider()
    st.subheader("Chain Handoff Editor")

    view_mode = st.radio("View as", ["Text", "Markdown"], horizontal=True, key="chain_view_mode")
    if view_mode == "Text":
        st.session_state["current_chain_input"] = st.text_area(
            "Handoff input to next agent",
            value=st.session_state.get("current_chain_input", ""),
            height=200,
        )
    else:
        st.markdown(st.session_state.get("current_chain_input", "") or "_No handoff text yet._")


# =========================
# 10. AI NOTE KEEPING TAB
# =========================

with notes_tab:
    st.subheader("AI Note Keeping")

    st.markdown("#### 1. Paste or Edit Note")
    st.session_state["note_raw"] = st.text_area(
        "Raw note (plain text)",
        value=st.session_state.get("note_raw", ""),
        height=180,
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Transform to Markdown (basic)"):
            text = st.session_state["note_raw"]
            # Simple heuristic: add bullets for paragraphs
            if text.strip():
                md = "\n\n".join(
                    [f"- {p.strip()}" if p.strip() else "" for p in text.split("\n\n")]
                )
            else:
                md = ""
            st.session_state["note_markdown"] = md
    with col2:
        if st.button("Clear Note"):
            st.session_state["note_raw"] = ""
            st.session_state["note_markdown"] = ""
            st.session_state["note_chat_history"] = []

    view_mode = st.radio(
        "Editing mode",
        ["Text", "Markdown Preview"],
        horizontal=True,
        key="note_view_mode",
    )
    if view_mode == "Text":
        st.session_state["note_markdown"] = st.text_area(
            "Markdown note (editable)",
            value=st.session_state.get("note_markdown", ""),
            height=220,
        )
    else:
        st.markdown(st.session_state.get("note_markdown", "") or "_No markdown yet._")

    st.markdown("#### 2. AI Tools for This Note")

    colm1, colm2, colm3 = st.columns(3)
    with colm1:
        note_model = st.selectbox(
            "Model",
            ["gpt-4o-mini", "gpt-4.1-mini",
             "gemini-2.5-flash", "gemini-2.5-flash-lite",
             "claude-3-5-sonnet", "grok-4-fast-reasoning", "grok-3-mini"],
            key="note_model",
        )
    with colm2:
        note_provider = st.selectbox(
            "Provider",
            ["openai", "gemini", "anthropic", "grok"],
            key="note_provider",
        )
    with colm3:
        note_max_tokens = st.number_input(
            "Max tokens", min_value=512, max_value=120000, value=4000, step=512, key="note_max_tokens"
        )

    st.markdown("##### AI Formatting")
    fmt_extra_prompt = st.text_input("Optional extra instructions for formatting", key="fmt_extra_prompt")
    if st.button("Run AI Formatting (Markdown)"):
        if not st.session_state["note_raw"].strip():
            st.error("Please provide note text first.")
        else:
            with st.spinner("Formatting note..."):
                try:
                    output = ai_format_markdown(
                        clients,
                        note_provider,
                        note_model,
                        st.session_state.get("note_raw", ""),
                        int(note_max_tokens),
                        extra_prompt=fmt_extra_prompt,
                    )
                    st.session_state["note_markdown"] = output
                    st.success("Formatting completed.")
                except Exception as e:
                    st.error(f"AI Formatting failed: {e}")

    st.markdown("##### AI Keywords")
    if st.button("Extract AI Keywords"):
        if not st.session_state["note_raw"].strip():
            st.error("Please provide note text first.")
        else:
            with st.spinner("Extracting keywords..."):
                try:
                    kws = ai_extract_keywords(
                        clients,
                        note_provider,
                        note_model,
                        st.session_state.get("note_raw", ""),
                        int(note_max_tokens),
                    )
                    st.markdown("**Keywords**:")
                    st.markdown(
                        " ".join([f'<span class="coral-keyword">{k}</span>' for k in kws]),
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.error(f"Keyword extraction failed: {e}")

    st.markdown("##### AI Summary")
    sum_extra_prompt = st.text_input("Optional extra instructions for summary", key="sum_extra_prompt")
    if st.button("Generate AI Summary (Markdown with coral keywords)"):
        if not st.session_state["note_raw"].strip():
            st.error("Please provide note text first.")
        else:
            with st.spinner("Summarizing..."):
                try:
                    summary_md = ai_summarize(
                        clients,
                        note_provider,
                        note_model,
                        st.session_state.get("note_raw", ""),
                        int(note_max_tokens),
                        extra_prompt=sum_extra_prompt,
                    )
                    st.markdown(summary_md, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Summary generation failed: {e}")

    st.markdown("##### AI Entities (20 with context)")
    if st.button("Generate Entities"):
        if not st.session_state["note_raw"].strip():
            st.error("Please provide note text first.")
        else:
            with st.spinner("Extracting entities..."):
                try:
                    entities_md = ai_extract_entities(
                        clients,
                        note_provider,
                        note_model,
                        st.session_state.get("note_raw", ""),
                        int(note_max_tokens),
                    )
                    st.markdown(entities_md, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Entity extraction failed: {e}")

    st.markdown("##### AI Chat on This Note")
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state["note_chat_history"]:
            st.markdown(f"**{msg['role'].capitalize()}:** {msg['content']}")

    user_q = st.text_input("Ask a question about this note", key="note_chat_input")
    if st.button("Send Question"):
        if not st.session_state["note_raw"].strip():
            st.error("Please provide note text first.")
        elif user_q.strip():
            st.session_state["note_chat_history"].append({"role": "user", "content": user_q})
            with st.spinner("Thinking..."):
                try:
                    answer = ai_chat_on_note(
                        clients,
                        note_provider,
                        note_model,
                        st.session_state.get("note_raw", ""),
                        user_q,
                        int(note_max_tokens),
                    )
                    st.session_state["note_chat_history"].append({"role": "assistant", "content": answer})
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Chat failed: {e}")

    st.markdown("##### AI Magic")

    colmA, colmB = st.columns(2)
    with colmA:
        if st.button("Magic Insight Map"):
            if not st.session_state["note_raw"].strip():
                st.error("Please provide note text first.")
            else:
                with st.spinner("Building insight map..."):
                    try:
                        insight_md = ai_magic_insight_map(
                            clients,
                            note_provider,
                            note_model,
                            st.session_state.get("note_raw", ""),
                            int(note_max_tokens),
                        )
                        st.markdown("###### Insight Map")
                        st.markdown(insight_md, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Insight map failed: {e}")

    with colmB:
        if st.button("Magic Action Plan"):
            if not st.session_state["note_raw"].strip():
                st.error("Please provide note text first.")
            else:
                with st.spinner("Deriving action plan..."):
                    try:
                        plan_md = ai_magic_action_plan(
                            clients,
                            note_provider,
                            note_model,
                            st.session_state.get("note_raw", ""),
                            int(note_max_tokens),
                        )
                        st.markdown("###### Action Plan")
                        st.markdown(plan_md, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Action plan failed: {e}")
