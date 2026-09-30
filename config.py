"""记忆网关配置：所有参数都从环境变量读取（Zeabur 的 Variables 里配置）。"""
import os


def _int(name: str, default: int) -> int:
    try:
        return int((os.environ.get(name, "") or "").strip() or default)
    except Exception:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float((os.environ.get(name, "") or "").strip() or default)
    except Exception:
        return default


# ---- DeepSeek（对话透传 + 记忆总结共用一个 Key） ----
# 官方 API 模型名：deepseek-flash（就是 DeepSeek-V4.1-Flash，注意不是 deepseek-v4.1-flash）
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip().rstrip("/")
SUMMARY_MODEL = os.environ.get("SUMMARY_MODEL", "deepseek-flash").strip()
SUMMARY_TEMPERATURE = _float("SUMMARY_TEMPERATURE", 0.2)
SUMMARY_MAX_TOKENS = _int("SUMMARY_MAX_TOKENS", 24000)  # flash推理烧16K+是常态（实测满一天的记录能把16K全烧光），不给够正文出不来

# ---- Supabase（读聊天记录 chat_messages + 存网关自己的记忆表 gateway_memory） ----
# 注意：网关部署在海外，这里填 Supabase 的"原始"项目地址（xxx.supabase.co），不要填中转域名
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip().rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "").strip()
CHAT_TABLE = os.environ.get("CHAT_TABLE", "chat_messages")
MEMORY_TABLE = os.environ.get("MEMORY_TABLE", "gateway_memory")
ASSISTANT_ID = os.environ.get("ASSISTANT_ID", "").strip()  # 可选：只处理某个助手的记录，留空=全部

# ---- Ombre Brain（只在长期记忆为空时，把浮现记忆"原样"搬进来，之后网关自理） ----
OB_MCP_URL = os.environ.get("OB_MCP_URL", "").strip()

# ---- 鉴权 ----
GATEWAY_KEY = os.environ.get("GATEWAY_KEY", "").strip()    # App 供应商里填的 API Key
ADMIN_SECRET = os.environ.get("ADMIN_SECRET", "").strip()  # 管理接口的钥匙

# ---- 行为参数 ----
RECENT_DAYS = _int("RECENT_DAYS", 3)                     # "近期记忆"保留最近几天
REFRESH_HOURS = _float("REFRESH_HOURS", 2.0)             # 自动总结刷新周期（小时）
LAST_WINDOW_MESSAGES = _int("LAST_WINDOW_MESSAGES", 20)  # "上个窗口原始聊天记录"条数
MAX_BACKFILL_DAYS = _int("MAX_BACKFILL_DAYS", 30)        # 首次部署最多往回总结多少天历史
LONGTERM_MAX_CHARS = _int("LONGTERM_MAX_CHARS", 2000)    # 长期记忆全文上限（字）
TIMEZONE = os.environ.get("TIMEZONE", "Asia/Shanghai")
USER_LABEL = os.environ.get("USER_LABEL", "宝宝")  # 记忆里对用户的称呼
AI_LABEL = os.environ.get("AI_LABEL", "我")          # 记录里 AI 的自称（第一人称）
UPSTREAM_READ_TIMEOUT = _int("UPSTREAM_READ_TIMEOUT", 300)  # 上游长回复读超时（秒）
WINDOW_TTL_HOURS = _float("WINDOW_TTL_HOURS", 1.0)       # 空闲窗口多久后允许重建（活跃窗口永不重建）
AUTO_REFRESH = os.environ.get("AUTO_REFRESH", "0") == "1"  # 是否启用后台自动刷新（默认关闭，用 UI 手动触发）

# ---- bugfix 新增参数 ----
ROUGH_MAX_TOKENS = _int("ROUGH_MAX_TOKENS", 8000)          # 粗略归档 max_tokens（原 1200 会被推理烧光导致空正文）
NEW_WINDOW_REDETECT_SECONDS = _float("NEW_WINDOW_REDETECT_SECONDS", 300)  # 缓存命中但 0 条 assistant 回复且超过此秒数 → 视为同开场白新窗，强制重建
CUR_WINDOW_MAX_AGE_HOURS = _float("CUR_WINDOW_MAX_AGE_HOURS", 6.0)        # 定位"当前窗口"时只看最近 N 小时的 user 消息（防陈年问候语误匹配）

# ---- 窗口内滚动压缩（治"越聊越卡"） ----
ROLLING_TRIGGER = _int("ROLLING_TRIGGER", 40)          # user/assistant 消息超过多少条开始压缩（0=关闭滚动压缩）
ROLLING_KEEP = _int("ROLLING_KEEP", 20)                # 压缩后保留最近多少条原文
ROLLUP_MAX_TOKENS = _int("ROLLUP_MAX_TOKENS", 3000)    # 单次滚动摘要输出上限

# ---- 记忆分层字数上限 ----
DAILY_MAX_CHARS = _int("DAILY_MAX_CHARS", 1000)        # 每日概括字数上限


def supabase_ready() -> bool:
    return bool(SUPABASE_URL and SUPABASE_KEY)


def deepseek_ready() -> bool:
    return bool(DEEPSEEK_API_KEY)
