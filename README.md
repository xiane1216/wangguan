# 记忆网关（memory-gateway）— 傻瓜式部署教程

## 这是个啥？

一个部署在 Zeabur 上的小网关，挡在你的 App 和 DeepSeek 中间：

```
你的手机 App ──> Zeabur 记忆网关 ──> DeepSeek API
                    │
                    ├── 自动读 Supabase 聊天记录（chat_messages 表）
                    ├── 自己总结「近期记忆 / 长期记忆」（用 deepseek-flash）
                    └── 首次启动把 Ombre Brain 的浮现记忆原样搬过来
```

以前你开窗，要让 AI 自己调工具去读 OB + Supabase，一次读 70K+ token，还要等它慢慢读；
现在开窗时**网关自动把三段记忆直接注入**，AI 不用调工具、不用等、也不会读出一堆时间戳和消息 id 浪费 token：

1. **长期记忆** —— Ombre Brain 搬运来的记忆原文 + 网关自己从聊天记录逐天总结、逐段合并的概括
2. **近期记忆** —— 最近 3 天每天的概括
3. **上个窗口原始聊天记录** —— 上个窗口最后 20 条原文，一字不改

### 为什么它不会像日记总结那样瞎编？

- **搬运和原始记录 = 原样复制**，根本不经过 AI，不可能产生幻觉；
- **每日总结 / 长期记忆合并** 用 deepseek-flash + 防幻觉铁律 prompt（只写原文明确出现的内容，不确定的不写，宁漏勿错），temperature 压到 0.2；
- **全程不读 memory_summaries 表**（那个日记总结表已弃用，网关从头到尾不碰它）。

---

## 准备清单

- [ ] Zeabur 账号（你已有 ✓）
- [ ] GitHub 账号（你已有 ✓）
- [ ] 你的 Supabase 项目地址（`https://xxxx.supabase.co`，是**原始地址**，不是中转域名）
- [ ] 你的 Supabase Key（App 外置记忆库里填的那个 Key，或 Supabase 官网 → 项目 → Settings → API → anon key）
- [ ] 我在聊天里发你的几个值（GATEWAY_KEY、ADMIN_SECRET 等，直接复制）

---

## 第一步：Supabase 建一张新表

网关需要一张自己的表存记忆。打开 [supabase.com](https://supabase.com) → 你的项目 → 左侧 **SQL Editor** → 把下面整段粘进去 → 点 **Run**：

```sql
create table if not exists public.gateway_memory (
    id bigserial primary key,
    kind text not null,          -- longterm / daily / state
    scope text not null default '',
    content text not null default '',
    meta text not null default '',
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint gateway_memory_kind_scope_key unique (kind, scope)
);

alter table public.gateway_memory enable row level security;

drop policy if exists "allow all for anon" on public.gateway_memory;
create policy "allow all for anon"
    on public.gateway_memory
    for all
    to anon, authenticated
    using (true)
    with check (true);

create index if not exists gateway_memory_kind_idx on public.gateway_memory (kind);
```

看到绿色 Success 就行。**这一步不会动你已有的 chat_messages 表，只是新建一张表。**

---

## 第二步：把代码传到 GitHub

1. 解压 `memory-gateway.zip`，里面有 10 个文件（都是代码和配置，不用看懂）
2. 打开 [github.com](https://github.com) → 右上角 **+** → **New repository**
3. 名字填 `memory-gateway`，其他不动，点 **Create repository**
4. 在创建完的页面点 **uploading an existing file** 链接
5. 把解压出来的**全部文件**拖进去（保持文件在根目录，不要套文件夹）
6. 点 **Commit changes**

---

## 第三步：Zeabur 部署 + 配环境变量

1. 打开 [zeabur.com](https://zeabur.com) → 新建一个项目（比如叫 `gateway`），区域选**香港**或**东京**
2. **Add Service** → **Git** → 选刚才的 `memory-gateway` 仓库
3. Zeabur 会识别 Dockerfile 自动构建，等状态变绿
4. 进服务的 **Variables（变量）** 标签，逐条添加（值用我在聊天里发你的，别抄这里的占位符）：

| 变量名 | 填什么 |
|---|---|
| `DEEPSEEK_API_KEY` | 你的 DeepSeek API Key（sk- 开头那条） |
| `SUPABASE_URL` | `https://xxxx.supabase.co`（你的**原始** Supabase 地址） |
| `SUPABASE_KEY` | 你的 Supabase Key |
| `OB_MCP_URL` | 你的 Ombre Brain MCP 地址 |
| `GATEWAY_KEY` | 我给你的网关钥匙（`gw-` 开头） |
| `ADMIN_SECRET` | 我给你的管理钥匙（`adm-` 开头） |

可选项（不填就用默认值，一般不用管）：

| 变量名 | 默认 | 含义 |
|---|---|---|
| `RECENT_DAYS` | 3 | 近期记忆保留几天 |
| `LAST_WINDOW_MESSAGES` | 20 | 上个窗口原始记录条数 |
| `REFRESH_HOURS` | 2 | 自动总结周期（小时） |
| `MAX_BACKFILL_DAYS` | 30 | 首次部署往回总结多少天历史 |
| `ASSISTANT_ID` | 空 | 只处理某个助手的记录（一般留空） |
| `USER_LABEL` / `AI_LABEL` | 用户 / AI | 总结里双方的称呼 |

5. **Networking** → **Generate Domain** 生成域名，记下来，下面叫它「网关域名」

> 注意：网关部署在海外，`SUPABASE_URL` 直接填原始的 `xxx.supabase.co` 就行，**不要**填之前那个中转域名（中转是给手机在大陆直连用的）。

---

## 第四步：验证

手机浏览器打开：

```
https://网关域名/
```

看到一页 JSON 就说明起来了。重点看：

- `"deepseek_configured": true`
- `"supabase_configured": true`
- 等 1~2 分钟后台跑完首次任务后再刷新，`"longterm_chars"` 应该变成几万（OB 的记忆搬过来了），`"recent_daily"` 里出现最近几天的日期

**首次部署后台会自动做三件事**（不用你操作，等几分钟）：
1. 把 OB 浮现记忆**原样**搬进长期记忆
2. 往回总结最近 30 天聊天，生成每日概括
3. 把 3 天前的概括合并进长期记忆

想手动触发/查看，用浏览器打开（把钥匙替换进去）：

| 地址 | 作用 |
|---|---|
| `/admin/memory?key=管理钥匙` | 看长期记忆全文 + 每日概括 |
| `/admin/preview?key=管理钥匙` | 预览一次开窗会注入什么 |
| `/admin/import_ob?key=管理钥匙` | 重新搬运一次 OB 记忆（长期记忆已有内容时需加 `&mode=replace`，会覆盖网关自己总结的部分） |
| `/admin/refresh?key=管理钥匙&days=3` | 强制重新总结最近 3 天 |

---

## 第五步：改 App 里的供应商（只改地址和 Key，其他都不动）

设置 → 供应商 → 你的 **DeepSeek 供应商**：

| 项目 | 原来 | 改成 |
|---|---|---|
| API 地址 (Base URL) | `https://api.deepseek.com` | `https://网关域名` |
| API Key | 你的 DeepSeek Key | 我给你的 `gw-` 开头的网关钥匙 |
| 模型、路径 | —— | **完全不动** |

改完开个新窗口聊一句，去 Zeabur 的 **Logs** 里应该能看到一行 `窗口 xxx 注入记忆 xxxx 字`——成了。

> 想切回直连？把地址改回 `https://api.deepseek.com`、Key 换回 DeepSeek 的就行，随时切，没有任何锁定。

---

## 它是怎么工作的（了解即可）

- **透传**：你的请求原样转发给 DeepSeek，流式、思考链、工具调用一个字节不改；
- **开窗识别**：一条没有历史回复的新对话 = 新窗口，网关在 system 后面追加记忆块；
- **冻结**：同一个窗口里记忆块永远不变（滑动续期），所以 DeepSeek 的**前缀缓存照样命中**，token 不会因为注入而翻倍；
- **总结节奏**：今天/昨天的概括随消息增加自动更新（消息数没变就不重复花钱），3 天前的冻结，再老的逐段并进长期记忆；
- **防幻觉**：OB 搬运和原始 20 条是复制粘贴，零幻觉；总结 prompt 里有铁律 + 低温，只许写真实发生过的事。

---

## 常见问题

**Q: 开窗没记忆？**
先看 `https://网关域名/` 的 `longterm_chars` 是不是 0 或 null。是 → 检查环境变量和第一步建表；不是 → 等 `/admin/preview` 里三段都有内容再开窗。

**Q: 总结有错/有幻觉，想重新生成？**
浏览器打开 `/admin/refresh?key=管理钥匙&days=5` 重新总结最近 5 天。长期记忆想恢复成 OB 原版：`/admin/import_ob?key=管理钥匙&mode=replace`（会丢掉网关后来总结合并的内容，想清楚再用）。

**Q: OB 还用吗？**
建议保留。它的写入工具（hold/grow 那些）照常能用，但网关不会自动同步 OB 的新记忆——以后长期记忆归网关自己总结管理。另外提醒一句：**你的 OB 的 MCP 地址目前没有任何鉴权**，知道地址的人都能读，别把地址发给别人。

**Q: 别人拿到网关域名会怎样？**
没有 `gw-` 钥匙打不进来，放心。

**Q: 免费版 Zeabur 会休眠吗？**
会，冷启动慢几秒正常。记忆都存在 Supabase 里，重启不丢；重启后第一次请求会重新装配记忆块，稍慢一点。

**Q: 模型名要改吗？**
App 里供应商的模型名你原来填什么就填什么。总结模型默认 `deepseek-flash`（就是 DeepSeek-V4.1-Flash 的官方 API 名，注意不是 `deepseek-v4.1-flash`，带版本号反而会报错）。

**Q: 环境变量改了怎么生效？**
Zeabur 保存后自动重新部署，等状态变 Running 即可。
