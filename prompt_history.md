# Prompt History — Multi-Agent Communication Protocol Study

> STAT GR5293, Columbia University, Spring 2026  
> Yi Zhang (yz5104) × Claude Sonnet 4.6  
> Conversation compacted after context limit; earlier session reconstructed from summary.

---

## Session 1 — 实验结果检查

**User:** 好耶，代码跑完了，你看看结果。

**Assistant:**
读取了 `results/results_summary.csv` 和 `results/results_raw.csv`，整理出以下结论：

- **360 runs** 全部完成（4 协议 × 3 域 × 10 样本 × 3 重复）。
- `results_messages.jsonl` 包含 1080 条消息，SHARED_MEMORY 的 Integrator `prompt_tokens`（469）远高于其他协议（188–254），确认黑板状态累积行为正常。
- **JSON parse errors：0**（`response_format=json_object` + 1-retry 效果良好）。
- **Truncation（finish_reason=length）：11/360**，集中在 MATH 域的 Markdown（6 次）和 SHARED_MEMORY（3 次），JSON 0 次。
- 关键 completion rate：Markdown/MATH 0.867（最高），Shared Memory/READING 0.556（最高），NL/NEWS 最差 0.275。

三条假设初步状态：
| 假设 | 结论 |
|---|---|
| H1：Shared Memory overhead 最高 | 支持（token 最多），但 JSON 并非全域最省 |
| H2：结构化协议效率更高 | 部分支持，方向与 proposal 相反（NEWS > READING > MATH）|
| H3：协议 × 域交互显著 | token 交互 p<.001 显著；completion 交互 p=.203 不显著 |

---

## Session 2 — 回填 Report Outline

**User:** 回填 report outline.

**Assistant:**
将 `report/REPORT_OUTLINE.md` 中所有 `TODO` 占位符替换为实验真实数字，涵盖：

- **Abstract**：加入 H1/H2/H3 verdict 句（"Shared Memory incurs 21–46% more prompt tokens… JSON reduces tokens on MATH but not NEWS/READING…"）。
- **Two-way ANOVA 表**（tokens）：Protocol η²=.262, Domain η²=.296, Interaction η²=.094；completion 交互 F=1.43, p=.203。
- **Per-domain one-way ANOVA 表**：MATH η²=.017, READING η²=.102, NEWS η²=.197。
- **Tukey HSD 表**（pooled tokens 6 对全部显著）及 per-domain 显著对。
- **Bootstrap 95% CI** 每格（协议 × 域）。
- **Pareto frontier**（per domain）。
- **Error analysis**：0/90 JSON parse errors；11/360 truncations（全在 MATH，JSON 无截断）。
- **Blackboard 验证**：SHARED_MEMORY prompt_tokens Planner 235 → Executor 322 → Integrator 469（单调递增）。

---

## Session 3 — Streamlit Smoke Test

**User:** 6 张 figures 目前看来没问题，接下来做 streamlit smoke test。

**Assistant:**
验证步骤：

1. **Import 级别**：`app.py` 成功解析，`pipeline.py` 所有依赖可导入。
2. **Protocol lookup**：`results/results_summary.csv` 加载正常，MATH→Markdown、READING→Shared Memory、NEWS→Shared Memory。
3. **HTTP 级别**：`streamlit run app.py` 启动，HTTP 200，无 traceback。

启动命令：

```bash
streamlit run app.py
```

然后在浏览器打开 http://localhost:8501。

---

**User:** 怎么测试这个 streamlit 啊，直接在 vscode 里运行这两个 py 吗？

**Assistant:**
不能直接 `python app.py`，必须用 Streamlit CLI：

```bash
# 在项目根目录
streamlit run app.py
```

API key 填在左侧 sidebar 的输入框（或提前 `export OPENAI_API_KEY=sk-...`）。

生成了三个测试问题（覆盖三条路由）：

| 域 | 测试输入 |
|---|---|
| MATH | `Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes muffins with 4. She sells the rest at $2 per egg. How much does she make per day?` |
| READING | `Super Bowl 50 was an American football game to determine the champion of the NFL for the 2015 season. The American Football Conference (AFC) champion, the Denver Broncos, defeated the National Football Conference (NFC) champion, the Carolina Panthers, 24–10... Question: Which team won Super Bowl 50?` |
| NEWS | `OpenAI slashed the price of its GPT-4o API by 50%, from $5 to $2.50 per million input tokens and from $15 to $10 per million output tokens, effective immediately. The company said the reduction reflects improved efficiency from model distillation and infrastructure optimizations. Summarize the key facts.` |

---

**User:** 已经跑起来了，我要填我的 api key 吗，然后帮我生成几个测试问题。  
（附截图：Streamlit 运行在 localhost:8501）

**Assistant:**
是的，需要在 sidebar 的 "OpenAI API Key" 框里填入真实 key（或者关掉 Streamlit，在终端先 `export OPENAI_API_KEY=sk-...` 再启动，这样 sidebar 会自动预填）。  
测试问题见上表。

---

**User:** 这是三个测试结果，如何。（粘贴了三路 agent message log）

**Assistant:**
三路全部通过：

- **MATH（Janet eggs）**：分类 MATH，推荐 Markdown，答案 $18/day ✅，prompt_tokens Planner<Executor<Integrator（正常顺序）。
- **READING（Super Bowl）**：分类 READING，推荐 Shared Memory，答案 "Denver Broncos" ✅，prompt_tokens 单调递增（黑板状态累积）。
- **NEWS（OpenAI 降价）**：分类 NEWS，推荐 Shared Memory，关键事实全覆盖 ✅，Integrator prompt_tokens 最大（状态最厚）。

Streamlit smoke test **通过**。

---

## Session 4 — README 教程 & GitHub 推送

**User:** 那可以把使用教程写到 readme.md 里，然后帮我上传到 github 吧。我本地应该是 install git 过了，然后邮箱什么的都连接过。这是我的 github：https://github.com/johnny031010-ctrl?tab=repositories。记得不要把 .env 传上去。

**Assistant:**
**README 改动**：将 Quickstart 从原来的 ~20 行扩展为完整 4 步教程：
1. `git clone` + `pip install`
2. `.env` 设置（`cp .env.example .env` + 填 key）
3. `jupyter nbconvert --execute` + 输出说明
4. `streamlit run app.py` + 5 步 Demo 流程 + 3 条测试问题

**Git 操作**：
```bash
git init -b main
git add <27 files>     # 扫描排除 .env，仅含 .env.example 占位符
git commit -m "Initial commit: Multi-Agent Communication Protocol Study"
git remote add origin https://github.com/johnny031010-ctrl/Multi-Agent-Communication-Protocol-Study
```

遇到 `gh` CLI 未安装 + SSH 无公钥问题，改为 HTTPS 推送，但需要用户手动输入凭证。

---

**User:** https://github.com/johnny031010-ctrl/Multi-Agent-Communication-Protocol-Study 我已经 push 上去了。

**Assistant:**
验证结果：

- `git ls-remote origin` → HEAD `19ecddd` 与本地一致 ✅
- GitHub API → repo public，default branch `main`，32 files ✅
- `.env` **未泄露**（树中仅 `.env.example` 占位符 + `.gitignore`）✅

项目全线完工：pipeline 方法论修好 → 360 run 跑完 → 6 张 figures → report outline 回填真实数字 → Streamlit demo 端到端测通 → README 写好教程 → 推到公开 GitHub。

> ⚠️ 鉴于 API key 之前有过短暂暴露，如果你还没轮换 key 记得去 OpenAI 后台 revoke 掉那条旧的。

---

## Session 5 — 用户名更新

**User:** https://github.com/KuriPantsu/Multi-Agent-Communication-Protocol-Study  
我把用户名改了，你帮我把这个项目里包含我以前那个用户名的地方给改一下吧。

**Assistant:**
扫描全项目，发现两处引用旧用户名 `johnny031010-ctrl`：

1. `README.md:46` — `git clone` URL
2. 本地 `git remote origin` URL

修改内容：

```diff
- git clone https://github.com/johnny031010-ctrl/<repo-name>.git
- cd <repo-name>
+ git clone https://github.com/KuriPantsu/Multi-Agent-Communication-Protocol-Study.git
+ cd Multi-Agent-Communication-Protocol-Study
```

```bash
git remote set-url origin https://github.com/KuriPantsu/Multi-Agent-Communication-Protocol-Study.git
```

Commit `7cd6016` 已推送到新用户名下的仓库 ✅。

---

## Session 6 — Prompt History 整理

**User:** 能不能帮我把我跟你对话的提示词和回答整理进 prompt_history.md？

**Assistant:** 生成本文件（`prompt_history.md`）。
