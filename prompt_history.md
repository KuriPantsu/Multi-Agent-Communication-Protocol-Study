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

---

## Session 7 — Final Presentation 幻灯片生成与清理

**User:** 仔细阅读并理解当前仓库，然后按照 Project_Rubric_STATGR5293_2026.pdf 里对 final presentation 的要求，用当前的项目的成果写一份 latex 的文件来生成 pdf 格式的幻灯片。幻灯片风格保持简洁实用。

**Assistant:**
先读取并理解了当前仓库结构、rubric、README、report outline、pipeline/demo 代码、实验配置、结果表和六张图：

- `Project_Rubric_STATGR5293_2026.pdf`：确认 final presentation 评分由三块组成：Problem Statement 10%、Major Contributions 10%、Evaluation 10%。
- `README.md` / `report/REPORT_OUTLINE.md`：确认项目主题、实验设计、主要结果、统计结论和 demo 叙事。
- `pipeline.py`：确认固定三代理链路 Planning → Execution → Integration，协议变量为 `NL`、`MARKDOWN`、`JSON`、`SHARED_MEMORY`。
- `results/results_summary.csv` / `results/experiment_config.json`：确认 360-run grid、模型 `gpt-4o-mini`、temperature 0.3、每格 N=30。
- `figures/fig1`–`fig6`：确认可用于幻灯片中的 token、completion、interaction、Pareto、input/output token、latency 图。

生成了 Beamer 幻灯片源码：

```text
presentation/final_presentation.tex
```

并用 `pdflatex` 编译生成：

```text
presentation/final_presentation.pdf
```

幻灯片结构：

1. Title page
2. Presentation Roadmap（显式对齐 rubric 三块）
3. Problem Statement
4. Research Questions and Hypotheses
5. System Model
6. Experimental Design
7. Contribution 1: Controlled Protocol Benchmark
8. Result: Protocol Strongly Affects Token Cost
9. Why the Cost Changes
10. Result: Quality Effects Are Domain-Dependent
11. Interaction Pattern
12. Efficiency--Effectiveness Trade-off
13. Latency and Operational Cost
14. Project Demo
15. Limitations and Future Work
16. Takeaways
17. Backup: Cell-Level Summary

关键内容：

- 强调项目研究的是 protocol layer，而不是一个新的 task-specific agent application。
- 展示 4 × 3 × 10 × 3 = 360 pipeline runs 的实验设计。
- 汇报 token ANOVA：Protocol F=87.40, p<.001, η²=.262；Domain F=148.19, p<.001, η²=.296；Interaction F=15.66, p<.001, η²=.094。
- 汇报 completion per-domain ANOVA：MATH η²=.017，READING η²=.102，NEWS η²=.197。
- 给出 practitioner recommendations：MATH 质量优先选 Markdown、成本优先选 JSON；READING/NEWS 质量优先选 Shared Memory，预算优先选 NL。
- 加入 Streamlit demo 说明：自动分类 domain、从 `results_summary.csv` 选择最佳协议、运行三代理 pipeline、展示 tokens/latency/cost/message log。

编译验证：

```bash
cd presentation
pdflatex -interaction=nonstopmode final_presentation.tex
```

最终 PDF 正常生成，共 17 页；第二次调整后无 LaTeX warning。

---

**User:** 清理掉 presentation 下 tex 和 pdf 以外后缀的文件，或者设置成不上传到 github

**Assistant:**
删除了 `presentation/` 下的 LaTeX 编译临时文件：

```text
presentation/final_presentation.aux
presentation/final_presentation.snm
presentation/final_presentation.log
presentation/final_presentation.nav
presentation/final_presentation.toc
presentation/final_presentation.out
```

清理后 `presentation/` 只保留：

```text
presentation/final_presentation.tex
presentation/final_presentation.pdf
```

同时更新 `.gitignore`，加入常见 LaTeX build artifacts，避免后续误上传：

```gitignore
*.aux
*.bbl
*.bcf
*.blg
*.fdb_latexmk
*.fls
*.log
*.nav
*.out
*.run.xml
*.snm
*.synctex.gz
*.toc
```

`git status --short` 显示当前新增 `presentation/`，并修改了 `.gitignore`。

---

**User:** 更新这次的对话到 prompt_history.md

**Assistant:** 将本次 final presentation 生成、PDF 编译验证、LaTeX 临时文件清理和 `.gitignore` 更新记录追加到 `prompt_history.md`。

---

## Session 8 — Final Presentation v0.2 视觉风格重做

**User:**  
给出新的 Design Philosophy，要求按照 Academic-minimal 风格改进原来的 v0.1 幻灯片：

- Style: Nature journal meets Figma's design system.
- NO gradients, glowing effects, glassmorphism, or techy dark backgrounds.
- Clean white/off-white `#FAFAFA` backgrounds.
- One accent color only: muted indigo `#4F46E5`.
- Generous whitespace.
- Simple matplotlib-style charts embedded as images.
- Font: Helvetica Neue or fallback to Calibri. Headers 32pt bold, body 18pt regular.
- Icons: flat line icons only, minimal usage.

**Assistant:**
重写了 `presentation/final_presentation.tex` 的 Beamer 样式和页面结构，把原来的默认 Beamer/Madrid 风格改成更接近 academic-minimal 的版本：

- 移除 `Madrid` 主题和默认 block-heavy 样式。
- 自定义白底 `#FAFAFA`、正文墨色、灰色辅助文本、单一 accent `#4F46E5`。
- 自定义 frame title、footer、item bullet、细线 divider。
- 页面改为更疏朗的排版：少量关键文字 + 大数字 + 左图右结论。
- 保留现有六张 matplotlib-style 实验图。
- 将部分表格型内容改成更清晰的 stats/key takeaway 结构。
- 将 v0.1 的 17 页调整为 v0.2 的 18 页，增加呼吸感。

技术处理：

- 尝试 `xelatex` + Helvetica Neue 时，发现本机 Helvetica Neue 触发 XeTeX 字体驱动错误。
- 尝试 `lualatex` 时，sandbox 中 LuaTeX font cache 不可写，不能稳定使用。
- 最终使用 `xelatex` + Arial fallback 编译，视觉仍保持 Helvetica/Calibri 系无衬线风格。
- 避免使用已被 LaTeX 占用的 `\accent` 宏名，改为 `\hl`。
- 对图表页进行高度约束和文字压缩，降低 overfull 风险。

生成并验证：

```bash
cd presentation
xelatex -interaction=nonstopmode final_presentation.tex
```

输出：

```text
presentation/final_presentation.tex
presentation/final_presentation.pdf
```

`pdfinfo` 显示新版 PDF 共 18 页，文件大小约 371 KB。编译后清理了 LaTeX 临时文件，`presentation/` 下只保留：

```text
presentation/final_presentation.tex
presentation/final_presentation.pdf
```

当前 git diff 显示：

```text
presentation/final_presentation.pdf | Bin 539828 -> 371281 bytes
presentation/final_presentation.tex | 636 ++++++++++++++++++------------------
```

---

**User:** 更新 prompt_history.md

**Assistant:** 将本次 v0.2 幻灯片视觉重做、编译验证、字体 fallback 处理和临时文件清理记录追加到 `prompt_history.md`。
