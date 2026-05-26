# Copilot Instructions

## Commands

- **Local preview:** `python3 -m http.server 8000`
- Open `http://localhost:8000/` from the repository root.
- Use a local HTTP server instead of opening files directly, because note pages load Markdown with `fetch(...)`.
- There are no project-defined build, test, lint, or single-test commands in this repository.

## High-level architecture

- This repository is a static personal site served directly from the repo root. There is no app framework, package manager, or build pipeline.
- Root entry pages are hand-authored HTML files such as `index.html`, `photos.html`, `CourseNotes.html`, `ReadingPapers.html`, and `Music.html`.
- Shared site-wide presentation lives in `assets/css/style.css`, and shared interaction/animation lives in `assets/js/script.js`.
- `Photos/*.html` contains individual photography collection pages. Images are stored under `assets/images/...`, and album/list pages link to those HTML pages directly.
- `CourseNotes/*.html`, `CourseNotes/*/*.html`, and `ReadingPaper/*.html` follow a two-level structure:
  - list/index pages are plain HTML; top-level `CourseNotes.html` and `ReadingPapers.html` now render cards from `assets/data/content-manifest.json`
  - note/detail pages are HTML shells that fetch sibling Markdown from `Markdown/*.md`, then render it client-side with `marked`, `highlight.js`, and `KaTeX` through `assets/js/note-renderer.js`
- `assets/data/content-manifest.json` is the minimal content manifest for top-level note/paper list pages; update this file when adding new courses or reading-paper entries.
- The music page is self-contained in `Music.html`: UI markup is HTML/CSS, and playlist metadata is an inline JavaScript array rather than a separate data file.

## Key conventions

- Keep the shared cyberpunk/neon look in `assets/css/style.css`; page-specific layout changes are usually implemented as inline `<style>` blocks inside the page being edited.
- Most pages include `<div class="background-glow"></div>` plus the shared `assets/js/script.js` background animation. Preserve those when creating new pages in the same style.
- Relative asset paths depend on directory depth:
  - root pages use `assets/...`
  - one-level nested pages use `../assets/...`
  - two-level nested note pages use `../../assets/...`
- Navigation is manual. When adding or renaming content, update the relevant parent listing page as well as the destination page; there is no generated routing or shared content registry.
- For Markdown-backed note pages, keep the existing shell pattern:
  - set `window.__NOTE_CONFIG__ = { markdownFile: './Markdown/<name>.md' }`
  - load `assets/js/note-renderer.js` (relative path depends on nesting level)
  - keep `assets/js/script.js` for global background/lightbox behavior
  - preserve `marked`, `highlight.js`, and `KaTeX` CDN includes (the shared renderer depends on them)
- Image layout inside Markdown is controlled through words in the image alt text. The note shells map `right`, `left`, `small`, `medium`, and `large` to CSS classes after rendering.
- The photo lightbox code in `assets/js/script.js` only activates on pages that provide the expected DOM hooks (`#lightbox`, `#lightbox-img`, `#caption`, `.close-btn`, and `.gallery-container`). Stream-style photo pages without that markup only use the shared background animation.

## Planning and execution workflow

- When starting an execution or planning task, first provide a concrete execution plan with clear steps.
- If a task naturally breaks into multiple subtasks, ask the user about each subtask with `ask_user` before including it in the final plan.
- After the plan is finalized, present the user with the specific tasks and the execution plan in the conversation before proceeding.
- When a plan is saved in the session workspace, also save a readable copy to `.github/plan.md` in this repository so it is easy for the user to review later.
- Write plans in Chinese when possible so they are easier for the user to read and review.


## Plan 模式约束

1. 进入 plan 模式后，任何方案都不能直接定稿。
2. 每形成一个计划步骤、设计决策或执行方案，必须先调用 `ask_user` 逐项询问用户是否接受。
3. 只有当前步骤得到用户明确确认后，才能进入下一步。
4. 所有计划细节确认后，除更新会话内计划外，还必须在仓库根目录创建或更新 `/doc/plan.md` 供用户审阅。
5. 若已有 `/doc/plan.md`，每次计划更新后都要同步更新该文件，确保用户可查看最新计划。

## `/doc/plan.md` 固定结构

`/doc/plan.md` 必须按以下顺序编写：

1. **功能目的**：问题、目标、范围边界。
2. **TodoList**：任务拆分。
3. **具体执行方案**：实施阶段、涉及模块、落地方式。

## Autopilot 模式约束

1. 进入 autopilot 模式后，不得直接开始执行。
2. 每次准备实施前，必须先调用 `ask_user`，确认是否按当前 `/doc/plan.md` 执行。
3. 只有在用户明确同意后，才能继续修改代码、运行命令或推进任务。
4. 若 `/doc/plan.md` 不存在、已过期，或用户要求调整方案，必须先回到计划确认流程。
5. 完成修改后，提醒用户运行 `/review`。

## `/review` 约束

1. 用户要求 `/review` 时，先检查 `/doc/plan.md` 与本次改动是否一致。
2. 若审查通过，在本文档末尾维护简要更新日志；若无更新日志则主动创建。
3. 若审查发现问题，调用 `ask_user` 反馈问题，并在 `/doc/plan.md` 追加代码审查部分，标注问题与受影响 TodoList 项，最后提醒用户重新执行 `/plan`。

## 执行行为准则

### 1. 编码前思考

**不要假设。不要隐藏困惑。呈现权衡。**

- **明确说明假设** — 如果不确定，询问而不是猜测
- **呈现多种解释** — 当存在歧义时，不要默默选择
- **适时提出异议** — 如果存在更简单的方法，说出来
- **困惑时停下来** — 指出不清楚的地方并要求澄清

### 2. 简洁优先

**用最少的代码解决问题。不要过度推测。**

- 不要添加要求之外的功能
- 不要为一次性代码创建抽象
- 不要添加未要求的"灵活性"或"可配置性"
- 不要为不可能发生的场景做错误处理
- 如果 200 行代码可以写成 50 行，重写它

**检验标准：** 资深工程师会觉得这过于复杂吗？如果是，简化。

### 3. 精准修改

**只碰必须碰的。只清理自己造成的混乱。**

编辑现有代码时：

- 不要"改进"相邻的代码、注释或格式
- 不要重构没坏的东西
- 匹配现有风格，即使你更倾向于不同的写法
- 如果注意到无关的死代码，提一下 —— 不要删除它

当你的改动产生孤儿代码时：

- 删除因你的改动而变得无用的导入/变量/函数
- 不要删除预先存在的死代码，除非被要求

**检验标准：** 每一行修改都应该能直接追溯到用户的请求。

### 4. 目标驱动执行

**定义成功标准。循环验证直到达成。**

将指令式任务转化为可验证的目标：

| 不要这样做... | 转化为... |
|--------------|-----------------|
| "添加验证" | "为无效输入编写测试，然后让它们通过" |
| "修复 bug" | "编写重现 bug 的测试，然后让它通过" |
| "重构 X" | "确保重构前后测试都能通过" |

对于多步骤任务，说明一个简短的计划：

```
1. [步骤] → 验证: [检查]
2. [步骤] → 验证: [检查]
3. [步骤] → 验证: [检查]
```
