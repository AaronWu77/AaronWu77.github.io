# CourseNotes 模板化改造计划（覆盖旧计划）

## 问题说明

当前 `CourseNotes` 目录下大量笔记详情页 HTML（如 `CourseNotes4-1.html`、`CourseNotes4-2.html`）结构几乎完全相同，只是 `window.__NOTE_CONFIG__.markdownFile` 的路径不同。  
这导致每新增一篇笔记都必须复制一个 HTML 壳文件，维护成本高、容易出错、文件数量膨胀。

## 目标与约束

- **目标**：将“每篇笔记一个 HTML”改为“统一详情模板 + 参数驱动 Markdown 文件”。
- **约束**：你已确认**不兼容旧链接**，只保证新链接全部可用。
- **保留项**：保持现有赛博风格、渲染链路（marked / highlight.js / KaTeX）和当前目录组织。

## 方案概述

1. 新增一个通用详情页模板（例如 `CourseNotes/note.html`）。
2. 通过 URL 参数传入：
   - `md`：目标 Markdown 文件路径（相对于 `CourseNotes/`）。
   - `back`：返回列表页链接（如 `./CourseNotes4.html`）。
   - `title`（可选）：页面标题。
3. 扩展 `assets/js/note-renderer.js`，支持从 URL 参数读取 Markdown 路径（并保留对 `window.__NOTE_CONFIG__` 的兼容逻辑）。
4. 批量更新 `CourseNotes1~8` 列表页里的 “Read Notes” 链接，统一指向新模板页。
5. 暂不删除旧详情页文件（避免一次性大规模文件删除风险）；后续可单独做清理批次。

## 执行计划

### 第 1 阶段：模板与参数协议设计
- 确定 `note.html` 的参数协议（`md/back/title`）和默认行为。
- 定义路径安全规则：限制 `md` 仅允许 `CourseNotes/` 目录内的相对路径，避免非法路径访问。
- 确定标题优先级：`title` 参数 > 默认标题。

### 第 2 阶段：实现统一模板页
- 新建 `CourseNotes/note.html`，复用现有详情页视觉结构与依赖引入方式。
- 在模板内读取 URL 参数，设置页面标题、返回按钮和 `window.__NOTE_CONFIG__`。
- 接入现有 `assets/js/note-renderer.js` 完成渲染。

### 第 3 阶段：增强渲染器兼容能力
- 在 `assets/js/note-renderer.js` 增加“参数驱动”入口：
  - 优先使用显式传入配置；
  - 其次读取 URL 参数；
  - 最后回退到 `window.__NOTE_CONFIG__`。
- 对 `md` 参数增加基础校验与错误提示，保证异常时页面可感知而不是静默失败。

### 第 4 阶段：批量迁移入口链接
- 更新 `CourseNotes/CourseNotes1.html` 到 `CourseNotes/CourseNotes8.html` 所有详情链接：
  - 从 `./CourseNotesX/CourseNotesX-Y.html`
  - 改为 `./note.html?md=./CourseNotesX/Markdown/CourseNotesX-Y.md&back=./CourseNotesX.html&title=...`
- 抽样检查每个课程页至少一条链接，确保跳转和渲染链路正确。

### 第 5 阶段：验证与收尾
- 全面检查新链接在本地服务器下可访问、可返回、可渲染。
- 检查图片相对路径、公式、代码高亮是否与原行为一致。
- 如有必要，在 `.github/plan.md` 同步最终执行记录与后续可选清理项（旧详情页删除）。

## 风险与应对

- **风险 1：参数路径错误导致 404**  
  应对：新增清晰的错误提示与路径校验，迁移时做批量核对。

- **风险 2：相对路径层级不一致导致资源错位**  
  应对：统一把模板固定在 `CourseNotes/` 目录，所有参数按该目录相对路径组织。

- **风险 3：一次性迁移链接量大**  
  应对：按课程页分批迁移，逐页核验后再继续。

## 本轮交付范围（实现阶段）

- 仅改造 `CourseNotes` 体系（不含 `ReadingPaper`）。
- 旧链接不保证继续可用，新链接必须全部正常运行。
