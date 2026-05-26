# 计算机体系结构课程笔记撰写计划

## 功能目的

**目标**：为浙江大学计算机体系结构课程的 14 份 PPT 各生成一份详细的中文学习笔记，使学生仅凭笔记即可完整学习整门课程。

**范围边界**：
- 仅处理 `Arch_1` 到 `Arch_14` 共 14 个课程 PPT（排除 DeepSeek 技术报告）
- 每个 PPT 对应一个独立的 `.md` 文件（共 14 个）
- 只创建 Markdown 文件，不修改现有 HTML

**笔记风格要求**：
- 主要使用中文，关键英文术语保留标注
- 越详细越好，将原理和推导过程解释清楚
- 不只是"抄写"PPT 内容，而是解释清楚每个概念的来龙去脉

---

## TodoList

| 编号 | 文件名 | 对应PPT | 幻灯片数 | 状态 |
|------|--------|---------|----------|------|
| 1 | `Arch1_Intro.md` | Arch_1_intro.pptx | 59 | ⏳ 待完成 |
| 2 | `Arch2_Ch1_Fundamentals1.md` | Arch_2_ch1_1_fundamentals1.pptx | 64 | ⏳ 待完成 |
| 3 | `Arch3_Ch1_Fundamentals2.md` | Arch_3_ch1_2_fundamentals2.pptx | 74 | ⏳ 待完成 |
| 4 | `Arch4_Pipeline.md` | Arch_4_pipeline.pptx | 106 | ⏳ 待完成 |
| 5 | `Arch5_Ch2_CacheBasics.md` | Arch_5_ch2_1_cache_basics.pptx | 38 | ⏳ 待完成 |
| 6 | `Arch6_Ch2_CacheMiss.md` | Arch_6_ch2_2_cache_miss.pptx | 71 | ⏳ 待完成 |
| 7 | `Arch7_Ch2_MemoryTech.md` | Arch_7_ch2_3_memory_technology.pptx | 28 | ⏳ 待完成 |
| 8 | `Arch8_Ch3_DynamicScheduling.md` | Arch_8_ch3_1_dynamic_scheduling.pptx | 158 | ⏳ 待完成 |
| 9 | `Arch9_Ch3_BranchPredictor.md` | Arch_9_ch3_2_branch_predictor.pptx | 56 | ⏳ 待完成 |
| 10 | `Arch10_Ch3_Speculation.md` | Arch_10_ch3_3_speculation.pptx | 42 | ⏳ 待完成 |
| 11 | `Arch11_Ch3_SuperscalarVLIW.md` | Arch_11_ch3_4_superscalar_VLIW.pptx | 47 | ⏳ 待完成 |
| 12 | `Arch12_Ch3_Multithreading.md` | Arch_12_ch3_5_multithreading.pptx | 67 | ⏳ 待完成 |
| 13 | `Arch13_Ch4_DLP.md` | Arch_13_ch4_dlp_vector_simd_gpu.pptx | 71 | ⏳ 待完成 |
| 14 | `Arch14_Ch5_Multiprocessor.md` | Arch_14_ch5_1_multiprocessor.pptx | 38 | ⏳ 待完成 |

**总计**：919 张幻灯片 → 14 份 Markdown 笔记

---

## 具体执行方案

### 第一阶段：PPT 内容提取

**工具**：`python-pptx`（已确认可用）

**提取逻辑**：
1. 逐张遍历幻灯片，提取所有文本框内容（标题 + 正文段落）
2. 保留缩进层级（通过 `paragraph.level` 判断）
3. 按顺序输出结构化文本，作为笔记撰写的原始素材

### 第二阶段：逐个PPT撰写笔记

**执行顺序**：按 Arch_1 → Arch_14 顺序逐一完成，便于前后概念衔接。

**每份笔记的结构模板**：
```markdown
# [章节标题]

## 一、概述
（本章核心问题 + 学习目标）

## 二、[主要概念1]
### 2.1 定义与动机
### 2.2 原理详解
### 2.3 关键公式/算法

## 三、[主要概念2]
...

## 总结
（知识点一览 + 与前后章节的联系）
```

**撰写原则**：
- 每个概念不只照搬PPT原文，而是用自己的语言解释清楚**为什么**需要这个设计、**如何**工作
- 关键英文缩写首次出现时给出全称，例如：IPC（Instructions Per Cycle，每时钟周期指令数）
- 对于公式推导，补充推导步骤和物理含义
- 对于重要图表（如流水线图、Cache结构图），用文字详细描述其含义

### 第三阶段：文件存放位置

所有笔记保存到：
```
CourseNotes/计算机体系结构/Markdown/
├── Arch1_Intro.md
├── Arch2_Ch1_Fundamentals1.md
├── Arch3_Ch1_Fundamentals2.md
├── Arch4_Pipeline.md
├── Arch5_Ch2_CacheBasics.md
├── Arch6_Ch2_CacheMiss.md
├── Arch7_Ch2_MemoryTech.md
├── Arch8_Ch3_DynamicScheduling.md
├── Arch9_Ch3_BranchPredictor.md
├── Arch10_Ch3_Speculation.md
├── Arch11_Ch3_SuperscalarVLIW.md
├── Arch12_Ch3_Multithreading.md
├── Arch13_Ch4_DLP.md
└── Arch14_Ch5_Multiprocessor.md
```
