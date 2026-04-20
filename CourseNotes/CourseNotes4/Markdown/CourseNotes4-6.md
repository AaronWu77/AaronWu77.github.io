# Chapter 13 Architectural Design

## Why Architecture?

Architecture 不是可运行的软件本身，而是软件结构的表示。  
它能帮助工程师：

- 分析设计是否满足需求
- 在还来得及修改的时候比较不同架构方案
- 降低实现风险

## Why is Architecture Important?

Architecture 的价值在于沟通。

- 让所有 stakeholders 对系统形成共同理解
- 把早期关键设计决策显式化
- 决定后续开发的大方向

它本质上是一种**容易理解的系统结构模式**。

## Architectural Descriptions

IEEE 推荐用 architectural description 来描述架构。

- 一个 architecture description 是一组文档产品
- 可以用多个 view 表示同一个系统
- 每个 view 都从某类 stakeholder concern 的角度看系统

## Architectural Genres and Styles

### Genres

Genre 可以理解为软件领域中的类别，不同类别里会有不同子类和风格。

### Styles

一个 architectural style 通常包含：

- components
- connectors
- constraints
- semantic models

常见架构风格包括：

- data-centered architectures
- data flow architectures
- call and return architectures
- object-oriented architectures
- layered architectures

## Architectural Patterns

架构模式常用于处理系统级问题：

- **Concurrency**：处理并发任务
- **Persistence**：处理持久化
- **Distribution**：处理分布式通信

例如：

- process management pattern
- task scheduler pattern
- DBMS-based persistence pattern
- application-level persistence pattern
- broker pattern

## Architectural Design

架构设计通常从这些问题开始：

1. 把软件放到 context 中看
2. 定义外部实体以及交互方式
3. 找出 architectural archetypes
4. 逐步细化系统组件结构

### Architectural Considerations

- **Economy**：保持简洁，避免不必要细节
- **Visibility**：让架构决策及其原因容易被看见
- **Spacing**：通过分离关注点减少隐藏依赖
- **Symmetry**：系统属性要一致、平衡
- **Emergence**：允许自组织行为和控制涌现

## Architectural Tradeoff Analysis

做架构权衡时，一般会：

- 收集场景
- 提炼需求、约束和环境
- 描述候选架构风格
- 分别评估质量属性
- 分析质量属性对架构属性的敏感性
- 评审候选架构

常用的视图包括：

- module view
- process view
- data flow view

## Architecture Reviews

架构评审的目标是：

- 检查架构是否满足系统质量要求
- 识别潜在风险
- 尽早发现设计问题，降低成本

常见方法有：

- experience-based reviews
- prototype evaluation
- scenario reviews
- checklists

## Agility and Architecture

敏捷开发里，架构也不能缺席。

- user stories 可以帮助演化 architectural model
- walking skeleton 可以在编码前先建立最小可行架构
- sprint 中产出的工作成果也应被拿来做架构评审

