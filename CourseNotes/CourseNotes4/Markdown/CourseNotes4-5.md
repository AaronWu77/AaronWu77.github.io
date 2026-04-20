# Chapter 12 Design Concepts

## Good Design

Mitch Kapor 提出的设计观很有代表性：

- **Firmness**：程序不能有影响功能的 bug
- **Commodity**：程序要适合它原本被设计出来做的事情
- **Delight**：使用体验要让人舒服

也就是说，好的设计不只是“能跑”，还要**正确、合用、好用**。

## Software Design

Software design 包含一组指导系统开发的原则、概念和实践。

- **Design principles**：设计时要遵守的总体哲学
- **Design concepts**：设计前必须理解的核心概念
- **Design practices**：随着方法演进不断变化的具体做法

## Software Engineering Design

软件工程设计通常分成四个层次：

- **Data/Class design**：把分析类转成实现类和数据结构
- **Architectural design**：定义主要结构元素之间的关系
- **Interface design**：定义软件、硬件和用户如何通信
- **Component-level design**：把结构元素转成组件的过程描述

## Design and Quality

设计必须同时满足三件事：

- 实现 analysis model 中的显式需求
- 满足客户的隐含需求
- 能为编码、测试和维护提供清晰指南

换句话说，设计不是画图而已，它要能**指导后续工作**。

## Quality Guidelines

一个好的设计应该：

- 使用可识别的架构风格或模式
- 由设计良好的组件组成
- 能以渐进方式实现
- 具备模块化结构
- 具有清晰的数据、架构、接口和组件表示
- 生成合适的数据结构
- 让组件保持独立功能特征
- 降低组件和外部环境之间的连接复杂度
- 用可重复的方法从需求分析中推导出来
- 使用能有效表达含义的表示法

## Design Principles

- 不要陷入 **tunnel vision**
- 设计要能追踪到 analysis model
- 不要重复造轮子
- 尽量缩小 software 和 real world problem 之间的认知距离
- 设计要统一、整合
- 设计要能适应变化
- 即使遇到异常数据和异常环境，也要尽量优雅退化
- **Design is not coding, coding is not design**
- 设计应该在进行中就被评估，而不是做完再看
- 设计应该被 review，以减少语义层面的错误

## Fundamental Concepts

- **Abstraction**：抽象 data、procedure、control
- **Architecture**：软件的整体结构
- **Patterns**：表达经过验证的设计方案本质
- **Separation of concerns**：把复杂问题拆开处理
- **Modularity**：把数据和功能分隔到模块里
- **Hiding**：通过受控接口隐藏细节
- **Functional independence**：单一职责、低耦合
- **Refinement**：逐步细化抽象
- **Aspects**：帮助理解全局需求如何影响设计
- **Refactoring**：重组设计，使其更简单

## OO Design Concepts

设计类（design classes）负责给分析类补足实现细节。

- **Data abstraction**：把数据作为结构来实现
- **Procedural abstraction**：把过程封装成可执行算法
