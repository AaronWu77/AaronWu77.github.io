# Chapter 14 Component-Level Design

## What is a Component?

Component 是系统中一个可模块化、可部署、可替换的部分。

- **OO view**：component 包含一组协作类
- **Conventional view**：component 包含处理逻辑、内部数据结构和接口

## Basic Design Principles

组件设计的核心原则包括：

- **OCP**：对扩展开放，对修改关闭
- **LSP**：子类应可替代父类
- **DIP**：依赖抽象，不依赖具体实现
- **ISP**：多个专用接口优于一个大而全接口
- **REP**：复用粒度应与发布粒度一致
- **CCP**：一起变化的类应该放在一起
- **CRP**：不一起复用的类不应强行放一起

## Design Guidelines

### Components

- 组件命名应与 architectural model 保持一致

### Interfaces

- 接口要清楚表达通信和协作方式

### Dependencies and Inheritance

- 依赖关系尽量从左到右
- 继承关系尽量从下到上

## Cohesion and Coupling

### Cohesion

cohesion 是组件内部职责是否单一、相关。

常见层次：

- functional
- layer
- communicational
- sequential
- procedural
- temporal
- utility

### Coupling

coupling 是组件之间的连接程度。

常见层次：

- content
- common
- control
- stamp
- data
- routine call
- type use
- inclusion/import
- external

## Component Level Design Steps

1. 找出问题域对应的设计类
2. 找出基础设施域对应的设计类
3. 对未复用组件的设计类进行展开
4. 描述持久数据源并找出管理类
5. 为类或组件建立行为表示
6. 细化部署图
7. 对设计表示进行整理，并始终考虑替代方案

## WebApp Component Design

WebApp 的 component 往往同时包含：

- content design
- functional design

### Content Design for WebApps

重点是内容对象如何组织成适合展示的包。

### Functional Design for WebApps

重点是处理逻辑、数据结构和接口协作。

## Component-Based Development

复用时通常会问：

- 是否有现成的 COTS 组件？
- 是否有内部可复用组件？
- 现有组件接口是否兼容当前架构？

常见阻碍包括：

- 缺少完整复用计划
- 工具和训练不足
- 组织不鼓励复用
- 缺少激励机制

## CBSE Activities

组件化开发通常包括：

- component qualification
- component adaptation
- component composition
- component update

## Reuse and Component Systems

复用组件时要关心：

- API
- 开发和集成工具
- 运行时资源需求
- 服务接口
- 安全特性
- 内嵌算法假设
- 异常处理

### Common Infrastructure

常见的组件基础设施包括：

- **OMG/CORBA**
- **COM**
- **JavaBeans**

它们的目标都是：**让组件可以被发现、组合、集成和复用**。

## Reuse Environment

一个完整的复用环境通常包含：

- component database
- classification information
- library management system
- retrieval system

这样才能支持组件的检索、组合和维护。
