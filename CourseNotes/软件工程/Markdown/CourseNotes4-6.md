# Chapter 13 Architectural Design

## Why Architecture?

The architecture is not the operational software. Rather, it is a representation that enables a software engineer to:  
架构不是可运行的软件，而是一种表示，它使软件工程师能够：

1. analyze the effectiveness of the design in meeting its stated requirements,  
   分析设计满足既定需求的有效性；
2. consider architectural alternatives at a stage when making design changes is still relatively easy, and  
   在设计修改仍相对容易时考虑不同架构方案；
3. reduce the risks associated with the construction of the software.  
   降低软件构建相关风险。

## Why is Architecture Important?

Representations of software architecture are an enabler for communication between all parties (stakeholders) interested in the development of a computer-based system.  
软件架构的表示有助于所有关心计算机系统开发的相关方（stakeholders）之间进行沟通。

The architecture highlights early design decisions that will have a profound impact on all software engineering work that follows and, as important, on the ultimate success of the system as an operational entity.  
架构强调早期设计决策，而这些决策会深刻影响后续所有软件工程工作，也会影响系统作为运行实体的最终成败。

Architecture constitutes a relatively small, intellectually graspable model of how the system is structured and how its components work together.  
架构构成了一个相对小而易于理解的模型，用来说明系统如何组织以及各组件如何协同。

## Architectural Descriptions

The IEEE Computer Society has proposed IEEE-Std-1471-2000, Recommended Practice for Architectural Description of Software-Intensive System.  
IEEE Computer Society 提出了 IEEE-Std-1471-2000，用于软件密集型系统架构描述的推荐实践。

It is used:

- to establish a conceptual framework and vocabulary for use during the design of software architecture,  
  建立软件架构设计时使用的概念框架和术语；
- to provide detailed guidelines for representing an architectural description, and  
  为架构描述的表示提供详细指导；
- to encourage sound architectural design practices.  
  鼓励良好的架构设计实践。

The IEEE Standard defines an architectural description (AD) as a collection of products to document an architecture.  
IEEE 标准将 architectural description（AD）定义为一组用于记录架构的产品。

The description itself is represented using multiple views, where each view is a representation of a whole system from the perspective of a related set of stakeholder concerns.  
该描述通常通过多个视图来表示，每个视图都从一组相关利益相关者关注点的角度表示整个系统。

## Architectural Genres

Genre implies a specific category within the overall software domain.  
Genre 指软件总体领域中的特定类别。

Within each category, you encounter a number of subcategories.  
每个类别中还会有若干子类别。

For example, within the genre of buildings, you would encounter houses, condos, apartment buildings, office buildings, industrial building, warehouses, and so on.  
例如在建筑领域中，会有房屋、公寓、办公楼、工业建筑、仓库等。

Within each general style, more specific styles might apply. Each style would have a structure that can be described using a set of predictable patterns.  
每个通用风格下还可能存在更具体的风格，而每种风格都可以用一组可预测的模式来描述结构。

## Architectural Styles

Each style describes a system category that encompasses:

1. a set of components that perform a function required by a system,  
   一组执行系统所需功能的组件；
2. a set of connectors that enable communication, coordination and cooperation among components,  
   一组使组件之间能够通信、协调与协作的连接器；
3. constraints that define how components can be integrated to form the system, and  
   定义组件如何集成形成系统的约束；
4. semantic models that enable a designer to understand the overall properties of a system by analyzing the known properties of its constituent parts.  
   通过分析组成部分已知属性来理解系统整体特性的语义模型。

Common styles:

- Data-centered architectures  
  数据中心架构
- Data flow architectures  
  数据流架构
- Call and return architectures  
  调用与返回架构
- Object-oriented architectures  
  面向对象架构
- Layered architectures  
  分层架构

## Architectural Patterns

- **Concurrency**：applications must handle multiple tasks in a manner that simulates parallelism  
  **并发**：应用必须以模拟并行的方式处理多个任务
- **Persistence**：Data persists if it survives past the execution of the process that created it. Two patterns are common:  
  **持久化**：如果数据能在创建它的进程结束后仍然存在，就称为持久化。常见两种模式：
  - a database management system pattern that applies the storage and retrieval capability of a DBMS to the application architecture  
    将 DBMS 的存储和检索能力应用到应用架构中；
  - an application level persistence pattern that builds persistence features into the application architecture  
    在应用架构中直接构建持久化特性。
- **Distribution**：the manner in which systems or components within systems communicate with one another in a distributed environment  
  **分布式**：系统或系统内组件在分布式环境中彼此通信的方式

A broker acts as a middle-man between the client component and a server component.  
broker 作为客户端组件和服务端组件之间的中间人。

## Architectural Design

- The software must be placed into context  
  软件必须放到上下文中考虑
- the design should define the external entities that the software interacts with and the nature of the interaction  
  设计应定义软件交互的外部实体及交互性质
- A set of architectural archetypes should be identified  
  应识别一组架构原型
- An archetype is an abstraction that represents one element of system behavior  
  archetype 是表示系统行为某一元素的抽象
- The designer specifies the structure of the system by defining and refining software components that implement each archetype  
  设计者通过定义和细化实现每个 archetype 的软件组件来指定系统结构

## Architectural Considerations

- **Economy**：The best software is uncluttered and relies on abstraction to reduce unnecessary detail.  
  **经济性**：最好的软件是简洁的，并依赖抽象减少不必要的细节。
- **Visibility**：Architectural decisions and the reasons for them should be obvious to software engineers who examine the model at a later time.  
  **可见性**：架构决策及其原因应对之后查看模型的软件工程师清晰可见。
- **Spacing**：Separation of concerns in a design without introducing hidden dependencies.  
  **间隔 / 分离**：在不引入隐藏依赖的情况下进行关注点分离。
- **Symmetry**：Architectural symmetry implies that a system is consistent and balanced in its attributes.  
  **对称性**：架构对称意味着系统属性一致且平衡。
- **Emergence**：Emergent, self-organized behavior and control.  
  **涌现性**：涌现式、自组织的行为和控制。

## Architectural Tradeoff Analysis

1. Collect scenarios.  
   收集场景。
2. Elicit requirements, constraints, and environment description.  
   提取需求、约束和环境描述。
3. Describe the architectural styles/patterns that have been chosen to address the scenarios and requirements:  
   描述为解决场景和需求而选择的架构风格 / 模式：
   - module view  
     模块视图
   - process view  
     过程视图
   - data flow view  
     数据流视图
4. Evaluate quality attributes by considered each attribute in isolation.  
   分别评估各个质量属性。
5. Identify the sensitivity of quality attributes to various architectural attributes for a specific architectural style.  
   识别某种架构风格下质量属性对架构属性的敏感性。
6. Critique candidate architectures using the sensitivity analysis conducted in step 5.  
   利用第 5 步的敏感性分析批评候选架构。

## Architectural Complexity

the overall complexity of a proposed architecture is assessed by considering the dependencies between components within the architecture.  
一个架构方案的整体复杂度，可以通过考虑架构内部组件之间的依赖关系来评估。

- **Sharing dependencies**：dependence relationships among consumers who use the same resource or producers who produce for the same consumers.  
  **共享依赖**：使用同一资源的消费者之间，或为同一消费者生产的生产者之间的依赖关系。
- **Flow dependencies**：dependence relationships between producers and consumers of resources.  
  **流依赖**：资源生产者和消费者之间的依赖关系。
- **Constrained dependencies**：constraints on the relative flow of control among a set of activities.  
  **约束依赖**：一组活动之间控制流相对顺序上的约束。

## Architectural Description Language

Architectural description language (ADL) provides a semantics and syntax for describing a software architecture.  
架构描述语言（ADL）为描述软件架构提供语义和语法。

It provides the designer with the ability to:

- decompose architectural components  
  分解架构组件
- compose individual components into larger architectural blocks  
  将单个组件组合成更大的架构块
- represent interfaces (connection mechanisms) between components  
  表示组件之间的接口（连接机制）

## Architecture Reviews

- Assess the ability of the software architecture to meet the systems quality requirements and identify potential risks  
  评估软件架构满足系统质量需求的能力并识别潜在风险
- Have the potential to reduce project costs by detecting design problems early  
  通过尽早发现设计问题来降低项目成本
- Often make use of experience-based reviews, prototype evaluation, scenario reviews, and checklists  
  常采用基于经验的评审、原型评估、场景评审和检查表

## Pattern-Based Architecture Review

- Identify and discuss the quality attributes by walking through the use cases.  
  通过遍历用例识别并讨论质量属性。
- Discuss a diagram of system’s architecture in relation to its requirements.  
  结合需求讨论系统架构图。
- Identify the architecture patterns used and match the system’s structure to the patterns’ structure.  
  识别使用的架构模式，并将系统结构与模式结构进行匹配。
- Use existing documentation and use cases to determine each pattern’s effect on quality attributes.  
  利用现有文档和用例判断每种模式对质量属性的影响。
- Identify all quality issues raised by architecture patterns used in the design.  
  识别设计中架构模式带来的所有质量问题。
- Develop a short summary of issues uncovered during the meeting and make revisions to the walking skeleton.  
  总结会议中发现的问题，并修改 walking skeleton。

## Agility and Architecture

- To avoid rework, user stories are used to create and evolve an architectural model (walking skeleton) before coding  
  为避免返工，用户故事会在编码前用于创建并演化架构模型（walking skeleton）。
- Hybrid models which allow software architects contributing users stories to the evolving storyboard  
  混合模型允许软件架构师为不断演进的 storyboard 贡献用户故事。
- Well run agile projects include delivery of work products during each sprint  
  运作良好的敏捷项目会在每个 sprint 交付工作产品。
- Reviewing code emerging from the sprint can be a useful form of architectural review  
  审查 sprint 产出的代码可以成为一种有用的架构评审形式。
