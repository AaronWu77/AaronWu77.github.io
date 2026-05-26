# Chapter 12 Design Concepts

Mitch Kapor, the creator of Lotus 1-2-3, presented a software design manifesto in *Dr. Dobbs Journal*. He said:  
Lotus 1-2-3 的创造者 Mitch Kapor 在 *Dr. Dobbs Journal* 上提出了一个软件设计宣言：

Good software design should exhibit:  
好的软件设计应该体现：

- **Firmness**: A program should not have any bugs that inhibit its function.  
  **Firmness（稳固性）**：程序不应该有阻碍其功能的 bug。
- **Commodity**: A program should be suitable for the purposes for which it was intended.  
  **Commodity（适用性）**：程序应该适合它被设计出来所要完成的目的。
- **Delight**: The experience of using the program should be pleasurable.  
  **Delight（愉悦性）**：使用程序的体验应该是愉快的。

## Design

Software Design encompasses the set of principles, concepts, and practices that lead to the development of a high quality system or product.  
软件设计包含一组原则、概念和实践，它们共同促成高质量系统或产品的开发。

Design principles establish the overriding philosophy that guides the designer as the work is performed.  
设计原则建立的是贯穿整个设计过程的总体哲学。

Design concepts must be understood before the mechanics of design practice are applied.  
在应用具体设计实践之前，必须先理解设计概念。

Software design practices change continuously as new methods, better analysis, and broader understanding evolve.  
随着新方法、更好的分析和更广的理解不断发展，软件设计实践也会持续变化。

## Software Engineering Design

- **Data/Class design** transforms analysis classes into implementation classes and data structures.  
  **数据 / 类设计**：把分析类转换成实现类和数据结构。
- **Architectural design** defines relationships among the major software structural elements.  
  **架构设计**：定义主要软件结构元素之间的关系。
- **Interface design** defines how software elements, hardware elements, and end-users communicate.  
  **接口设计**：定义软件元素、硬件元素和终端用户如何通信。
- **Component-level design** transforms structural elements into procedural descriptions of software components.  
  **组件级设计**：把结构元素转换成软件组件的过程描述。

## Design and Quality

the design must implement all of the explicit requirements contained in the analysis model, and it must accommodate all of the implicit requirements desired by the customer.  
设计必须实现分析模型中所有显式需求，并满足客户希望的隐式需求。

the design must be a readable, understandable guide for those who generate code and for those who test and subsequently support the software.  
设计必须对编写代码、测试以及后续支持软件的人来说是可读、可理解的指南。

the design should provide a complete picture of the software, addressing the data, functional, and behavioral domains from an implementation perspective.  
设计应该从实现角度给出软件的完整图景，覆盖数据、功能和行为领域。

## Quality Guidelines

A design should exhibit an architecture that (1) has been created using recognizable architectural styles or patterns, (2) is composed of components that exhibit good design characteristics and (3) can be implemented in an evolutionary fashion.  
设计应体现这样的架构：(1) 使用可识别的架构风格或模式创建；(2) 由具有良好设计特性的组件组成；(3) 能够以渐进方式实现。

A design should be modular; that is, the software should be logically partitioned into elements or subsystems.  
设计应具有模块化，也就是把软件在逻辑上划分成元素或子系统。

A design should contain distinct representations of data, architecture, interfaces, and components.  
设计应包含数据、架构、接口和组件的清晰表示。

A design should lead to data structures that are appropriate for the classes to be implemented and are drawn from recognizable data patterns.  
设计应导向适合待实现类的数据结构，并且这些结构应来自可识别的数据模式。

A design should lead to components that exhibit independent functional characteristics.  
设计应导向具有独立功能特征的组件。

A design should lead to interfaces that reduce the complexity of connections between components and with the external environment.  
设计应导向能够降低组件之间以及与外部环境连接复杂度的接口。

A design should be derived using a repeatable method that is driven by information obtained during software requirements analysis.  
设计应通过可重复的方法推导，并由软件需求分析获得的信息驱动。

A design should be represented using a notation that effectively communicates its meaning.  
设计应使用能有效传达含义的表示法来表达。

## Design Principles

- The design process should not suffer from tunnel vision.  
  设计过程不应陷入隧道视野。
- The design should be traceable to the analysis model.  
  设计应能追溯到分析模型。
- The design should not reinvent the wheel.  
  设计不应重复造轮子。
- The design should minimize the intellectual distance between the software and the problem as it exists in the real world.  
  设计应尽量缩小软件与现实世界问题之间的认知距离。
- The design should exhibit uniformity and integration.  
  设计应体现统一性和整合性。
- The design should be structured to accommodate change.  
  设计应能适应变化。
- The design should be structured to degrade gently, even when aberrant data, events, or operating conditions are encountered.  
  即使遇到异常数据、事件或运行条件，设计也应能够平缓退化。
- Design is not coding, coding is not design.  
  设计不是编码，编码也不是设计。
- The design should be assessed for quality as it is being created, not after the fact.  
  设计应在创建过程中就进行质量评估，而不是事后才评估。
- The design should be reviewed to minimize conceptual (semantic) errors.  
  设计应经过评审，以尽量减少概念（语义）错误。

## Fundamental Concepts

- **Abstraction**：data, procedure, control  
  **抽象**：数据、过程、控制
- **Architecture**：the overall structure of the software  
  **架构**：软件的整体结构
- **Patterns**：conveys the essence of a proven design solution  
  **模式**：传达经过验证的设计方案的本质
- **Separation of concerns**：any complex problem can be more easily handled if it is subdivided into pieces  
  **关注点分离**：复杂问题拆分成多个部分后更容易处理
- **Modularity**：compartmentalization of data and function  
  **模块化**：数据与功能的分隔
- **Hiding**：controlled interfaces  
  **隐藏**：通过受控接口隐藏细节
- **Functional independence**：single-minded function and low coupling  
  **功能独立性**：单一功能、低耦合
- **Refinement**：elaboration of detail for all abstractions  
  **细化**：逐步展开抽象细节
- **Aspects**：a mechanism for understanding how global requirements affect design  
  **方面 / 切面**：理解全局需求如何影响设计的一种机制
- **Refactoring**：a reorganization technique that simplifies the design  
  **重构**：简化设计的一种重组技术

## OO Design Concepts

## Design Classes

provide design detail that will enable analysis classes to be implemented  
设计类提供实现分析类所需的设计细节。

## Data Abstraction

implemented as a data structure  
作为数据结构来实现。

## Procedural Abstraction

implemented with a "knowledge" of the object that is associated with enter details of enter algorithm  
通过与对象相关的“知识”以及进入细节的算法来实现。
