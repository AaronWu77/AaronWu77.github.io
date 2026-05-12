# Chapter 14 Component-Level Design

## What is a Component?

OMG Unified Modeling Language Specification [OMG01] defines a component as a modular, deployable, and replaceable part of a system that encapsulates implementation and exposes a set of interfaces.  
OMG UML 规范将 component 定义为系统中一个模块化、可部署、可替换的部分，它封装实现并暴露一组接口。

- **OO view**：a component contains a set of collaborating classes  
  **面向对象视角**：component 包含一组协作类
- **Conventional view**：a component contains processing logic, the internal data structures that are required to implement the processing logic, and an interface that enables the component to be invoked and data to be passed to it.  
  **传统视角**：component 包含处理逻辑、实现处理逻辑所需的内部数据结构，以及用于调用和传递数据的接口

## Basic Design Principles

- **Open-Closed Principle (OCP)**：A module [component] should be open for extension but closed for modification.  
  **开闭原则**：模块 / 组件应对扩展开放，对修改关闭。
- **Liskov Substitution Principle (LSP)**：Subclasses should be substitutable for their base classes.  
  **里氏替换原则**：子类应可替代其基类。
- **Dependency Inversion Principle (DIP)**：Depend on abstractions. Do not depend on concretions.  
  **依赖倒置原则**：依赖抽象，不依赖具体实现。
- **Interface Segregation Principle (ISP)**：Many client-specific interfaces are better than one general purpose interface.  
  **接口隔离原则**：多个面向客户端的接口优于一个通用接口。
- **Release Reuse Equivalency Principle (REP)**：The granule of reuse is the granule of release.  
  **复用发布等价原则**：复用粒度就是发布粒度。
- **Common Closure Principle (CCP)**：Classes that change together belong together.  
  **共同闭包原则**：一起变化的类应放在一起。
- **Common Reuse Principle (CRP)**：Classes that aren’t reused together should not be grouped together.  
  **共同复用原则**：不一起复用的类不应分在一起。

## Design Guidelines

- Naming conventions should be established for components that are specified as part of the architectural model and then refined and elaborated as part of the component-level model  
  组件命名应先在架构模型中建立约定，再在组件级模型中细化。
- Interfaces provide important information about communication and collaboration.  
  接口提供关于通信和协作的重要信息。
- Dependencies and inheritance should be modeled from left to right and from bottom (derived classes) to top (base classes).  
  依赖与继承应分别从左到右、从下到上建模。

## Cohesion

the single-mindedness of a module  
模块的单一目标性。

OO view: cohesion implies that a component or class encapsulates only attributes and operations that are closely related to one another and to the class or component itself  
面向对象视角：cohesion 表示组件或类只封装彼此以及与自身密切相关的属性和操作。

### Levels of cohesion

- Functional  
  功能性
- Layer  
  层次性
- Communicational  
  通信性
- Sequential  
  顺序性
- Procedural  
  过程性
- Temporal  
  时间性
- Utility  
  工具性

## Coupling

The degree to which a component is connected to other components and to the external world  
组件与其他组件及外部世界的连接程度。

OO view: a qualitative measure of the degree to which classes are connected to one another  
面向对象视角：类彼此连接程度的定性度量。

### Level of coupling

- Content  
  内容耦合
- Common  
  公共耦合
- Control  
  控制耦合
- Stamp  
  印记耦合
- Data  
  数据耦合
- Routine call  
  例程调用耦合
- Type use  
  类型使用耦合
- Inclusion or import  
  包含 / 导入耦合
- External  
  外部耦合

## Component Level Design Steps

1. Identify all design classes that correspond to the problem domain.  
   识别与问题域对应的所有设计类。
2. Identify all design classes that correspond to the infrastructure domain.  
   识别与基础设施域对应的所有设计类。
3. Elaborate all design classes that are not acquired as reusable components.  
   展开所有未作为可复用组件获取的设计类。
4. Specify message details when classes or component collaborate.  
   当类或组件协作时，指定消息细节。
5. Identify appropriate interfaces for each component.  
   为每个组件识别合适的接口。
6. Elaborate attributes and define data types and data structures required to implement them.  
   展开属性并定义实现它们所需的数据类型和数据结构。
7. Describe processing flow within each operation in detail.  
   详细描述每个操作内部的处理流程。
8. Describe persistent data sources (databases and files) and identify the classes required to manage them.  
   描述持久化数据源（数据库和文件），并识别管理它们所需的类。
9. Develop and elaborate behavioral representations for a class or component.  
   为类或组件开发并细化行为表示。
10. Elaborate deployment diagrams to provide additional implementation detail.  
   细化部署图，提供更多实现细节。
11. Factor every component-level design representation and always consider alternatives.  
   对每一种组件级设计表示进行分解，并始终考虑替代方案。

## Component-Level Design - II

- Collaboration Diagram  
  协作图
- Activity Diagram  
  活动图
- Statechart  
  状态图
- Refactoring  
  重构

## Component Design for WebApps

WebApp component is:

1. a well-defined cohesive function that manipulates content or provides computational or data processing for an end-user, or  
   一个定义清晰、内聚的功能，用于操作内容或为终端用户提供计算 / 数据处理；
2. a cohesive package of content and functionality that provides end-user with some required capability.  
   一个由内容和功能组成的内聚包，为终端用户提供某种所需能力。

Therefore, component-level design for WebApps often incorporates elements of content design and functional design.  
因此，WebApp 的组件级设计通常会包含内容设计和功能设计的元素。

## Content Design for WebApps

focuses on content objects and the manner in which they may be packaged for presentation to a WebApp end-user  
关注内容对象，以及它们如何被打包后展示给 WebApp 终端用户。

For example, consider a Web-based video surveillance capability within SafeHomeAssured.com:

1. the content objects that represent the space layout (the floor plan) with additional icons representing the location of sensors and video cameras;  
   表示空间布局（平面图）的内容对象，并用额外图标表示传感器和摄像头位置；
2. the collection of thumbnail video captures (each a separate data object), and  
   缩略视频捕获集合（每个都是独立数据对象）；
3. the streaming video window for a specific camera.  
   某个摄像头的流视频窗口。

Each of these components can be separately named and manipulated as a package.  
这些组件都可以单独命名并作为一个包进行操作。

## Functional Design for WebApps

Modern Web applications deliver increasingly sophisticated processing functions that:

1. perform localized processing to generate content and navigation capability in a dynamic fashion;  
   以动态方式执行局部处理，生成内容和导航能力；
2. provide computation or data processing capability that is appropriate for the WebApp’s business domain;  
   提供适合 WebApp 业务领域的计算或数据处理能力；
3. provide sophisticated database query and access, or  
   提供复杂数据库查询和访问；
4. establish data interfaces with external corporate systems.  
   与外部企业系统建立数据接口。

To achieve these capabilities, you will design and construct WebApp functional components that are identical in form to software components for conventional software.  
为了实现这些能力，需要设计并构建与传统软件组件形式相同的 WebApp 功能组件。

## Designing Conventional Components

The design of processing logic is governed by the basic principles of algorithm design and structured programming.  
处理逻辑的设计受算法设计和结构化编程基本原则约束。

The design of data structures is defined by the data model developed for the system.  
数据结构的设计由系统的数据模型决定。

The design of interfaces is governed by the collaborations that a component must effect.  
接口设计由组件必须实现的协作关系决定。

## Component-Based Development

When faced with the possibility of reuse, the software team asks:

- Are commercial off-the-shelf (COTS) components available to implement the requirement?  
  是否有现成的商用现成组件（COTS）可以实现该需求？
- Are internally-developed reusable components available to implement the requirement?  
  是否有内部开发的可复用组件可以实现该需求？
- Are the interfaces for available components compatible within the architecture of the system to be built?  
  可用组件的接口是否与待构建系统的架构兼容？

At the same time, they are faced with the following impediments to reuse:

- Few companies and organizations have anything that even slightly resembles a comprehensive software reusability plan.  
  很少有公司或组织拥有真正全面的软件复用计划。
- Although an increasing number of software vendors currently sell tools or components that provide direct assistance for software reuse, the majority of software developers do not use them.  
  虽然越来越多厂商提供复用工具或组件，但大多数开发者并未使用。
- Relatively little training is available to help software engineers and managers understand and apply reuse.  
  可用于帮助工程师和管理者理解并应用复用的培训相对较少。
- Many software practitioners continue to believe that reuse is more trouble than it’s worth.  
  很多从业者仍认为复用“麻烦大于收益”。
- Many companies continue to encourage software development methodologies which do not facilitate reuse.  
  许多公司仍在鼓励不利于复用的软件开发方法。
- Few companies provide incentives to produce reusable program components.  
  很少有公司为生产可复用组件提供激励。

## The CBSE Process

1. Define the domain to be investigated.  
   定义要研究的领域。
2. Categorize the items extracted from the domain.  
   对从领域中提取的项目进行分类。
3. Collect a representative sample of applications in the domain.  
   收集该领域中的代表性应用样本。
4. Analyze each application in the sample.  
   分析样本中的每个应用。
5. Develop an analysis model for the objects.  
   为这些对象建立分析模型。

## Domain Engineering

- Is component functionality required on future implementations?  
  该组件功能是否会在未来实现中继续需要？
- How common is the component’s function within the domain?  
  该组件功能在领域内有多常见？
- Is there duplication of the component’s function within the domain?  
  领域内是否存在该功能的重复？
- Is the component hardware-dependent?  
  该组件是否依赖硬件？
- Does the hardware remain unchanged between implementations?  
  不同实现之间硬件是否保持不变？
- Can the hardware specifics be removed to another component?  
  硬件细节能否移到另一个组件中？
- Is the design optimized enough for the next implementation?  
  设计是否足够优化以支持下一次实现？
- Can we parameterize a non-reusable component so that it becomes reusable?  
  能否通过参数化让不可复用组件变得可复用？
- Is the component reusable in many implementations with only minor changes?  
  该组件是否能在多个实现中仅做少量修改就复用？
- Is reuse through modification feasible?  
  通过修改实现复用是否可行？
- Can a non-reusable component be decomposed to yield reusable components?  
  不可复用组件能否分解为可复用组件？
- How valid is component decomposition for reuse?  
  为复用而进行组件分解的有效性如何？

## Identifying Reusable Components

- a library of components must be available  
  必须存在组件库
- components should have a consistent structure  
  组件应有一致的结构
- a standard should exist, e.g., OMG/CORBA, Microsoft COM, Sun JavaBeans  
  应存在标准，例如 OMG/CORBA、Microsoft COM、Sun JavaBeans

## Component-Based SE

- Component qualification  
  组件鉴定
- Component adaptation  
  组件适配
- Component composition  
  组件组合
- Component update  
  组件更新

## CBSE Activities

application programming interface (API)  
应用程序编程接口（API）

development and integration tools required by the component  
组件所需的开发和集成工具

run-time requirements including resource usage (e.g., memory or storage), timing or speed, and network protocol  
运行时需求，包括资源使用（如内存或存储）、时序或速度、网络协议

service requirements including operating system interfaces and support from other components  
服务需求，包括操作系统接口和其他组件支持

security features including access controls and authentication protocol  
安全特性，包括访问控制和认证协议

embedded design assumptions including the use of specific numerical or non-numerical algorithms  
嵌入式设计假设，包括特定数值或非数值算法的使用

exception handling  
异常处理

## Qualification

The implication of easy integration is:

1. that consistent methods of resource management have been implemented for all components in the library;  
   组件库中的所有组件都实现了一致的资源管理方法；
2. that common activities such as data management exist for all components;  
   所有组件都具备诸如数据管理之类的公共活动；
3. that interfaces within the architecture and with the external environment have been implemented in a consistent manner.  
   架构内部以及与外部环境之间的接口都以一致方式实现。

## Adaptation

An infrastructure must be established to bind components together  
必须建立基础设施来把组件绑定在一起。

Architectural ingredients for composition include:

- Data exchange model  
  数据交换模型
- Automation  
  自动化
- Structured storage  
  结构化存储
- Underlying object model  
  底层对象模型

## Composition

The Object Management Group has published a common object request broker architecture (OMG/CORBA).  
Object Management Group 发布了通用对象请求代理体系结构（OMG/CORBA）。

An object request broker (ORB) provides services that enable reusable components (objects) to communicate with other components, regardless of their location within a system.  
ORB 提供服务，使可复用组件（对象）无论位于系统何处都能与其他组件通信。

Integration of CORBA components (without modification) within a system is assured if an interface definition language (IDL) interface is created for every component.  
如果为每个组件都创建 IDL 接口，则可保证 CORBA 组件无需修改即可集成到系统中。

Objects within the client application request one or more services from the ORB server. Requests are made via an IDL or dynamically at run time.  
客户端应用中的对象通过 ORB 服务器请求一个或多个服务，请求可通过 IDL 或运行时动态完成。

An interface repository contains all necessary information about the service’s request and response formats.  
接口仓库包含服务请求和响应格式所需的全部信息。

## OMG / CORBA

The component object model (COM) provides a specification for using components produced by various vendors within a single application running under the Windows operating system.  
组件对象模型（COM）提供了在 Windows 操作系统下将不同厂商生产的组件用于单一应用中的规范。

COM encompasses two elements:

- COM interfaces (implemented as COM objects)  
  COM 接口（以 COM 对象实现）
- a set of mechanisms for registering and passing messages between COM interfaces.  
  一组用于注册和在 COM 接口之间传递消息的机制

The JavaBeans component system is a portable, platform independent CBSE infrastructure developed using the Java programming language.  
JavaBeans 组件系统是一个可移植、平台无关的 CBSE 基础设施，由 Java 语言开发。

The JavaBeans component system encompasses a set of tools, called the Bean Development Kit (BDK), that allows developers to:

- analyze how existing Beans (components) work  
  分析现有 Bean（组件）的工作方式
- customize their behavior and appearance  
  定制其行为和外观
- establish mechanisms for coordination and communication  
  建立协调和通信机制
- develop custom Beans for use in a specific application  
  开发用于特定应用的自定义 Bean
- test and evaluate Bean behavior  
  测试并评估 Bean 行为

## Classification

- Enumerated classification：components are described by defining a hierarchical structure in which classes and varying levels of subclasses of software components are defined  
  枚举分类：通过定义层次结构来描述组件
- Faceted classification：a domain area is analyzed and a set of basic descriptive features are identified  
  面向特征的分类：分析领域并识别一组基本描述特征
- Attribute-value classification：a set of attributes are defined for all components in a domain area  
  属性-值分类：为领域中的所有组件定义一组属性

## Reuse Environment

A component database capable of storing software components and the classification information necessary to retrieve them.  
一个能够存储软件组件及其检索所需分类信息的组件数据库。

A library management system that provides access to the database.  
一个提供数据库访问的库管理系统。

A software component retrieval system (e.g., an object request broker) that enables a client application to retrieve components and services from the library server.  
一个软件组件检索系统（例如 object request broker），使客户端应用能够从库服务器检索组件和服务。

CBSE tools that support the integration of reused components into a new design or implementation.  
支持将复用组件集成到新设计或实现中的 CBSE 工具。
