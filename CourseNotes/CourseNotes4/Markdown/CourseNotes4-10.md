# Chapter 17 WebApp Design

## Why WebApp Design Matters

WebApp design becomes important when：

- content and function are complex
- the site contains hundreds of content objects, functions, and analysis classes
- the success of the WebApp directly affects business success

设计重点不是“好看”这么简单，而是要让复杂内容、功能和导航都能被用户顺利理解和使用。

### Design vs. WebApps

WebApp 设计通常要同时考虑：

- business purpose
- usability
- content structure
- navigation
- performance
- security

## WebApp Quality Concerns

### Security

- Rebuff external attacks
- Exclude unauthorized access
- Ensure user privacy

### Availability

WebApp 在需要的时候是否可用，是否能稳定提供服务。

### Scalability

系统能不能承受用户量、事务量的明显变化。

### Time to Market

上线速度也很重要，尤其是业务型 WebApp。

## Quality Dimensions for End-Users

### Time

- 页面更新后改了什么？
- 如何让用户看出变化？

### Structural

- 网站内部和外部链接是否都有效？
- 图片是否正常？
- 页面之间是否形成完整连接？

### Content

- 关键页面内容是否正确？
- 高频更新页面是否保持关键短语？
- 动态页面是否也保持内容质量？

### Accuracy and Consistency

- 今天下载的页面和昨天是否一致？
- 页面数据是否足够准确？

### Response Time and Latency

- 服务器响应是否足够快？
- 提交后端到端响应是否可接受？
- 是否存在慢到让用户放弃的部分？

### Performance

- 浏览器、Web、站点之间的连接是否足够快？
- 性能是否会随时间、负载变化？

### Consistency

WebApp 的内容、图形风格、架构模板、交互模式、导航机制都应该保持一致。

## WebApp Design Goals

- **Identity**：建立符合业务目的的形象
- **Robustness**：提供稳定、相关的内容和功能
- **Navigability**：导航要直观、可预测
- **Visual appeal**：视觉呈现要有吸引力
- **Compatibility**：兼容目标环境和配置

## WebApp Design Pyramid

WebApp design 可以看成一个分层目标体系：

1. 先解决用户定位问题
2. 再解决当前可做什么
3. 再解决如何从当前位置继续移动

### Three Core Questions

- **Where am I?**：我在哪
- **What can I do now?**：我现在能做什么
- **Where have I been, where am I going?**：我去过哪、能去哪

这三点基本上就是 WebApp 导航体验的核心。

## WebApp Interface Design

WebApp interface 设计要关注：

- **Anticipation**：预判用户下一步
- **Communication**：告诉用户当前状态
- **Controlled autonomy**：允许用户自由移动，但仍遵守站点约定
- **Efficiency**：优化用户效率，而不是开发者效率

## Interface Design Principles

### I

- **Focus**：界面始终围绕当前任务
- **Fitt's Law**：目标越远、越小，获取越慢
- **Human interface objects**：复用成熟的人机界面对象
- **Latency reduction**：让用户感觉操作已经完成
- **Learnability**：易学、也要易复学

### II

- **Maintain work product integrity**：自动保存，避免内容丢失
- **Readability**：信息要清晰可读
- **Track state**：能记录用户状态，方便下次继续
- **Visible navigation**：让用户感觉自己始终在同一个工作空间里

### III

- 留白不要怕
- 突出内容
- 布局尽量符合左上到右下的阅读习惯
- 导航、内容、功能要按区域分组
- 不要滥用滚动条
- 设计时要考虑分辨率和窗口大小

## Content Design

Content design 关注内容对象本身的设计方式。

- content object 更接近传统软件中的 data object
- 需要设计对象之间的关系和实例化机制
- 内容对象既有内容属性，也有实现属性

## Content Architecture

Content architecture 关注内容如何组织、展示和导航。

信息架构（information architecture）也常用来指：

- 更好的组织
- 标注
- 导航
- 搜索

## Architecture Design

WebApp architecture 关注应用整体如何组织来处理：

- 用户交互
- 内部处理
- 导航
- 内容呈现

Architecture design 会与 interface design、aesthetic design、content design 并行进行。

## MVC Architecture

- **Model**：应用相关内容和处理逻辑
- **View**：界面展示与接口相关功能
- **Controller**：协调 model 和 view 的数据流

## Navigation Design

Navigation design 从用户层次和 use-case 出发。

每个用户在系统中的导航需求可能不同，因此 WebApp 要用 **Navigation Semantic Units (NSUs)** 来组织导航结构。

### Navigation Semantic Unit

NSU 可以理解成一组：

- 相关信息
- 相关导航结构
- 完成某类用户需求所需的协作单元

### Ways of Navigation (WoN)

WoN 表示针对特定用户画像、最适合的导航路径。

它通常由：

- Navigation nodes
- Navigation links

组成。

## Navigation Syntax

常见导航语法包括：

- text-based links
- icons
- buttons and switches
- graphical metaphors

还有一些典型导航结构：

- **Horizontal navigation bar**：横向导航栏
- **Vertical navigation column**：纵向导航栏
- **Tabs**：标签式导航
- **Site maps**：站点地图

## Component-Level Design

WebApp components 需要提供：

- 动态生成内容和导航
- 领域相关的数据处理能力
- 数据库查询和访问
- 与外部系统的数据接口

这一层关注的是：**组件如何支撑 WebApp 的整体行为和导航体验。**
