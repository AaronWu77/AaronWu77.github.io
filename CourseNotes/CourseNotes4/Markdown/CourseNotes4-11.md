# Chapter 18 MobileApp Design

## Mobile Development Considerations

移动应用开发比 WebApp 更受平台和设备限制，常见问题包括：

- 多种硬件和软件平台
- 多种框架和语言
- 各种 app store 的审核规则不同
- 开发周期短
- UI 受限

另外还有一些移动端特有难点：

- 摄像头 / 传感器交互复杂
- 上下文使用很重要
- 电源管理
- 安全与隐私策略
- 设备算力和存储限制
- 外部服务集成
- 测试复杂

## MobileApp Development Process Model

移动应用开发流程通常是：

1. Formulation
2. Planning
3. Analysis
4. Engineering
5. Implementation and testing
6. User evaluation

这个流程的重点是：先定义问题，再逐步落地，而不是一开始就直接写界面。

## MobileApp Quality Checklist

### 1

- 内容 / 功能 / 导航是否能按用户偏好定制？
- 是否能适应不同带宽和弱网情况？
- 是否能根据上下文做调整？
- 是否考虑了目标设备的电量？
- 是否合理使用图像、音频、视频和云服务？

### 2

- 页面是否容易阅读和导航？
- 是否考虑不同屏幕尺寸？
- UI 是否符合目标设备的交互标准？
- 是否符合用户对可靠性、安全性、隐私的预期？
- 是否保证应用能持续更新？
- 是否在所有目标设备和环境中测试过？

## MobileApp UI Design Considerations

移动端 UI 设计需要特别关注：

- 定义品牌特征
- 聚焦产品组合
- 提炼核心 user stories
- 优化 UI flows 和 elements
- 定义缩放规则
- 建立用户性能面板
- 让专门的 UI 工程能力参与进来

## MobileApp Design Mistakes

常见错误有：

- **Kitchen sink**：什么都往里塞
- **Inconsistency**：不一致
- **Overdesigning**：过度设计
- **Lack of speed**：速度差
- **Verbiage**：文字太多
- **Non-standard interaction**：非标准交互
- **Help-and FAQ-itis**：过度依赖帮助和 FAQ

## MobileApp Design Best Practices

- 先明确受众
- 结合使用场景设计
- 简洁和偷懒之间要分清
- 利用平台优势
- 让高级功能仍然可发现
- 标签要清晰一致
- 不要为了“聪明图标”牺牲理解成本
- 长滚动表单通常比多屏切换更好

## Assessing Mobile Interactive Development Environments

评估移动开发环境时，可以看：

- 通用生产力功能
- 第三方 SDK 集成
- 编译后工具
- 空中更新支持
- 端到端移动开发能力
- 文档和教程
- 图形化界面构建器

## MobileApp Middleware

Middleware 的作用包括：

- 协调分布式组件通信
- 隐藏移动环境细节
- 支持上下文感知

它的价值就是：**让开发者少处理底层差异，把精力放在业务和体验上。**
