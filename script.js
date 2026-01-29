// 这是一个简单的 JavaScript 示例
// 它会在页面加载完成后运行

document.addEventListener('DOMContentLoaded', () => {
    console.log('欢迎！脚本已成功加载。');

    // 下面我们演示如何用 JavaScript 操作网页内容
    // 我们的目标是：添加一个按钮，点击后可以切换"深色模式"

    // 1. 获取要放置按钮的位置（这里选择 header 区域）
    const header = document.querySelector('.header');

    if (header) {
        // 2. 创建一个新的按钮元素
        const themeBtn = document.createElement('button');
        
        // 3. 设置按钮的文字和样式
        themeBtn.textContent = '切换深色模式';
        themeBtn.style.cssText = `
            margin-top: 15px;
            padding: 8px 16px;
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        `;
        
        // 4. 添加点击事件监听器
        themeBtn.addEventListener('click', () => {
            // 在 body 标签上切换 'dark-mode' 类
            document.body.classList.toggle('dark-mode');
            
            // 更新按钮文字
            if (document.body.classList.contains('dark-mode')) {
                themeBtn.textContent = '切换回浅色模式';
                themeBtn.style.backgroundColor = '#e74c3c'; // 换个颜色
            } else {
                themeBtn.textContent = '切换深色模式';
                themeBtn.style.backgroundColor = '#3498db';
            }
        });

        // 5. 将按钮添加到页面中
        header.appendChild(themeBtn);
    }
});
