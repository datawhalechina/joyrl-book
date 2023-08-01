安装

```bash
# 安装node(Mac)
brew install node
# windows，安装后需要重启
https://nodejs.org/en/
# 全局安装docsify-cli，没有代理加速可以安装cnpm镜像加速
npm i docsify-cli -g
# 初始化，会在docs文件夹下生成README，index.html文件
docsify init ./docs
```

本地部署，预览网站就在http://localhost:3000网址打开

```bash
cd docs
docsify serve
## 或者
docsify serve ./docs
```

配置侧边栏

index.html文件`window.$docsify`中增加`loadSidebar: true`，然后在docs下新建`_sidebar.md`

```html
window.$docsify = {
      name: '',
      repo: '',
      loadSidebar: true,
    }
```

latex 公式显示问题

https://github.com/scruel/docsify-latex

## 编写规范

1. 不能在代码块里面加入`\`，否则会导致后面的公式不渲染
2. 英文和文中的数字用`$\text{}$`
3. 公式编号用`\tag{}`，公式和图表引用要在实际的公式和图表之前，便于阅读
### 参考

[docsify中文文档](https://jingping-ye.github.io/docsify-docs-zh/#/%E5%BF%AB%E9%80%9F%E4%B8%8A%E6%89%8B/%E5%BC%80%E5%A7%8B)