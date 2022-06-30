# Ubuntu20.04安装LaTeX并配置中文环境


# 简介
Latex是一个基于TEX的排班系统。粘贴复制的话就不搞了。简而言之，通俗来说，就是用来写文章的，而且完全不需要你去花时间去自己考虑排版、页码等没必要花时间去弄的玩意。而且，对于数学公式的输入也是极为方便，这点在后面会展示。

---  
  
<br />
<br />

      
        
# 安装

1. **安装发行版** `sudo apt-get install texlive-full`  
1. **安装XeLaTeX编译引擎**`sudo apt-get install texlive-xetex`
1. **安装中文支持包（如果需要）**`sudo apt-get install texlive-lang-chinese`
1. **安装编辑器**（图形化界面的选择有很多，例如 TeXStudio，TeXmaker等，可以看做是一个编辑器，这里安装的是TexStudio）`sudo apt-get install texstudio`
1. **配置**
    - 设置编译器为XeLaTeX，TeXstudio中在Options->Configure TeXstudio->Build->Default Compiler中更改默认编译器为XeLaTeX 
    - 更改软件界面语言，将Options->Configure TeXstudio->General->Language改为zh-CN即可将界面设置为中文


<br />
<br />
<br />

# 其他工具和资源
### Mathpix

在写论文或者文档的时候，经常会碰到要输入数学公式的情况。如果去手敲，效率很低。这里介绍一个数学公式的latex语句生成工具 [Mathpix](https://standard-robots.yuque.com/dvszm5/khqrwm/rwgsu7)。 碰到文献里的数学公式，只需要用这个软件，就可以轻松获得这个公式的latex语句。

<br />

### 模板
模板这里分享一个latex模板网站  [模板](https://www.latextemplates.com/)




