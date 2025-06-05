# HelloWorld

个人用来学习相关模块的库，主要使用JupyterNotebook脚本来编写，方便代码块执行，保存并记录相关输出

## 内容

目前里面相关库有Matplotlib、numpy、scipy、PyBullet、MuJoCo、pytransform3d等库，另外有movement_primitives等相关Github公开项目

## 安装

此项目并不特别需要相关库依赖，建议使用解析器使用python3.9，推荐使用conda建立相关虚拟环境，安装jupyter notebook和对应所需库即可

1. Git

    ```bash
    git clone https://github.com/ShimonWang/HelloWorld.git
    ```

2. 导航到仓库根目录
    
    ```bash
    cd HelloWorld
    ```
   
3. conda建立环境

    ```bash
    conda create --name hello39 python=3.9
    ```
   
    ```bash
    conda activate hello39
    ```

4. 安装JupterNotebook

    ```bash
    pip install jupyternotebook
    ```

5. 启动脚本

    ```bash
    jupyter notebook
    ``` 
