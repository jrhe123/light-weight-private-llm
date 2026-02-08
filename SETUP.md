# 环境设置指南

每个章节使用独立的虚拟环境，完全隔离，避免版本冲突。

**重要**：使用 Python 3.10 创建虚拟环境。

---

## 设置新章节（首次设置）

以 Chapter 4 为例：

```bash
# 1. 进入章节目录
cd /Users/jiaronghe/Desktop/projects/light-weight-private-llm/chapter4

# 2. 使用 Python 3.10 创建虚拟环境
/Users/jiaronghe/.pyenv/versions/3.10.12/bin/python3.10 -m venv .venv
# 或者如果系统有 python3.10：
# python3.10 -m venv .venv

# 3. 激活虚拟环境
source .venv/bin/activate

# 4. 验证 Python 版本（应该是 3.10.x）
python --version

# 5. 安装依赖
pip install --upgrade pip
pip install -r requirements.txt

# 6. 安装并注册 Jupyter kernel
pip install ipykernel
python -m ipykernel install --user --name=chapter4 --display-name "Python (chapter4)"
```

**注意**：如果注册 kernel 后，kernel 配置文件指向了错误的 Python 路径，需要手动修复：

```bash
# 检查 kernel 配置
cat ~/Library/Jupyter/kernels/chapter4/kernel.json

# 如果路径不对，编辑文件，将 Python 路径改为：
# /Users/jiaronghe/Desktop/projects/light-weight-private-llm/chapter4/.venv/bin/python
```

---

## 在 Jupyter 中使用

### 1. 选择正确的 Kernel

在 Jupyter notebook 中：
- 点击右上角的 kernel 名称
- 选择对应的 kernel（如 "Python (chapter4)"）

### 2. 验证环境

在 notebook 中运行：

```python
import sys
print("Python 版本:", sys.version.split()[0])
print("Python 路径:", sys.executable)
```

应该显示对应章节的 `.venv` 路径。

---

## 日常切换章节

### 切换到 Chapter 3

```bash
cd /Users/jiaronghe/Desktop/projects/light-weight-private-llm/chapter3
source .venv/bin/activate
```

### 切换到 Chapter 4

```bash
cd /Users/jiaronghe/Desktop/projects/light-weight-private-llm/chapter4
source .venv/bin/activate
```

然后在 Jupyter 中选择对应的 kernel。

---

## Kernel 配置文件位置

**用户级别的 kernel 配置**：
```
~/Library/Jupyter/kernels/chapter4/kernel.json
~/Library/Jupyter/kernels/chapter3/kernel.json
```

**查看所有已注册的 kernel**：
```bash
jupyter kernelspec list
```

**查看特定 kernel 配置**：
```bash
cat ~/Library/Jupyter/kernels/chapter4/kernel.json
```

## 手动选择 Python 解释器
```
0. update .vscode/settings.json
1. 按 Cmd+Shift+P
2. 输入并选择：Python: Select Interpreter
3. 选择：./chapter4/.venv/bin/python
```

---

## 总结

- 每个章节有独立的 `.venv` 文件夹和 `requirements.txt`
- 使用 Python 3.10 创建虚拟环境
- 每个虚拟环境需要注册 Jupyter kernel
- 在 Jupyter 中选择对应的 kernel 才能使用正确的环境
