# MCP环境配置完整指南

## 📋 问题诊断
当前报错：`[WinError 2] 系统找不到指定的文件。`
原因：系统找不到`uvx`命令，因为相关路径未添加到环境变量中。

## 🔧 解决方案

### 方案一：添加环境变量（推荐用于开发）

#### 步骤1：确定用户目录路径
您的用户目录路径应该是：
```
C:\Users\17321\.local\bin
```

#### 步骤2：添加到系统PATH环境变量

**方法A：通过系统设置**
1. 按 `Win + R` 打开运行对话框
2. 输入 `sysdm.cpl` 按回车
3. 点击"环境变量"按钮
4. 在"用户变量"部分找到"Path"，双击编辑
5. 点击"新建"，添加路径：`C:\Users\17321\.local\bin`
6. 点击"确定"保存所有更改
7. 重新打开命令行窗口测试

**方法B：通过PowerShell命令**
```powershell
# 查看当前PATH
$env:PATH -split ';'

# 添加新路径到用户环境变量
[Environment]::SetEnvironmentVariable("PATH", $env:PATH + ";C:\Users\17321\.local\bin", "User")

# 重新加载环境变量（重启PowerShell后生效）
```

#### 步骤3：验证配置
重新打开PowerShell，测试命令：
```powershell
# 检查uvx是否可用
uvx --version

# 检查uv是否可用  
uv --version
```

### 方案二：临时解决方案（快速修复）

如果您暂时不想修改环境变量，可以禁用MCP功能：

#### 修改配置文件
编辑 `voice_llm/llm_worker/config.py`：
```python
'juhe_mcp': {
    'enabled': False,  # 添加这一行，禁用MCP
    'command': 'uvx',
    'args': [...],
    ...
}
```

#### 或者修改enhanced_mcp_client.py
```python
def _check_mcp_availability(self) -> bool:
    """检查MCP服务是否可用 - 临时禁用版本"""
    # 临时禁用MCP检测，避免警告
    self.mcp_available = False
    
    if not hasattr(self, '_mcp_disabled_logged'):
        logger.info("📡 MCP功能已临时禁用，使用直接API调用")
        self._mcp_disabled_logged = True
    
    return False
```

## 🚀 完整MCP安装流程

如果您想完整配置MCP环境，请按以下步骤操作：

### 1. 安装uv包管理器
```powershell
# 安装uv
pip install uv

# 验证安装
uv --version
```

### 2. 创建MCP项目
```powershell
# 初始化MCP服务器项目
uv init mcp-server
cd mcp-server

# 添加MCP依赖
uv add "mcp[cli]"
```

### 3. 安装额外依赖
```powershell
pip install "mcp[cli]"
pip install aiohttp jsonschema pyyaml
```

### 4. 配置环境变量
按照上述"方案一"的步骤添加环境变量。

### 5. 测试MCP功能
```powershell
# 测试MCP代理
uvx juhe-mcp-proxy --help
```

## 💡 推荐做法

### 对于生产环境
- 建议使用**方案二**（禁用MCP），因为直接API调用更稳定
- MCP功能目前还在发展阶段，可能存在稳定性问题

### 对于开发环境
- 可以尝试**方案一**（配置完整MCP环境）
- 有助于测试和开发MCP相关功能

## 🔍 故障排除

### 常见问题1：uvx命令不存在
**解决方案**：
1. 检查uv是否正确安装：`uv --version`
2. 检查PATH环境变量是否包含`.local\bin`路径
3. 重启PowerShell/命令行

### 常见问题2：MCP连接失败
**解决方案**：
1. 检查网络连接
2. 验证API token是否有效
3. 检查防火墙设置

### 常见问题3：权限问题
**解决方案**：
1. 以管理员身份运行PowerShell
2. 检查用户权限设置

## 🎯 快速修复当前问题

如果您只是想让系统停止报warning，最简单的方法是：

1. 打开 `enhanced_mcp_client.py`
2. 找到 `_check_mcp_availability` 方法
3. 在方法开头添加：
   ```python
   # 临时禁用MCP检测
   self.mcp_available = False
   return False
   ```

这样可以立即解决警告问题，同时不影响系统的其他功能。