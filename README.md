# group5-canny-cutile
Canny Edge Detection implementation with cuTile for SoftEng 751 2026


## 项目分工
- jiaxi liu:  cuTile GPU 实现 + 实时视频
- xudong ma :纯 Python Baseline + 性能测试 + 报告素材
- shiying yang: 依赖分析、参数优化、集成

## 当前进度 (2026.4)
- ✅ 纯 Python Canny Baseline 已完成（包含所有步骤 + 中间结果可视化 + Benchmark）
- 测试图片位于 `data/` 文件夹
- 中间结果保存在 `report/` 文件夹

## 如何运行 Baseline

```bash
# 1. 激活环境
.venv\Scripts\activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 运行 Jupyter Notebook
jupyter notebook