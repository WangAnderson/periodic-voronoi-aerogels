# 贡献与问题反馈

欢迎通过 GitHub Issues 反馈问题，通过 Pull Request 提交修改。

请在问题中提供 Python 与依赖版本、运行命令、最小配置、完整错误信息，以及相关 `metadata.json` 或 `feasibility.json`。上传前移除个人路径和其他私人信息。不要上传无关的大型 STL 或整份工作目录。

修改几何、周期拓扑、孔隙率估计或网格导出时，请说明数值影响，并运行：

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

请保留两类模型的入口及输出单位约定。涉及求解器或制造用途的改动应说明验证范围，不将展示网格声明为未经检查的封闭实体。提交贡献即同意将其按本项目 MIT 许可证发布。
