# 示例及复现

本目录保留已有运行的 `preview.png`、`config_used.json` 和 `metadata.json`。表中的孔隙率来自随附元数据，并非根据图片像素或展示网格体积推算。

| 示例 | 目标孔隙率 | 控制 Sobol 估计 | 独立 Sobol 估计 |
| --- | --- | --- | --- |
| `sheet_p95` | 95% | 95.0000% | 94.9903% |
| `nanowire_p95` | 95% | 94.9997% | 94.9961% |
| `nanowire_p97` | 97% | 96.9994% | 96.9957% |
| `nanowire_p99` | 99% | 98.9994% | 98.9866% |

片层模型为 1000 nm 胞元、5 nm 厚度；纳米线三例均为 1000 nm 胞元、400 个种子、20 nm 直径、随机种子 42。纳米线图片为纯白背景，不含坐标轴和尺寸文字，原图保留 600 dpi 导出设置。

从仓库根目录复现结构：

```bash
python -m voronoi_aerogel.cli --config examples/sheet_p95/config_used.json --output output/reproduce_sheet_p95
python -m voronoi_nanowire.cli --config examples/nanowire_p95/config_used.json --output output/reproduce_nanowire_p95
python -m voronoi_nanowire.cli --config examples/nanowire_p97/config_used.json --output output/reproduce_nanowire_p97
python -m voronoi_nanowire.cli --config examples/nanowire_p99/config_used.json --output output/reproduce_nanowire_p99
```

示例来自项目整理前的既有运行。配置保留原运行的相对输出目录；以上命令以 `--output` 为新结果指定独立目录。选中结构的重复性还取决于数值库版本；渲染字体、库版本和绘图设置也可能使像素不同。

为控制仓库体积，本目录仅收录预览和参数记录。元数据可能列出该次运行还生成过的 STL、HTML、BDF 等文件，这些大文件不在本目录中；请运行代码重新生成。
