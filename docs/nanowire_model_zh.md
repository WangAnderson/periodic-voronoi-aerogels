# 纳米线气凝胶 Voronoi 模型使用说明

本说明针对项目中的**纳米线模型** `voronoi_nanowire`。它在三维周期 Voronoi 剖分的边上构造有限直径的纳米棒，通过选择部分完整的 Voronoi 边控制孔隙率。这是可复现的随机几何模型；仅由孔隙率和尺寸参数生成结构，不代表已经从实验图像反演或验证了真实材料的形貌。项目中另有片层模型 `voronoi_aerogel`，两套程序的入口和配置文件不同。

## 1. 应该打开哪个文件

将仓库克隆或下载解压后，在编辑器中打开仓库根目录 `periodic-voronoi-aerogels`（包含 `pyproject.toml`、`voronoi_aerogel` 和 `voronoi_nanowire` 的文件夹），然后按目的选择文件。下表中的路径均相对于仓库根目录：

| 目的 | 文件 | 是否需要修改 |
| --- | --- | --- |
| 在 Spyder 里直接改参数并按运行按钮 | `run_nanowire_spyder.py` | 修改这里的 `NanowireConfig(...)` 和 `STEP` |
| 通过终端运行时修改默认模型的尺寸、孔隙率等 | `nanowire_config.json` | 修改这里 |
| 分别生成已有的 95%、97%、99% 示例 | `nanowire_configs/porosity_95.json`、`porosity_97.json`、`porosity_99.json` | 原样运行即可；建议复制一份再试新参数 |
| 查看命令行入口 | `voronoi_nanowire/cli.py` | 一般不修改 |
| 查看建模与孔隙率验证流程 | `voronoi_nanowire/pipeline.py` | 一般不修改 |
| 查看 PNG、HTML 和显示网格的生成 | `voronoi_nanowire/exporters.py` | 一般不修改 |
| 查看参数默认值和合法范围 | `voronoi_nanowire/config.py` | 一般不修改 |

运行程序有两种方式：在 Spyder 中打开并直接运行 `run_nanowire_spyder.py`，或者在项目根目录的终端执行 `python -m voronoi_nanowire.cli ...`。不要在编辑器中直接运行 `cli.py`：它使用包内相对导入，直接运行单个文件可能出现导入错误。终端方式主要编辑 **JSON 配置文件**；Spyder 方式直接编辑 Python 脚本中的参数。

## 2. 在 Spyder 中直接修改和运行

1. 用 Spyder 打开仓库根目录中的 `run_nanowire_spyder.py`。
2. 在 `main()` 内的 `NanowireConfig(...)` 修改参数。例如将 `target_porosity=0.95` 改成 `0.97`，将 `rod_diameter_nm=20.0` 改成需要的直径，`cube_size_nm=1000.0` 改成需要的胞元边长。长度单位都是 nm；`0.97` 表示 97% 孔隙率。每组新参数请给 `output_dir` 的末尾文件夹起一个新名字。
3. **第一步，生成结构演示图：** 保持 `STEP = "image"`，保存文件后点击 Spyder 的“运行文件”按钮或按 `F5` 运行**整个文件**。程序生成 `preview.png`、可旋转的 `nanowire_preview.html`，并保存本次模型的 `display_mesh.npz` 网格缓存。
4. **第二步，输出同一模型的 STL：** 只把 `STEP = "image"` 改成 `STEP = "stl"`，保留其他参数和 `output_dir`，再次保存并按 `F5`。程序直接读取第一步的网格缓存，写出 `nanowire_display_nm.stl`，不会重新抽取随机骨架或重画图片。

该脚本默认把输出放在项目内的 `output\spyder_nanowire`，即使 Spyder 当前工作目录设在别处，也会使用脚本所在项目作为路径基准。它直接构造 `NanowireConfig`，因此运行这个脚本时，`nanowire_config.json` 的修改不会自动生效。不要只选中一段代码运行；应运行整个文件。第二步会核对当前参数与第一步保存的配置及网格；如果中途修改孔隙率、尺寸或其他参数，需先重新运行第一步。若重复使用旧输出文件夹，之前的 STL 可能仍在其中，应以第二步完成后的新 STL 为准。Spyder 的“运行配置”可以设置工作目录，但此脚本已经采用脚本所在目录定位项目。[Spyder 官方编辑器文档](https://docs.spyder-ide.org/current/panes/editor.html)说明了“运行文件”按钮、`F5` 和逐文件运行配置。

Spyder 使用的 Python 环境必须安装本项目依赖。可在 Spyder 的 IPython 控制台运行 `import sys; print(sys.executable)` 查看实际解释器路径，再把 `requirements.txt` 中的依赖安装到该解释器对应的环境；否则即使系统终端里能运行，也可能在 Spyder 中报缺少模块。[Spyder 官方常见问题](https://docs.spyder-ide.org/current/faq.html)也说明了不同环境中运行代码和安装依赖的注意事项。

下面的终端与 JSON 用法适合批量生成多个模型，或不使用 Spyder 时参考。

## 3. 准备 Python 环境

需要 Python 3.11 或更新版本。打开 PowerShell 或编辑器内置终端，执行：

```powershell
# 在仓库根目录执行以下命令
python --version
python -m pip install -r requirements.txt
```

主要依赖是 NumPy、SciPy、scikit-image、Matplotlib 和 Plotly。首次安装需要能够访问 Python 软件包源；若已安装依赖，可以跳过安装命令。以下所有命令都假定终端当前目录是仓库根目录。项目路径可自行选择，不依赖开发者的磁盘或用户名。

## 4. 修改 JSON 模型参数

打开 `nanowire_config.json`。当前默认配置的关键部分如下：

```json
{
  "cube_size_nm": 1000.0,
  "num_seeds": 400,
  "seed_min_distance_factor": 0.3,
  "rod_diameter_nm": 20.0,
  "target_porosity": 0.95,
  "porosity_tolerance": 0.005,
  "random_seed": 42,
  "qmc_power": 18,
  "validation_qmc_power": 19,
  "tube_sides": 12,
  "output_dir": "output/nanowire_default"
}
```

| 参数 | 含义与修改示例 |
| --- | --- |
| `cube_size_nm` | 立方体胞元的边长，单位 **nm**。`1000.0` 表示 1000 nm，即 1 μm。 |
| `num_seeds` | 单胞中的 Voronoi 种子数。改变它会改变剖分和骨架；不等于最终纳米棒数量。 |
| `seed_min_distance_factor` | 随机种子的最小间距控制因子；保持 `0.3` 可复现已有示例。 |
| `rod_diameter_nm` | 纳米棒直径，单位 **nm**。例如 `20.0` 表示直径 20 nm，程序自动取半径 10 nm。 |
| `target_porosity` | 目标孔隙率，填写 **0 到 1 之间的小数**：`0.95`、`0.97`、`0.99` 分别表示 95%、97%、99%。不要填写 `95`。 |
| `porosity_tolerance` | 孔隙率允许的绝对误差。`0.005` 是 0.5 个百分点，不是相对误差 0.5%。 |
| `random_seed` | 随机种子；相同参数和种子可重现同一个随机结构。 |
| `qmc_power` | 用于选择纳米棒的 Sobol 采样规模指数，`18` 表示 `2^18` 个采样点。提高它通常更慢。 |
| `validation_qmc_power` | 用另一组点独立核验孔隙率的规模指数，`19` 表示 `2^19` 个点。 |
| `tube_sides` | 图片和 STL 中纳米棒圆周的多边形边数。当前为 `12`；增加会使表面更圆滑，也会增大文件和渲染开销。它不改变孔隙率计算中的物理直径。 |
| `output_dir` | 输出文件夹，相对于运行命令时的当前目录。请为不同参数使用不同输出文件夹，方便保留各次结果。 |

单根纳米棒没有独立的“长度”参数；长度由 Voronoi 剖分边决定。胞元边长、种子数及随机种子都会影响边的长度和分布。

例如要生成边长 800 nm、棒直径 16 nm、孔隙率 97% 的新模型，可复制一份配置文件并把 `cube_size_nm`、`rod_diameter_nm`、`target_porosity` 分别改为 `800.0`、`16.0`、`0.97`，同时将 `output_dir` 改为新的文件夹。JSON 中数字后不能写 `nm` 或 `%`，各字段之间要有英文逗号，最后一项后不要加逗号。

更改胞元尺寸、种子数或棒直径后，原来的选边和可达到的孔隙率范围可能变化。程序计算的是当前选定贯通骨架和连接生长策略下的可达到范围；失败不表示所有其他选边策略都不可能达到该目标。程序不会为了凑孔隙率而改变指定的棒直径。

## 5. 运行并生成模型演示图片

使用默认配置：

```powershell
# 在仓库根目录执行以下命令
python -m voronoi_nanowire.cli --config nanowire_config.json
```

运行成功后，默认输出目录为 `output\nanowire_default`。其中：

| 文件 | 用途 |
| --- | --- |
| `preview.png` | 600 dpi 的高清静态演示图，纯白背景，仅显示纳米棒和胞元边框；无坐标轴、标题及尺寸文字。 |
| `nanowire_preview.html` | 可在浏览器中旋转、缩放的交互式展示文件；其标题和标注样式与静态 PNG 不完全相同。 |
| `config_used.json` | 本次实际使用的完整参数，便于复现。 |
| `metadata.json` | 候选/选中边数、控制与独立核验孔隙率、拓扑等信息。 |

直接生成已有的三组示例：

```powershell
python -m voronoi_nanowire.cli --config nanowire_configs\porosity_95.json --output output\my_p95
python -m voronoi_nanowire.cli --config nanowire_configs\porosity_97.json --output output\my_p97
python -m voronoi_nanowire.cli --config nanowire_configs\porosity_99.json --output output\my_p99
```

命令中的 `--output` 临时覆盖配置文件里的 `output_dir`，不会改写原配置。若只想临时改变孔隙率，也可使用 `--porosity`：

```powershell
python -m voronoi_nanowire.cli --config nanowire_config.json --porosity 0.97 --output output\trial_p97
```

`--seed 123` 可临时指定新的随机种子。`--quick` 会降低采样量并减少棒的圆周分段，仅用于较快检查流程或构图；正式结果请去掉此选项。

## 6. 输出 STL 模型文件

在运行命令后加 `--stl`：

```powershell
python -m voronoi_nanowire.cli --config nanowire_configs\porosity_97.json --output output\my_p97_stl --stl
```

程序会同时生成 PNG、HTML 和 `output\my_p97_stl\nanowire_display_nm.stl`。这是**二进制 STL 展示网格**，纳米棒直径和选中骨架与同一次运行的图片一致。胞元灰色线框只是一种图片辅助标记，不写入 STL。`metadata.json` 中的 `files.display_stl` 会记录 STL 文件名。

**单位：** STL 格式本身不保存单位；此文件的数值坐标约定为 **nm**。导入软件若以 μm 为单位，应把坐标缩小 `0.001` 倍；若以 mm 为单位，缩小 `0.000001` 倍；若以 m 为单位，缩小 `1e-9` 倍。导入后先检查模型外形尺寸是否约为配置中的 `cube_size_nm`。

**用途限制：** 这个 STL 与 PNG 共用圆柱和节点球组成的显示网格。相交纳米棒没有做布尔并集，节点球与棒之间也可能存在内部重叠面；它不保证水密、单一实体或严格裁剪到胞元边界。因此适合在三维查看器中展示和检查形态，不能直接把其三角面包围体积当作已验证的实体体积，也不能未经修复就用作打印或要求封闭实体的有限元网格。`metadata.json` 中的孔隙率是对有限直径纳米棒**并集**的独立采样估计，不是由此 STL 的三角面体积计算出的值。

若后续需要可打印或可用于体网格划分的水密纳米棒 STL，需要另做纳米棒实体并集、胞元裁剪和网格质量检查；目前的 `--stl` 不声称完成这些步骤。

## 7. 怎么看计算结果和排查常见问题

终端会显示选边状态、控制孔隙率和独立核验孔隙率。例如 `porosity=96.9994%` 表示选边时计算的值，`Independent porosity: 96.9957%` 表示使用另一组采样点得到的值。以 `metadata.json` 中的 `porosity.target`、`porosity.control_qmc`、`porosity.independent_qmc` 核对目标和实际结果，以 `geometry.selected_edge_count` 查看保留下来的纳米棒边数。

- 提示 `No module named voronoi_nanowire`：确认终端已进入仓库根目录，并使用 `python -m voronoi_nanowire.cli`。
- 提示缺少 `numpy`、`scipy`、`matplotlib` 等模块：在实际运行程序的同一个 Python 环境执行 `python -m pip install -r requirements.txt`。
- 提示目标不可达或独立核验失败：查看输出目录中的 `feasibility.json`。可调整孔隙率、棒直径或种子数，再以新目录重新运行。该报告针对当前骨架和生长策略，不是所有可能结构的全局不可行证明；不要把失败时未完成的文件当成最终模型。
- 图片生成较慢或文件较大：600 dpi 图片会完整绘制所有纳米棒三角面；95% 模型的杆更多，通常比 99% 模型耗时更长。

## 8. 建模过程简述

程序首先在立方体中布置周期性随机种子并建立三维 Voronoi 剖分，提取共享面的交线作为候选纳米棒中心线。随后先保留连通且沿三个方向贯通的骨架，再按固定随机种子增加完整的 Voronoi 边。对于指定直径，程序把纳米棒按圆柱加圆端帽的胶囊体处理，用 Sobol 采样估计这些胶囊体的并集占胞元的体积分数，以 `1 − 实体体积分数` 得到孔隙率，并使用独立采样再次核验。最后把选中边转成有尺寸的圆柱和真实节点球，输出模型演示图及可选的 STL 展示网格。
