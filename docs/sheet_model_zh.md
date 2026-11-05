# 氧化铝气凝胶片层 Voronoi 模型：中文使用说明

本页说明片层命令行 / Python API。图形界面操作和 Windows 下载请见 [Aerogel Studio 桌面软件说明](desktop_studio_zh.md)。

本程序在立方体中生成三维周期 Voronoi 片层网络，供气凝胶微结构展示、几何参数研究和后续数值模拟使用。它是基于给定尺度与孔隙率的随机几何模型，不包含由显微图像反演结构、材料烧结过程或材料参数标定。

片层模型代码位于 `voronoi_aerogel/`，与 `voronoi_nanowire/` 中的纳米线模型分别运行。本文所有路径均相对于下载后的仓库根目录 `periodic-voronoi-aerogels/`，所有命令均在该目录执行。

## 1. 安装与首次运行

需要 Python 3.11 或以上版本。建议为本项目建立独立环境；已有环境也可使用，但请确认安装依赖和运行程序使用的是同一个 Python。

在仓库根目录运行：

```bash
python -m pip install -e .
```

此命令安装本地项目和 NumPy、SciPy、scikit-image、Matplotlib、Plotly 等运行依赖。生成模型本身不需要安装 COMSOL 或 Lumerical。首次安装依赖通常需要能够访问 Python 软件包源。

先运行快速检查：

```bash
python -m voronoi_aerogel.cli --config default_config.json --quick --output output/sheet_quick
```

快速模式仍执行几何、选面、体素化和导出流程，只降低采样量并使用较粗体素。它适合检查安装、参数可行性和输出流程，不能替代正式网格收敛研究。

正式运行：

```bash
python -m voronoi_aerogel.cli --config default_config.json --output output/sheet_full
```

安装本项目后，也可用等价的命令行入口：

```bash
generate-voronoi-aerogel --config default_config.json --output output/sheet_full
```

运行过程中终端会显示种子数量、Voronoi 逻辑面数量、骨架和选面结果、孔隙率核验、体素化以及网格导出进度。正常结束后显示 `Finished` 和 `Model written to`。大模型的某些阶段耗时较长，进度不会逐个面刷新。

## 2. 从哪里修改参数

用任意文本编辑器或 Python 编辑器打开仓库根目录的 `default_config.json`。建议复制为新文件，例如 `sheet_config_97.json`，再修改，便于保留不同参数组。

JSON 使用英文双引号，字段之间用英文逗号分隔，最后一项后不加逗号；不能在数值后写 `nm`、`%` 或代码注释。所有长度使用 **nm**。孔隙率是 **0 到 1 的小数**：`0.95` 表示 95%，`0.97` 表示 97%。

最常修改的字段：

| 字段 | 实际默认值 | 含义 |
| --- | --- | --- |
| `cube_size_nm` | `1000.0` | 立方体胞元边长，默认 1000 nm |
| `thickness_nm` | `5.0` | 片层总厚度，沿中面法向两侧各延伸一半 |
| `diameter_min_nm` | `50.0` | 优先选择的片层等效直径下限 |
| `diameter_max_nm` | `80.0` | 优先选择的片层等效直径上限 |
| `target_porosity` | `0.95` | 目标孔隙率 |
| `porosity_tolerance` | `0.005` | 选面控制所用孔隙率估计的绝对容差，默认 ±0.5 个百分点 |
| `min_in_range_fraction` | `0.90` | 最终所选逻辑面中，等效直径在上述窗口内的最低数量比例 |
| `random_seed` | `42` | 随机种子，用于复现种子生成和选面过程 |
| `num_seeds` | `null` | Voronoi 种子数；`null` 表示由尺度和目标固相体积分数自动估算 |
| `output_dir` | `"output/default_model"` | 输出目录，默认相对于当前工作目录 |

“等效直径”由每个完整逻辑面的面积计算：`d_eq = 2 × sqrt(A / π)`。Voronoi 面仍是不规则多边形，不会被改成直径相同的圆片。`min_in_range_fraction` 按面数量统计，而非面积或体积占比。

计算分辨率及进阶参数：

| 字段 | 实际默认值 | 含义 |
| --- | --- | --- |
| `seed_min_distance_factor` | `0.32` | 种子最小周期距离系数；实际最小距离为该值乘 `cube_size_nm / num_seeds^(1/3)` |
| `qmc_power` | `20` | 控制孔隙率的 Sobol 采样指数，点数为 `2^20 = 1,048,576` |
| `validation_qmc_power` | `19` | 独立核验 Sobol 点数指数，默认 `2^19 = 524,288` |
| `voxel_size_nm` | `2.5` | 实体 STL 的请求体素间距；程序会略微缩小间距，使整数个体素恰好填满胞元 |
| `max_generation_attempts` | `4` | 几何与选面尝试次数；每次重试使用确定性的新随机种子 |
| `max_selected_faces` | `25000` | 固相选面生长的数量上限 |
| `min_face_area_nm2` | `25.0` | 允许选作固相的最小逻辑面面积，单位 nm² |
| `min_face_edge_nm` | `1.0` | 允许选作固相的面内最短边长度下限 |
| `winding_offset_bound` | `1` | 周期覆盖空间中搜索绕回路径的平移层数界限 |
| `preview_face_limit` | `15000` | PNG 预览最多显示的三角面数量；只影响预览，不影响 STL/BDF 导出 |

`qmc_power` 和 `validation_qmc_power` 的配置范围为 12–24。指数每增加 1，采样点数翻倍，也会提高时间和内存消耗。

`num_seeds=null` 时，程序可能在重试时增加种子数；显式设置正整数后，重试保持该种子数不变。修改种子数、胞元尺寸或随机种子都会改变片径分布，不能把片径窗口理解为对所有面逐一裁剪的硬尺寸。

## 3. 修改孔隙率、尺寸并运行

例如，希望保留 1000 nm 胞元、5 nm 片厚和 50–80 nm 等效片径窗口，尝试生成 97% 孔隙率的模型：

1. 复制 `default_config.json` 为 `sheet_config_97.json`。
2. 将 `target_porosity` 改为 `0.97`。
3. 将 `output_dir` 改为 `"output/sheet_97"`。
4. 保存配置后运行：

```bash
python -m voronoi_aerogel.cli --config sheet_config_97.json --quick
```

快速检查可行后，用新的输出目录执行正式版本：

```bash
python -m voronoi_aerogel.cli --config sheet_config_97.json --output output/sheet_97_full
```

如需改变胞元或片厚，分别修改 `cube_size_nm` 和 `thickness_nm`。程序不会为了满足孔隙率而偷偷增加片厚。目标孔隙率、片径比例和连通拓扑未必能同时实现；失败时会报告原因，不能保证任意尺寸组合都可生成 97% 或 99% 模型。

命令行还允许临时覆盖输出目录和随机种子：

```bash
python -m voronoi_aerogel.cli --config default_config.json --seed 123 --output output/sheet_seed123
```

片层命令行没有 `--porosity` 选项；孔隙率在 JSON 配置中修改。`--quick` 将控制采样指数最多设为 16、独立核验指数最多设为 15，并将请求体素间距至少设为片厚，其余配置仍生效。

每组参数建议使用新的输出文件夹。程序会覆盖同名输出，而失败运行也可能留下未完成文件；请根据最后一次运行状态和最终 `metadata.json` 判断结果。

## 4. 在 Spyder 或其他 Python 编辑器中运行

可以直接调用 Python API。将以下代码保存为仓库根目录的个人运行脚本，例如 `my_sheet_run.py`，在安装本项目的同一 Python 环境中打开并运行：

```python
from pathlib import Path

from voronoi_aerogel import ModelConfig, generate_model

project_dir = Path(__file__).resolve().parent
config = ModelConfig.from_json(project_dir / "default_config.json")
config.target_porosity = 0.95
config.cube_size_nm = 1000.0
config.thickness_nm = 5.0
config.output_dir = str(project_dir / "output" / "my_sheet_model")

result = generate_model(config, progress=print)
print("输出目录：", result.output_dir)
print("独立核验孔隙率：", result.validation_porosity)
```

运行整个脚本，例如在 Spyder 中按 F5。此代码通过脚本位置确定路径，不依赖编辑器的当前工作目录。片层 API 一次完成图片、STL 和 BDF 等输出，没有单独的“先图片、后 STL”缓存步骤。纳米线模型的两步操作见其专用说明。

若编辑器提示找不到 `voronoi_aerogel`，请确认它使用的 Python 已执行 `python -m pip install -e .`。切换解释器后通常需要重启编辑器内核。

## 5. 输出文件及单位

| 文件 | 用途及单位 |
| --- | --- |
| `preview.png` | 片层中面三角网格的快速预览；可能按 `preview_face_limit` 抽取三角面，因此不是完整实体网格截图 |
| `aerogel_solid_nm.stl` | 体素并集提取的封闭三角表面，数值坐标单位 nm；STL 格式本身不保存单位 |
| `aerogel_shell.bdf` | NASTRAN 中面三角壳网格，GRID 坐标及 PSHELL 厚度单位 μm |
| `periodic_node_pairs.json` | 壳网格相对边界节点配对和等价类，坐标及平移量单位 nm |
| `import_lumerical.lsf` | STL 导入及 FDTD 区域设置模板，使用 `1e-9` 将 STL 的 nm 数值换算为 m |
| `README_COMSOL.md` | 生成模型对应的 COMSOL 导入及周期条件说明 |
| `README_LUMERICAL.md` | 生成模型对应的 Lumerical 导入及后续设置说明 |
| `periodic_geometry.npz` | 主胞元种子、所选逻辑面编号、顶点和面积等复现数据 |
| `config_used.json` | 本次运行实际生效的配置快照，包含命令行覆盖值 |
| `feasibility.json` | 几何、选面和拓扑预检结果及历次尝试；`PASSED` 表示预检通过 |
| `metadata.json` | 完整成功运行后的几何统计、孔隙率、拓扑、网格质量和文件索引 |

导出 STL 后，使用外部软件打开时必须指定正确单位或缩放比例。例如转换为 μm 时，所有 STL 坐标乘 `0.001`；转换为 m 时乘 `1e-9`。不要把默认 1000 的胞元边长直接按 mm 导入。

壳网格保留配置中的片厚属性；实体 STL 的厚度精度则受体素间距和表面提取离散误差影响。两个表示不会在所有尺度上完全相同。

COMSOL 的节点配对 JSON 只是描述数据，不会自动激活热周期边界条件。BDF 中的材料卡是占位信息，需在求解器中设置氧化铝的适用材料参数并验证约束。Lumerical 脚本同样是模板，仍需设置材料色散、光源、监视器和收敛网格；其 x/y 使用周期边界，z 使用带缓冲区的 PML，以支持有限厚度样品的透射问题。

## 6. Voronoi 片层构建方法

1. **生成周期随机种子。** 在立方体三维环面上采用带最小距离抑制的随机布点，相对边界两侧也遵守最小距离。
2. **计算周期 Voronoi。** 将种子复制到 3×3×3 邻域，由 SciPy/Qhull 构建普通三维 Voronoi 图。公共面以相邻胞元编号和整数周期平移规范化，合并重复副本。
3. **保留逻辑面。** 每个片层对应一个完整的不规则 Voronoi 公共面。几何先在展开周期空间中保存，导出主胞元时才平移、裁剪；跨出一侧的面在相对侧继续出现。
4. **构造双相骨架。** 在胞元对偶图中保护开放孔道骨架，再从其余可用面中建立相连的固相骨架。对周期回路的整数平移矢量计算秩，以检查逻辑网络具有三个独立绕回方向。
5. **添加完整片层。** 从固相邻接前沿随机生长，优先选择等效片径位于指定窗口的面。保护孔道对应的面不被选择。接近目标时比较候选面的边际并集体积；跨过目标后保留误差较小的一侧，不缩短或切掉最后一片。
6. **估计孔隙率。** 每个面沿法向扩展为有限厚度棱柱，用共享的 scrambled Sobol 点集估计所有片层的周期物理并集。交叠区域只计一次。孔隙率定义为 `1 − 并集固相体积 / 立方体体积`。
7. **核验并导出。** 使用不同扰动种子的 Sobol 点集独立核验，随后体素化片层并集，以 marching cubes 提取封闭主胞元表面并检查水密性；另外导出精确中面壳网格及周期节点配对。

## 7. 如何解释孔隙率和连通性

`metadata.json` 同时记录四种孔隙率：

- `control_qmc`：选面控制所使用的 Sobol 并集估计；`porosity_tolerance` 的成功判据作用于此值。
- `independent_qmc`：独立 Sobol 点集的复核结果；当前片层程序会记录此值，但不会因为它超过目标容差而自动拒绝模型。
- `voxel_count`：实体化占据网格中的空体素比例。
- `closed_mesh_volume`：最终封闭网格体积换算得到的孔隙率。

上述数值有采样和网格离散误差，不应只选其中最接近目标的一个报告。`reference_95pct_binomial_half_width` 只是二项抽样参考误差尺度，不能当作 scrambled Sobol 估计的严格 95% 置信区间。

完整逻辑面是离散选择单元，因此一般不能精确达到任意孔隙率。预检给出的可达区间受当前双骨架和排除条件限制，其中致密端也是未完全施加片径数量比例约束的乐观界限；目标位于区间内不等于最终一定成功。

固相、孔相“三向贯通”的检查针对所选面邻接图及未阻断面的胞元对偶图。程序没有额外计算最终有限厚度、体素化后两相的空间连通分量；较厚片层或较粗体素可能改变局部孔道。正式输运模拟还需独立检查最终导入几何的连通性和周期边界匹配。

`feasibility.json` 在昂贵实体化之前写入。只有程序正常完成并生成最终 `metadata.json` 才代表完整导出成功。旧输出目录中残留的 `metadata.json` 可能来自前一次运行，使用新目录能避免混淆。

## 8. 内存、分辨率与常见问题

默认 1000 nm 胞元配合 2.5 nm 体素间距，每轴为 400 个体素，总计 6400 万个体素。占据数组以外还有 marching cubes、顶点、三角面和网格边统计等中间数组，峰值内存会明显大于占据数组本身。默认快速模式使用至少 5 nm 间距，对应每轴 200 个体素、合计 800 万体素。

体素间距减半时，体素总数约增加到 8 倍。默认 5 nm 片厚在正式默认网格中只有两个体素分辨率，应至少比较更细网格，确认体积、孔隙率和所研究物理量收敛后再用于定量结论。较细网格的计算成本需要根据机器内存评估。

- **缺少 Python 包：** 在编辑器实际使用的 Python 环境中，从仓库根目录重新执行 `python -m pip install -e .`。
- **配置文件找不到：** 确认终端当前目录为仓库根目录，或给 `--config` 传入完整路径。
- **预检失败或片径比例不满足：** 查看 `feasibility.json` 的 `attempts` 和状态。调整孔隙率、片径窗口、种子数量或允许的片径比例，并使用新目录重试；不要只放宽采样量而忽视约束冲突。
- **种子布置失败：** 最小距离系数可能过大，导致规定尝试次数内无法放下全部种子。适当降低 `seed_min_distance_factor`。
- **内存不足：** 先用 `--quick` 检查流程，必要时采用较大体素间距。若改变了尺寸或分辨率，须重新评估误差。
- **最终 STL 不是水密网格或边界节点未一一匹配：** 程序会抛出错误。该运行不应视为成功；保存配置及报错内容以便复现。
- **预览显示稀疏或断续：** 片层 PNG 预览可能抽取三角面。可提高 `preview_face_limit`，或在外部三维查看器中直接打开完整 STL/BDF；预览抽样不会删改导出的网格。

固定配置和随机种子用于复现同一计算设置下的生成结果；跨 SciPy/Qhull 等依赖版本时，几何编号或数值细节可能变化。用于发表的结果请一并保留配置、依赖版本、`metadata.json` 和复现几何数据。
