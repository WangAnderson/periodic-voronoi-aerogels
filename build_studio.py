"""Build a portable Windows folder. Run in the environment with studio extras."""
from pathlib import Path
import subprocess
import sys
import os


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    subprocess.run([
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--onedir", "--windowed",
        "--name", "AerogelStudio", "--distpath", str(root / "dist"),
        "--workpath", str(root / "build" / "studio"),
        "--specpath", str(root / "build"),
        "--icon", str(root / "aerogel_studio" / "assets" / "app_icon.ico"),
        "--add-data", str(root / "aerogel_studio" / "assets" / "app_icon.ico") + os.pathsep + "aerogel_studio/assets",
        "--collect-all", "pyvista", "--collect-all", "pyvistaqt",
        "--hidden-import", "vtkmodules.all",
        "--hidden-import", "vtkmodules.util.numpy_support",
        "--exclude-module", "PyQt5", "--exclude-module", "PyQt6",
        "--exclude-module", "PySide2", "--exclude-module", "IPython",
        "--exclude-module", "jupyterlab", "--exclude-module", "notebook",
        "--exclude-module", "torch", "--exclude-module", "tensorflow",
        "--exclude-module", "OCP", "--exclude-module", "pandas",
        "--exclude-module", "bokeh", "--exclude-module", "sphinx",
        "--exclude-module", "tkinter", "--exclude-module", "pytest",
        str(root / "run_aerogel_studio.py"),
    ], cwd=root, check=True)
    import shutil
    destination = root / "dist" / "AerogelStudio"
    # scipy.stats Sobol loads this data lazily, outside normal import analysis.
    import scipy
    sobol_data = Path(scipy.__file__).parent / "stats" / "_sobol_direction_numbers.npz"
    target_data = destination / "_internal" / "scipy" / "stats" / sobol_data.name
    target_data.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sobol_data, target_data)
    shutil.copy2(root / "LICENSE", destination / "LICENSE.txt")
    python_license = Path(sys.base_prefix) / "LICENSE_PYTHON.txt"
    if python_license.is_file():
        shutil.copy2(python_license, destination / "PYTHON_LICENSE.txt")
    shutil.copy2(root / "aerogel_studio" / "assets" / "app_icon.ico", destination / "AerogelStudio.ico")
    shutil.copy2(root / "STUDIO_VALIDATION.md", destination / "验证记录.md")
    shutil.copy2(root / "docs" / "desktop_studio_zh.md", destination / "使用说明.md")
    shutil.copytree(root / "studio_demo" / "nanowire_100nm", destination / "示例项目" / "纳米线_100nm", dirs_exist_ok=True)
    (destination / "START_HERE.txt").write_text(
        "双击 AerogelStudio.exe 启动。请保留同目录 _internal 文件夹。\n"
        "无需安装 Python。生成数据默认保存到 文档/AerogelStudio。\n"
        "完整说明见 使用说明.md。\n", encoding="utf-8")
    # Keep dependency notices separate from this project's MIT license.
    from importlib import metadata
    notices = ["Aerogel Studio application source: MIT.\n"
               "Bundled third-party libraries retain their own licenses.\n"
               "Qt/PySide6 and shiboken6 are dynamically linked LGPL components.\n"
               "Qt source and license information: https://www.qt.io/licensing/\n"
               "PySide source: https://code.qt.io/cgit/pyside/pyside-setup.git/\n"]
    license_root = destination / "THIRD_PARTY_LICENSES"
    for package in ("PySide6", "PySide6_Essentials", "PySide6_Addons", "shiboken6",
                    "numpy", "scipy", "scikit-image", "matplotlib", "pillow", "plotly",
                    "pyvista", "pyvistaqt", "vtk", "QtPy", "PyInstaller", "pooch",
                    "scooby", "requests", "urllib3", "certifi", "charset-normalizer",
                    "idna", "rich", "cyclopts", "pyvista-validation"):
        try:
            dist = metadata.distribution(package)
        except metadata.PackageNotFoundError:
            continue
        notices.append(f"{dist.metadata['Name']} {dist.version}\n")
        for relative in dist.files or []:
            if any(token in str(relative).lower() for token in ("license", "copying", "notice")):
                source = Path(dist.locate_file(relative))
                if source.is_file() and source.suffix.lower() in ("", ".txt", ".md", ".rst", ".html"):
                    target = license_root / package / str(relative).replace("../", "")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
    (destination / "THIRD_PARTY_NOTICES.txt").write_text("\n".join(notices), encoding="utf-8")
    print(f"Portable application: {destination}")
