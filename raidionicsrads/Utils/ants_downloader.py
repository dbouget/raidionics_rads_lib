import os
import platform
import shutil
import urllib.request
import tarfile


def ensure_ants_present():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return  # Only needed on macOS ARM64

    ants_path = os.path.join(os.path.dirname(__file__), "..", "ANTs")
    if os.path.exists(ants_path):
        return  # Already extracted

    url = "https://github.com/raidionics/Raidionics-dependencies/releases/download/v1.0.0/ANTsX-v2.4.3_macos_arm.tar.gz"
    download_path = os.path.join(ants_path, "..", "ANTsX.tar.gz")

    print("Downloading ANTs for macOS ARM...")
    os.makedirs(ants_path, exist_ok=True)
    urllib.request.urlretrieve(url, download_path)

    ants_path_tmp = os.path.join(os.path.dirname(__file__), "..", "ANTs_tmp")
    print(f"Extracting ANTs in {ants_path}")
    with tarfile.open(download_path, "r:gz") as tar:
        tar.extractall(path=ants_path_tmp)
    tar.close()
    os.remove(download_path)

    shutil.move(os.path.join(ants_path_tmp, "install", "bin"), os.path.join(ants_path, "bin"))
    shutil.move(os.path.join(ants_path_tmp, "install", "lib"), os.path.join(ants_path, "lib"))
    shutil.move(os.path.join(ants_path_tmp, "install", "Scripts"), os.path.join(ants_path, "Scripts"))
    shutil.rmtree(ants_path_tmp)