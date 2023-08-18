
import launch
import os
import pkg_resources
import sys
from tqdm import tqdm
import urllib.request
import asyncio

req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")

models_dir = os.path.abspath("models/roop")
model_url = "https://huggingface.co/henryruhs/roop/resolve/main/inswapper_128.onnx"
model_name = os.path.basename(model_url)
model_path = os.path.join(models_dir, model_name)

async def download(url, path):
    request = urllib.request.urlopen(url)
    total = int(request.headers.get('Content-Length', 0))
    with tqdm(total=total, desc='Downloading', unit='B', unit_scale=True, unit_divisor=1024) as progress:
        urllib.request.urlretrieve(url, path, reporthook=lambda count, block_size, total_size: progress.update(block_size))

async def check_and_install_packages():
    to_install = []
    with open(req_file) as file:
        for package in file:
            package = package.strip()
            if not launch.is_installed(package):
                to_install.append(package)
            elif "==" in package:
                package_name, package_version = package.split("==")
                installed_version = pkg_resources.get_distribution(package_name).version
                if installed_version != package_version:
                    to_install.append(package)
    if to_install:
        packages_str = " ".join(to_install)
        launch.run_pip(f"install {packages_str}", f"sd-webui-roop requirements: {packages_str}")

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

async def main():
    tasks = []
    if not os.path.exists(model_path):
        tasks.append(asyncio.create_task(download(model_url, model_path)))
    tasks.append(asyncio.create_task(check_and_install_packages()))
    await asyncio.gather(*tasks)

asyncio.run(main())
