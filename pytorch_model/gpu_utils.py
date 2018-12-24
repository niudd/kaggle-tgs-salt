import subprocess
import re


def show_gpu_usage():
    info = subprocess.check_output(['nvidia-smi -i 0,1'], shell=True).decode("utf-8")

    pat = re.compile(r'\d+MiB')
    match = pat.findall(info)
    memory_usage0 = match[0]
    memory_usage1 = match[2]

    pat = re.compile(r'\d+%')
    match = pat.findall(info)
    util_usage0 = match[1]
    util_usage1 = match[3]

    gpu0 = {'name': 'gpu0', 'memory_usage': memory_usage0, 'util_usage': util_usage0}
    gpu1 = {'name': 'gpu1', 'memory_usage': memory_usage1, 'util_usage': util_usage1}

    return gpu0, gpu1


if __name__ == "__main__":
    print(show_gpu_usage())