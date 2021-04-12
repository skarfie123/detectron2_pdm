# modified from https://github.com/airesearch-in-th/kora
import os
from subprocess import PIPE, Popen

from IPython.display import Javascript

# Install
url = "https://github.com/gravitational/teleconsole/releases/download/0.4.0/teleconsole-v0.4.0-linux-amd64.tar.gz"
os.system(f"curl -L {url} | tar xz")  # download & extract
os.system("mv teleconsole /usr/local/bin/")  # in PATH

# Set PS1, directory
with open("/root/.bashrc", "a") as f:
    f.write('PS1="\e[1;36m\w\e[m# "\n')
    f.write("cd /content \n")
    f.write(
        "PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/tools/node/bin:/tools/google-cloud-sdk/bin:/opt/bin \n"
    )


def start():
    """ start the teleconsole, and print URL """
    process = Popen("teleconsole", shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    lines = [process.stdout.readline().decode() for i in range(6)]
    try:
        url = lines[-1].strip().split()[-1]
        print("Console URL:", url)
        display(Javascript('window.open("{url}");'.format(url=url)))
    except:
        print("".join(lines))


def stop():
    os.system("pkill teleconsole")


def restart():
    stop()
    start()
