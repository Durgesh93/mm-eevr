import os
import sys
import site
import glob
import shutil
import platform
import subprocess
import importlib.metadata as im

# ============================================================
# Environment flags
# ============================================================
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ============================================================
# Required packages
# ============================================================
BASE_REQUIRED = {
    "numpy":"1.26.4",
    "tensorflow":"2.18.0",
    "tensorflow-probability":"0.25.0",
    "tf-keras":"2.18.0",
    "protobuf":"4.25.8",
    "transformers":"4.46.3",
    "wheel":"0.35.1",
    "datasets":None,
    "evaluate":None,
    "scikit-learn":None,
    "imbalanced-learn":None,
    "wordcloud":None,
    "umap-learn":None,
}

CUDA_REQUIRED = [
    "nvidia-cublas-cu12",
    "nvidia-cuda-cupti-cu12",
    "nvidia-cuda-nvcc-cu12",
    "nvidia-cuda-nvrtc-cu12",
    "nvidia-cuda-runtime-cu12",
    "nvidia-cudnn-cu12",
    "nvidia-cufft-cu12",
    "nvidia-curand-cu12",
    "nvidia-cusolver-cu12",
    "nvidia-cusparse-cu12",
    "nvidia-nccl-cu12",
    "nvidia-nvjitlink-cu12",
]


# ============================================================
# Helpers
# ============================================================
def run(cmd,check=True):
    print("\nRUN:")
    print(" ".join(cmd))

    result = subprocess.run(
        cmd,
        text=True,
    )

    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode,cmd)

    return result


def run_capture(cmd):
    result = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
    )

    return result.returncode,result.stdout,result.stderr


def get_version(pkg):
    try:
        return im.version(pkg)
    except im.PackageNotFoundError:
        return None


def detect_base():
    base = {}

    base["python"] = sys.version
    base["executable"] = sys.executable
    base["platform"] = platform.platform()
    base["system"] = platform.system()
    base["machine"] = platform.machine()
    base["is_linux"] = platform.system() == "Linux"
    base["is_root"] = hasattr(os,"geteuid") and os.geteuid() == 0
    base["conda_prefix"] = os.environ.get("CONDA_PREFIX")
    base["virtual_env"] = os.environ.get("VIRTUAL_ENV")
    base["use_cuda"] = True
    base["tf_package"] = "tensorflow[and-cuda]==2.18.0"    
    return base


def print_base(base):
    print("\nEnvironment detected:")
    print("Python:",base["python"])
    print("Executable:",base["executable"])
    print("Platform:",base["platform"])
    print("Conda:",base["conda_prefix"])
    print("Virtualenv:",base["virtual_env"])
    print("Root user:",base["is_root"])
    print("TensorFlow selected:",base["tf_package"])


def needs_install(base):
    reinstall_needed = False

    for pkg,required_version in BASE_REQUIRED.items():
        installed_version = get_version(pkg)

        if installed_version is None:
            print(f"Missing: {pkg}")
            reinstall_needed = True
            continue

        if required_version is not None and installed_version != required_version:
            print(f"Version mismatch: {pkg} installed={installed_version}, required={required_version}")
            reinstall_needed = True

    if base["use_cuda"]:
        for pkg in CUDA_REQUIRED:
            installed_version = get_version(pkg)

            if installed_version is None:
                print(f"Missing CUDA package: {pkg}")
                reinstall_needed = True

    return reinstall_needed


def add_nvidia_library_paths():
    paths = []

    for base_path in site.getsitepackages():
        nvidia_base = os.path.join(base_path,"nvidia")

        if os.path.exists(nvidia_base):
            paths += glob.glob(os.path.join(nvidia_base,"*","lib"))

    paths = [p for p in paths if os.path.exists(p)]
    old_ld = os.environ.get("LD_LIBRARY_PATH","")

    for p in paths:
        if p not in old_ld:
            old_ld = p + ":" + old_ld

    os.environ["LD_LIBRARY_PATH"] = old_ld
    return paths


def check_tensorflow_cuda():
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    add_nvidia_library_paths()

    import tensorflow as tf
    import tf_keras as tfk
    import tensorflow_probability as tfp

    print("\nTensorFlow check:")
    print("TensorFlow:",tf.__version__)
    print("tf-keras:",tfk.__version__)
    print("TFP:",tfp.__version__)
    print("Built with CUDA:",tf.test.is_built_with_cuda())
    print("GPUs:",tf.config.list_physical_devices("GPU"))


# ============================================================
# Main installer
# ============================================================
def prepare_env(force=False):
    base = detect_base()

    print_base(base)
    
    if not force and not needs_install(base):
        print("\nEnvironment packages OK. No installation needed.")
        check_tensorflow_cuda()
        return

    print("\nInstalling/reinstalling required packages...")

    uninstall_list = [
        "tensorflow",
        "tensorflow-cpu",
        "tensorflow-intel",
        "tensorflow-gpu",
        "tensorflow-probability",
        "tf-keras",
        "keras",
        "keras-nightly",
        "protobuf",
        "tensorboard",
        "transformers",
        "tokenizers",
        *CUDA_REQUIRED,
    ]

    run([sys.executable,"-m","pip","uninstall","-y",*uninstall_list],check=False)

    run(
        [
            sys.executable,"-m","pip","install",
            "--upgrade","pip","setuptools",
        ],
        check=True,
    )

    run(
        [
            sys.executable,"-m","pip","install",
            "wheel==0.35.1",
        ],
        check=True,
    )

    tf_stack = [
        "numpy==1.26.4",
        "protobuf==4.25.8",
        base["tf_package"],
        "tf-keras==2.18.0",
        "tensorflow-probability==0.25.0",
    ]

    run(
        [
            sys.executable,"-m","pip","install",
            "--no-cache-dir",
            *tf_stack,
        ],
        check=True,
    )

    nlp_stack = [
        "transformers==4.46.3",
        "tokenizers<0.21",
    ]

    run(
        [
            sys.executable,"-m","pip","install",
            "--no-cache-dir",
            *nlp_stack,
        ],
        check=True,
    )

    extras = [
        "datasets",
        "evaluate",
        "scikit-learn",
        "imbalanced-learn",
        "wordcloud",
        "umap-learn",
    ]

    run(
        [
            sys.executable,"-m","pip","install",
            "--no-cache-dir",
            *extras,
        ],
        check=True,
    )

    print("\nPip check:")
    run([sys.executable,"-m","pip","check"],check=False)

    if base["use_cuda"]:
        add_nvidia_library_paths()

    print("\nDone. Restart the kernel/runtime before importing TensorFlow or Transformers.")
    print("After restart, run:")
    print("prepare_env(force=False)")