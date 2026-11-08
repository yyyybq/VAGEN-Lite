# Source on the login node before downloads or data preparation:  source eval_external/tools/login_env.sh
# Through the default proxy Hugging Face gives ~0.1 MB/s; hf-mirror without the proxy gives ~30 MB/s (2026-10-18).
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
# Our own HF_HOME: its hub/ links to the shared ~/.cache/huggingface/hub (models), while datasets/ and the
# unpacked videos are ours (the shared datasets/ and vsibench/ were left root-owned by earlier sco jobs).
export HF_HOME=/mnt/umm/users/yinbaiqiao/eval_external_assets/hf_home
export LMUData=/mnt/umm/users/yinbaiqiao/eval_external_assets/LMUData
export PATH=/mnt/umm/users/yinbaiqiao/.conda/envs/eval-external/bin:$PATH
export PYTHONNOUSERSITE=1
