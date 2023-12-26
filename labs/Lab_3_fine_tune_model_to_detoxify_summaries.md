# Fine-Tune FLAN-T5 with Reinforcement Learning (PPO) and PEFT to Generate Less-Toxic Summaries

In this notebook, you will fine-tune a FLAN-T5 model to generate less toxic content with Meta AI's hate speech reward model. The reward model is a binary classifier that predicts either "not hate" or "hate" for the given text. You will use Proximal Policy Optimization (PPO) to fine-tune and reduce the model's toxicity.

# Table of Contents

- [ 1 - Set up Kernel and Required Dependencies](#1)
- [ 2 - Load FLAN-T5 Model, Prepare Reward Model and Toxicity Evaluator](#2)
  - [ 2.1 - Load Data and FLAN-T5 Model Fine-Tuned with Summarization Instruction](#2.1)
  - [ 2.2 - Prepare Reward Model](#2.2)
  - [ 2.3 - Evaluate Toxicity](#2.3)
- [ 3 - Perform Fine-Tuning to Detoxify the Summaries](#3)
  - [ 3.1 - Initialize `PPOTrainer`](#3.1)
  - [ 3.2 - Fine-Tune the Model](#3.2)
  - [ 3.3 - Evaluate the Model Quantitatively](#3.3)
  - [ 3.4 - Evaluate the Model Qualitatively](#3.4)

<a name='1'></a>
## 1 - Set up Kernel and Required Dependencies

First, check that the correct kernel is chosen.

<img src="images/kernel_set_up.png" width="300"/>

You can click on that (top right of the screen) to see and check the details of the image, kernel, and instance type.

<img src="images/w3_kernel_and_instance_type.png" width="600"/>

<img src="data:image/svg+xml;base64,Cjxzdmcgd2lkdGg9IjgwMCIgaGVpZ2h0PSIxMjUiIHZpZXdCb3g9IjAgMCA4MDAgMTI1IiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgogICAgPGRlZnM+CiAgICAgICAgPGxpbmVhckdyYWRpZW50IGlkPSJmYWRlR3JhZGllbnQiIHgxPSIwIiB4Mj0iMSI+CiAgICAgICAgICAgIDxzdG9wIG9mZnNldD0iMCUiIHN0b3AtY29sb3I9IiNGMEYwRjAiLz4KICAgICAgICAgICAgPHN0b3Agb2Zmc2V0PSIxMDAlIiBzdG9wLWNvbG9yPSIjRjBGMEYwIiBzdG9wLW9wYWNpdHk9IjAiLz4KICAgICAgICA8L2xpbmVhckdyYWRpZW50PgogICAgICAgIDxtYXNrIGlkPSJmYWRlTWFzayI+CiAgICAgICAgICAgIDxyZWN0IHg9IjAiIHk9IjAiIHdpZHRoPSI3NTAiIGhlaWdodD0iMTI1IiBmaWxsPSJ3aGl0ZSIvPgogICAgICAgICAgICA8cmVjdCB4PSI3NTAiIHk9IjAiIHdpZHRoPSI1MCIgaGVpZ2h0PSIxMjUiIGZpbGw9InVybCgjZmFkZUdyYWRpZW50KSIvPgogICAgICAgIDwvbWFzaz4KICAgIDwvZGVmcz4KICAgIDxwYXRoIGQ9Ik01MCw5NyBBNTAsNTAgMCAwIDEgNTMsMyBMNzk3LCAzIEw3OTcsOTcgTDUwLDk3IFoiIGZpbGw9IiNGMEYwRjAiIHN0cm9rZT0iI0UwRTBFMCIgc3Ryb2tlLXdpZHRoPSIxIiBtYXNrPSJ1cmwoI2ZhZGVNYXNrKSIvPgogICAgPHRleHQgeD0iMTAwIiB5PSIzNCIgZm9udC1mYW1pbHk9IkFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjE0IiBmaWxsPSIjMzMzMzMzIj5QbGVhc2UgbWFrZSBzdXJlIHRoYXQgeW91IGNob29zZTwvdGV4dD4gCiAgICA8dGV4dCB4PSIzMjAiIHk9IjM0IiBmb250LWZhbWlseT0iQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTQiIGZpbGw9IiMzMzMzMzMiIGZvbnQtd2VpZ2h0PSJib2xkIj5tbC5tNS4yeGxhcmdlPC90ZXh0PgogICAgPHRleHQgeD0iNDE4IiB5PSIzNCIgZm9udC1mYW1pbHk9IkFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjE0IiBmaWxsPSIjMzMzMzMzIj5pbnN0YW5jZSB0eXBlLjwvdGV4dD4KICAgIDx0ZXh0IHg9IjEwMCIgeT0iNTYiIGZvbnQtZmFtaWx5PSJBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzMzMzMzMyI+VG8gZmluZCB0aGF0IGluc3RhbmNlIHR5cGUsIHlvdSBtaWdodCBoYXZlIHRvIHNjcm9sbCBkb3duIHRvIHRoZSAiQWxsIEluc3RhbmNlcyIgc2VjdGlvbiBpbiB0aGUgZHJvcGRvd24uPC90ZXh0PgogICAgPHRleHQgeD0iMTAwIiB5PSI3OCIgZm9udC1mYW1pbHk9IkFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjE0IiBmaWxsPSIjMzMzMzMzIj5DaG9pY2Ugb2YgYW5vdGhlciBpbnN0YW5jZSB0eXBlIG1pZ2h0IGNhdXNlIHRyYWluaW5nIGZhaWx1cmUva2VybmVsIGhhbHQvYWNjb3VudCBkZWFjdGl2YXRpb24uPC90ZXh0Pgo8L3N2Zz4K" alt="Time alert close"/>

Now install the required packages to use PyTorch and Hugging Face transformers and datasets.

<img src="data:image/svg+xml;base64,Cjxzdmcgd2lkdGg9IjgwMCIgaGVpZ2h0PSIxMjUiIHZpZXdCb3g9IjAgMCA4MDAgMTI1IiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgogICAgPGRlZnM+CiAgICAgICAgPGxpbmVhckdyYWRpZW50IGlkPSJmYWRlR3JhZGllbnQiIHgxPSIwIiB4Mj0iMSI+CiAgICAgICAgICAgIDxzdG9wIG9mZnNldD0iMCUiIHN0b3AtY29sb3I9IiNGMEYwRjAiLz4KICAgICAgICAgICAgPHN0b3Agb2Zmc2V0PSIxMDAlIiBzdG9wLWNvbG9yPSIjRjBGMEYwIiBzdG9wLW9wYWNpdHk9IjAiLz4KICAgICAgICA8L2xpbmVhckdyYWRpZW50PgogICAgICAgIDxtYXNrIGlkPSJmYWRlTWFzayI+CiAgICAgICAgICAgIDxyZWN0IHg9IjAiIHk9IjAiIHdpZHRoPSI3NTAiIGhlaWdodD0iMTI1IiBmaWxsPSJ3aGl0ZSIvPgogICAgICAgICAgICA8cmVjdCB4PSI3NTAiIHk9IjAiIHdpZHRoPSI1MCIgaGVpZ2h0PSIxMjUiIGZpbGw9InVybCgjZmFkZUdyYWRpZW50KSIvPgogICAgICAgIDwvbWFzaz4KICAgIDwvZGVmcz4KICAgIDxwYXRoIGQ9Ik0zLDUwIEE1MCw1MCAwIDAgMSA1MywzIEw3OTcsMyBMNzk3LDk3IEw5Nyw5NyBMNTAsMTE1IEwzLDk3IFoiIGZpbGw9IiNGMEYwRjAiIHN0cm9rZT0iI0UwRTBFMCIgc3Ryb2tlLXdpZHRoPSIxIiBtYXNrPSJ1cmwoI2ZhZGVNYXNrKSIvPgogICAgPGNpcmNsZSBjeD0iNTAiIGN5PSI1MCIgcj0iMzAiIGZpbGw9IiM1N2M0ZjgiIHN0cm9rZT0iIzU3YzRmOCIgc3Ryb2tlLXdpZHRoPSIxIi8+CiAgICA8Y2lyY2xlIGN4PSI1MCIgY3k9IjUwIiByPSIyNSIgZmlsbD0iI0YwRjBGMCIvPgogICAgPGxpbmUgeDE9IjUwIiB5MT0iNTAiIHgyPSI1MCIgeTI9IjMwIiBzdHJva2U9IiM1N2M0ZjgiIHN0cm9rZS13aWR0aD0iMyIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+CiAgICA8bGluZSB4MT0iNTAiIHkxPSI1MCIgeDI9IjY1IiB5Mj0iNTAiIHN0cm9rZT0iIzU3YzRmOCIgc3Ryb2tlLXdpZHRoPSIzIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz4KICAgIDx0ZXh0IHg9IjEwMCIgeT0iMzQiIGZvbnQtZmFtaWx5PSJBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzMzMzMzMyI+VGhlIG5leHQgY2VsbCBtYXkgdGFrZSBhIGZldyBtaW51dGVzIHRvIHJ1bi4gUGxlYXNlIGJlIHBhdGllbnQuPC90ZXh0PgogICAgPHRleHQgeD0iMTAwIiB5PSI1NiIgZm9udC1mYW1pbHk9IkFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjE0IiBmaWxsPSIjMzMzMzMzIj5JZ25vcmUgdGhlIHdhcm5pbmdzIGFuZCBlcnJvcnMsIGFsb25nIHdpdGggdGhlIG5vdGUgYWJvdXQgcmVzdGFydGluZyB0aGUga2VybmVsIGF0IHRoZSBlbmQuPC90ZXh0Pgo8L3N2Zz4K" alt="Time alert open medium"/>


```python
%pip install --upgrade pip
%pip install --disable-pip-version-check \
    torch==1.13.1 \
    torchdata==0.5.1 --quiet

%pip install \
    transformers==4.27.2 \
    datasets==2.11.0 \
    evaluate==0.4.0 \
    rouge_score==0.1.2 \
    peft==0.3.0 --quiet

# Installing the Reinforcement Learning library directly from github.
%pip install git+https://github.com/lvwerra/trl.git@25fa1bd    
```

    Requirement already satisfied: pip in /opt/conda/lib/python3.10/site-packages (23.3.1)
    Collecting pip
      Downloading pip-23.3.2-py3-none-any.whl.metadata (3.5 kB)
    Downloading pip-23.3.2-py3-none-any.whl (2.1 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.1/2.1 MB[0m [31m19.6 MB/s[0m eta [36m0:00:00[0m:00:01[0m
    [?25hInstalling collected packages: pip
      Attempting uninstall: pip
        Found existing installation: pip 23.3.1
        Uninstalling pip-23.3.1:
          Successfully uninstalled pip-23.3.1
    Successfully installed pip-23.3.2
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0mNote: you may need to restart the kernel to use updated packages.
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0mNote: you may need to restart the kernel to use updated packages.
    [31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    spyder 5.3.3 requires pyqt5<5.16, which is not installed.
    spyder 5.3.3 requires pyqtwebengine<5.16, which is not installed.
    pathos 0.3.1 requires dill>=0.3.7, but you have dill 0.3.6 which is incompatible.
    pathos 0.3.1 requires multiprocess>=0.70.15, but you have multiprocess 0.70.14 which is incompatible.
    sagemaker 2.199.0 requires urllib3<1.27, but you have urllib3 2.1.0 which is incompatible.
    spyder 5.3.3 requires ipython<8.0.0,>=7.31.1, but you have ipython 8.18.1 which is incompatible.
    spyder 5.3.3 requires pylint<3.0,>=2.5.0, but you have pylint 3.0.2 which is incompatible.[0m[31m
    [0m[33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0mNote: you may need to restart the kernel to use updated packages.
    Collecting git+https://github.com/lvwerra/trl.git@25fa1bd
      Cloning https://github.com/lvwerra/trl.git (to revision 25fa1bd) to /tmp/pip-req-build-3tnnltjy
      Running command git clone --filter=blob:none --quiet https://github.com/lvwerra/trl.git /tmp/pip-req-build-3tnnltjy
    [33m  WARNING: Did not find branch or tag '25fa1bd', assuming revision or ref.[0m[33m
    [0m  Running command git checkout -q 25fa1bd
      Resolved https://github.com/lvwerra/trl.git to commit 25fa1bd
      Preparing metadata (setup.py) ... [?25ldone
    [?25hRequirement already satisfied: torch>=1.4.0 in /opt/conda/lib/python3.10/site-packages (from trl==0.4.2.dev0) (1.13.1)
    Requirement already satisfied: transformers>=4.18.0 in /opt/conda/lib/python3.10/site-packages (from trl==0.4.2.dev0) (4.27.2)
    Requirement already satisfied: numpy>=1.18.2 in /opt/conda/lib/python3.10/site-packages (from trl==0.4.2.dev0) (1.26.2)
    Requirement already satisfied: accelerate in /opt/conda/lib/python3.10/site-packages (from trl==0.4.2.dev0) (0.25.0)
    Requirement already satisfied: datasets in /opt/conda/lib/python3.10/site-packages (from trl==0.4.2.dev0) (2.11.0)
    Requirement already satisfied: typing-extensions in /opt/conda/lib/python3.10/site-packages (from torch>=1.4.0->trl==0.4.2.dev0) (4.3.0)
    Requirement already satisfied: nvidia-cuda-runtime-cu11==11.7.99 in /opt/conda/lib/python3.10/site-packages (from torch>=1.4.0->trl==0.4.2.dev0) (11.7.99)
    Requirement already satisfied: nvidia-cudnn-cu11==8.5.0.96 in /opt/conda/lib/python3.10/site-packages (from torch>=1.4.0->trl==0.4.2.dev0) (8.5.0.96)
    Requirement already satisfied: nvidia-cublas-cu11==11.10.3.66 in /opt/conda/lib/python3.10/site-packages (from torch>=1.4.0->trl==0.4.2.dev0) (11.10.3.66)
    Requirement already satisfied: nvidia-cuda-nvrtc-cu11==11.7.99 in /opt/conda/lib/python3.10/site-packages (from torch>=1.4.0->trl==0.4.2.dev0) (11.7.99)
    Requirement already satisfied: setuptools in /opt/conda/lib/python3.10/site-packages (from nvidia-cublas-cu11==11.10.3.66->torch>=1.4.0->trl==0.4.2.dev0) (69.0.2)
    Requirement already satisfied: wheel in /opt/conda/lib/python3.10/site-packages (from nvidia-cublas-cu11==11.10.3.66->torch>=1.4.0->trl==0.4.2.dev0) (0.42.0)
    Requirement already satisfied: filelock in /opt/conda/lib/python3.10/site-packages (from transformers>=4.18.0->trl==0.4.2.dev0) (3.6.0)
    Requirement already satisfied: huggingface-hub<1.0,>=0.11.0 in /opt/conda/lib/python3.10/site-packages (from transformers>=4.18.0->trl==0.4.2.dev0) (0.17.3)
    Requirement already satisfied: packaging>=20.0 in /opt/conda/lib/python3.10/site-packages (from transformers>=4.18.0->trl==0.4.2.dev0) (21.3)
    Requirement already satisfied: pyyaml>=5.1 in /opt/conda/lib/python3.10/site-packages/PyYAML-6.0-py3.10-linux-x86_64.egg (from transformers>=4.18.0->trl==0.4.2.dev0) (6.0)
    Requirement already satisfied: regex!=2019.12.17 in /opt/conda/lib/python3.10/site-packages (from transformers>=4.18.0->trl==0.4.2.dev0) (2022.7.9)
    Requirement already satisfied: requests in /opt/conda/lib/python3.10/site-packages (from transformers>=4.18.0->trl==0.4.2.dev0) (2.31.0)
    Requirement already satisfied: tokenizers!=0.11.3,<0.14,>=0.11.1 in /opt/conda/lib/python3.10/site-packages (from transformers>=4.18.0->trl==0.4.2.dev0) (0.13.3)
    Requirement already satisfied: tqdm>=4.27 in /opt/conda/lib/python3.10/site-packages (from transformers>=4.18.0->trl==0.4.2.dev0) (4.64.1)
    Requirement already satisfied: psutil in /opt/conda/lib/python3.10/site-packages (from accelerate->trl==0.4.2.dev0) (5.9.0)
    Requirement already satisfied: safetensors>=0.3.1 in /opt/conda/lib/python3.10/site-packages (from accelerate->trl==0.4.2.dev0) (0.4.1)
    Requirement already satisfied: pyarrow>=8.0.0 in /opt/conda/lib/python3.10/site-packages (from datasets->trl==0.4.2.dev0) (14.0.1)
    Requirement already satisfied: dill<0.3.7,>=0.3.0 in /opt/conda/lib/python3.10/site-packages (from datasets->trl==0.4.2.dev0) (0.3.6)
    Requirement already satisfied: pandas in /opt/conda/lib/python3.10/site-packages (from datasets->trl==0.4.2.dev0) (2.1.3)
    Requirement already satisfied: xxhash in /opt/conda/lib/python3.10/site-packages (from datasets->trl==0.4.2.dev0) (3.4.1)
    Requirement already satisfied: multiprocess in /opt/conda/lib/python3.10/site-packages (from datasets->trl==0.4.2.dev0) (0.70.14)
    Requirement already satisfied: fsspec>=2021.11.1 in /opt/conda/lib/python3.10/site-packages (from fsspec[http]>=2021.11.1->datasets->trl==0.4.2.dev0) (2022.7.1)
    Requirement already satisfied: aiohttp in /opt/conda/lib/python3.10/site-packages (from datasets->trl==0.4.2.dev0) (3.9.1)
    Requirement already satisfied: responses<0.19 in /opt/conda/lib/python3.10/site-packages (from datasets->trl==0.4.2.dev0) (0.18.0)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /opt/conda/lib/python3.10/site-packages (from packaging>=20.0->transformers>=4.18.0->trl==0.4.2.dev0) (3.0.9)
    Requirement already satisfied: charset-normalizer<4,>=2 in /opt/conda/lib/python3.10/site-packages (from requests->transformers>=4.18.0->trl==0.4.2.dev0) (2.0.4)
    Requirement already satisfied: idna<4,>=2.5 in /opt/conda/lib/python3.10/site-packages (from requests->transformers>=4.18.0->trl==0.4.2.dev0) (3.3)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/conda/lib/python3.10/site-packages (from requests->transformers>=4.18.0->trl==0.4.2.dev0) (2.1.0)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/lib/python3.10/site-packages (from requests->transformers>=4.18.0->trl==0.4.2.dev0) (2023.11.17)
    Requirement already satisfied: attrs>=17.3.0 in /opt/conda/lib/python3.10/site-packages (from aiohttp->datasets->trl==0.4.2.dev0) (23.1.0)
    Requirement already satisfied: multidict<7.0,>=4.5 in /opt/conda/lib/python3.10/site-packages (from aiohttp->datasets->trl==0.4.2.dev0) (6.0.4)
    Requirement already satisfied: yarl<2.0,>=1.0 in /opt/conda/lib/python3.10/site-packages (from aiohttp->datasets->trl==0.4.2.dev0) (1.9.4)
    Requirement already satisfied: frozenlist>=1.1.1 in /opt/conda/lib/python3.10/site-packages (from aiohttp->datasets->trl==0.4.2.dev0) (1.4.1)
    Requirement already satisfied: aiosignal>=1.1.2 in /opt/conda/lib/python3.10/site-packages (from aiohttp->datasets->trl==0.4.2.dev0) (1.3.1)
    Requirement already satisfied: async-timeout<5.0,>=4.0 in /opt/conda/lib/python3.10/site-packages (from aiohttp->datasets->trl==0.4.2.dev0) (4.0.3)
    Requirement already satisfied: python-dateutil>=2.8.2 in /opt/conda/lib/python3.10/site-packages (from pandas->datasets->trl==0.4.2.dev0) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /opt/conda/lib/python3.10/site-packages (from pandas->datasets->trl==0.4.2.dev0) (2022.1)
    Requirement already satisfied: tzdata>=2022.1 in /opt/conda/lib/python3.10/site-packages (from pandas->datasets->trl==0.4.2.dev0) (2023.3)
    Requirement already satisfied: six>=1.5 in /opt/conda/lib/python3.10/site-packages (from python-dateutil>=2.8.2->pandas->datasets->trl==0.4.2.dev0) (1.16.0)
    Building wheels for collected packages: trl
      Building wheel for trl (setup.py) ... [?25ldone
    [?25h  Created wheel for trl: filename=trl-0.4.2.dev0-py3-none-any.whl size=67532 sha256=9628bbe170b3a3068af0886a070262e922ecee9bca5a6b211d49de4046cbe944
      Stored in directory: /tmp/pip-ephem-wheel-cache-68ep07wk/wheels/24/b4/20/2fa3a1e47c0411c39e198029315e3af2a2c1d59132913f136f
    Successfully built trl
    Installing collected packages: trl
    Successfully installed trl-0.4.2.dev0
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0mNote: you may need to restart the kernel to use updated packages.


<img src="data:image/svg+xml;base64,Cjxzdmcgd2lkdGg9IjgwMCIgaGVpZ2h0PSI1MCIgdmlld0JveD0iMCAwIDgwMCA1MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxkZWZzPgogICAgICAgIDxsaW5lYXJHcmFkaWVudCBpZD0iZmFkZUdyYWRpZW50IiB4MT0iMCIgeDI9IjEiPgogICAgICAgICAgICA8c3RvcCBvZmZzZXQ9IjAlIiBzdG9wLWNvbG9yPSIjRjBGMEYwIi8+CiAgICAgICAgICAgIDxzdG9wIG9mZnNldD0iMTAwJSIgc3RvcC1jb2xvcj0iI0YwRjBGMCIgc3RvcC1vcGFjaXR5PSIwIi8+CiAgICAgICAgPC9saW5lYXJHcmFkaWVudD4KICAgICAgICA8bWFzayBpZD0iZmFkZU1hc2siPgogICAgICAgICAgICA8cmVjdCB4PSIwIiB5PSIwIiB3aWR0aD0iNzUwIiBoZWlnaHQ9IjUwIiBmaWxsPSJ3aGl0ZSIvPgogICAgICAgICAgICA8cmVjdCB4PSI3NTAiIHk9IjAiIHdpZHRoPSI1MCIgaGVpZ2h0PSI1MCIgZmlsbD0idXJsKCNmYWRlR3JhZGllbnQpIi8+CiAgICAgICAgPC9tYXNrPgogICAgPC9kZWZzPgogICAgPHBhdGggZD0iTTI1LDUwIFEwLDUwIDAsMjUgTDUwLDMgTDk3LDI1IEw3OTcsMjUgTDc5Nyw1MCBMMjUsNTAgWiIgZmlsbD0iI0YwRjBGMCIgc3Ryb2tlPSIjRTBFMEUwIiBzdHJva2Utd2lkdGg9IjEiIG1hc2s9InVybCgjZmFkZU1hc2spIi8+Cjwvc3ZnPgo=" alt="Time alert close"/>

Import the necessary components. Some of them are new for this week, they will be discussed later in the notebook. 


```python
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, GenerationConfig
from datasets import load_dataset
from peft import PeftModel, PeftConfig, LoraConfig, TaskType

# trl: Transformer Reinforcement Learning library
from trl import PPOTrainer, PPOConfig, AutoModelForSeq2SeqLMWithValueHead
from trl import create_reference_model
from trl.core import LengthSampler

import torch
import evaluate

import numpy as np
import pandas as pd

# tqdm library makes the loops show a smart progress meter.
from tqdm import tqdm
tqdm.pandas()
```

<a name='2'></a>
## 2 - Load FLAN-T5 Model, Prepare Reward Model and Toxicity Evaluator

<a name='2.1'></a>
### 2.1 - Load Data and FLAN-T5 Model Fine-Tuned with Summarization Instruction

You will keep working with the same Hugging Face dataset [DialogSum](https://huggingface.co/datasets/knkarthick/dialogsum) and the pre-trained model [FLAN-T5](https://huggingface.co/docs/transformers/model_doc/flan-t5). 


```python
model_name="google/flan-t5-base"
huggingface_dataset_name = "knkarthick/dialogsum"

dataset_original = load_dataset(huggingface_dataset_name)

dataset_original
```


    Downloading readme:   0%|          | 0.00/4.65k [00:00<?, ?B/s]


    Downloading and preparing dataset csv/knkarthick--dialogsum to /root/.cache/huggingface/datasets/knkarthick___csv/knkarthick--dialogsum-cd36827d3490488d/0.0.0/6954658bab30a358235fa864b05cf819af0e179325c740e4bc853bcc7ec513e1...



    Downloading data files:   0%|          | 0/3 [00:00<?, ?it/s]



    Downloading data:   0%|          | 0.00/11.3M [00:00<?, ?B/s]



    Downloading data:   0%|          | 0.00/1.35M [00:00<?, ?B/s]



    Downloading data:   0%|          | 0.00/442k [00:00<?, ?B/s]



    Extracting data files:   0%|          | 0/3 [00:00<?, ?it/s]



    Generating train split: 0 examples [00:00, ? examples/s]



    Generating test split: 0 examples [00:00, ? examples/s]



    Generating validation split: 0 examples [00:00, ? examples/s]


    Dataset csv downloaded and prepared to /root/.cache/huggingface/datasets/knkarthick___csv/knkarthick--dialogsum-cd36827d3490488d/0.0.0/6954658bab30a358235fa864b05cf819af0e179325c740e4bc853bcc7ec513e1. Subsequent calls will reuse this data.



      0%|          | 0/3 [00:00<?, ?it/s]





    DatasetDict({
        train: Dataset({
            features: ['id', 'dialogue', 'summary', 'topic'],
            num_rows: 12460
        })
        test: Dataset({
            features: ['id', 'dialogue', 'summary', 'topic'],
            num_rows: 1500
        })
        validation: Dataset({
            features: ['id', 'dialogue', 'summary', 'topic'],
            num_rows: 500
        })
    })



The next step will be to preprocess the dataset. You will take only a part of it, then filter the dialogues of a particular length (just to make those examples long enough and, at the same time, easy to read). Then wrap each dialogue with the instruction and tokenize the prompts. Save the token ids in the field `input_ids` and decoded version of the prompts in the field `query`.

You could do that all step by step in the cell below, but it is a good habit to organize that all in a function `build_dataset`:


```python
def build_dataset(model_name,
                  dataset_name,
                  input_min_text_length, 
                  input_max_text_length):

    """
    Preprocess the dataset and split it into train and test parts.

    Parameters:
    - model_name (str): Tokenizer model name.
    - dataset_name (str): Name of the dataset to load.
    - input_min_text_length (int): Minimum length of the dialogues.
    - input_max_text_length (int): Maximum length of the dialogues.
        
    Returns:
    - dataset_splits (datasets.dataset_dict.DatasetDict): Preprocessed dataset containing train and test parts.
    """
    
    # load dataset (only "train" part will be enough for this lab).
    dataset = load_dataset(dataset_name, split="train")
    
    # Filter the dialogues of length between input_min_text_length and input_max_text_length characters.
    dataset = dataset.filter(lambda x: len(x["dialogue"]) > input_min_text_length and len(x["dialogue"]) <= input_max_text_length, batched=False)

    # Prepare tokenizer. Setting device_map="auto" allows to switch between GPU and CPU automatically.
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")
    
    def tokenize(sample):
        
        # Wrap each dialogue with the instruction.
        prompt = f"""
Summarize the following conversation.

{sample["dialogue"]}

Summary:
"""
        sample["input_ids"] = tokenizer.encode(prompt)
        
        # This must be called "query", which is a requirement of our PPO library.
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    # Tokenize each dialogue.
    dataset = dataset.map(tokenize, batched=False)
    dataset.set_format(type="torch")
    
    # Split the dataset into train and test parts.
    dataset_splits = dataset.train_test_split(test_size=0.2, shuffle=False, seed=42)

    return dataset_splits

dataset = build_dataset(model_name=model_name,
                        dataset_name=huggingface_dataset_name,
                        input_min_text_length=200, 
                        input_max_text_length=1000)

print(dataset)
```

    Found cached dataset csv (/root/.cache/huggingface/datasets/knkarthick___csv/knkarthick--dialogsum-cd36827d3490488d/0.0.0/6954658bab30a358235fa864b05cf819af0e179325c740e4bc853bcc7ec513e1)



    Filter:   0%|          | 0/12460 [00:00<?, ? examples/s]



    Downloading tokenizer_config.json:   0%|          | 0.00/2.54k [00:00<?, ?B/s]



    Downloading spiece.model:   0%|          | 0.00/792k [00:00<?, ?B/s]



    Downloading tokenizer.json:   0%|          | 0.00/2.42M [00:00<?, ?B/s]



    Downloading (…)cial_tokens_map.json:   0%|          | 0.00/2.20k [00:00<?, ?B/s]



    Map:   0%|          | 0/10022 [00:00<?, ? examples/s]


    DatasetDict({
        train: Dataset({
            features: ['id', 'dialogue', 'summary', 'topic', 'input_ids', 'query'],
            num_rows: 8017
        })
        test: Dataset({
            features: ['id', 'dialogue', 'summary', 'topic', 'input_ids', 'query'],
            num_rows: 2005
        })
    })


In the previous lab, you fine-tuned the PEFT model with summarization instructions. The training in the notebook was done on a subset of data. Then you downloaded the checkpoint of the fully trained PEFT model from S3. 

Let's load the same model checkpoint here:


```python
!aws s3 cp --recursive s3://dlai-generative-ai/models/peft-dialogue-summary-checkpoint/ ./peft-dialogue-summary-checkpoint-from-s3/ 
```

    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
    download: s3://dlai-generative-ai/models/peft-dialogue-summary-checkpoint/special_tokens_map.json to peft-dialogue-summary-checkpoint-from-s3/special_tokens_map.json
    download: s3://dlai-generative-ai/models/peft-dialogue-summary-checkpoint/adapter_config.json to peft-dialogue-summary-checkpoint-from-s3/adapter_config.json
    download: s3://dlai-generative-ai/models/peft-dialogue-summary-checkpoint/tokenizer_config.json to peft-dialogue-summary-checkpoint-from-s3/tokenizer_config.json
    download: s3://dlai-generative-ai/models/peft-dialogue-summary-checkpoint/tokenizer.json to peft-dialogue-summary-checkpoint-from-s3/tokenizer.json
    download: s3://dlai-generative-ai/models/peft-dialogue-summary-checkpoint/adapter_model.bin to peft-dialogue-summary-checkpoint-from-s3/adapter_model.bin


List the model item and check its size (it's less than 15 Mb):


```python
!ls -alh ./peft-dialogue-summary-checkpoint-from-s3/adapter_model.bin
```

    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
    -rw-r--r-- 1 root root 14M May 15  2023 ./peft-dialogue-summary-checkpoint-from-s3/adapter_model.bin


Prepare a function to pull out the number of model parameters (it is the same as in the previous lab):


```python
def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"\ntrainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"
```

Add the adapter to the original FLAN-T5 model. In the previous lab you were adding the fully trained adapter only for inferences, so there was no need to pass LoRA configurations doing that. Now you need to pass them to the constructed PEFT model, also putting `is_trainable=True`.


```python
lora_config = LoraConfig(
    r=32, # Rank
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM # FLAN-T5
)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name, 
                                              torch_dtype=torch.bfloat16)

peft_model = PeftModel.from_pretrained(model, 
                                       './peft-dialogue-summary-checkpoint-from-s3/', 
                                       lora_config=lora_config,
                                       torch_dtype=torch.bfloat16, 
                                       device_map="auto",                                       
                                       is_trainable=True)

print(f'PEFT model parameters to be updated:\n{print_number_of_trainable_model_parameters(peft_model)}\n')

```


    Downloading config.json:   0%|          | 0.00/1.40k [00:00<?, ?B/s]



    Downloading model.safetensors:   0%|          | 0.00/990M [00:00<?, ?B/s]



    Downloading generation_config.json:   0%|          | 0.00/147 [00:00<?, ?B/s]


    PEFT model parameters to be updated:
    
    trainable model parameters: 3538944
    all model parameters: 251116800
    percentage of trainable model parameters: 1.41%
    


In this lab, you are preparing to fine-tune the LLM using Reinforcement Learning (RL). RL will be briefly discussed in the next section of this lab, but at this stage, you just need to prepare the Proximal Policy Optimization (PPO) model passing the instruct-fine-tuned PEFT model to it. PPO will be used to optimize the RL policy against the reward model.


```python
ppo_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(peft_model,                                                               
                                                               torch_dtype=torch.bfloat16,
                                                               is_trainable=True)

print(f'PPO model parameters to be updated (ValueHead + 769 params):\n{print_number_of_trainable_model_parameters(ppo_model)}\n')
print(ppo_model.v_head)
```

    Detected kernel version 4.14.330, which is below the recommended minimum of 5.5.0; this can cause the process to hang. It is recommended to upgrade the kernel to the minimum version or higher.


    PPO model parameters to be updated (ValueHead + 769 params):
    
    trainable model parameters: 3539713
    all model parameters: 251117569
    percentage of trainable model parameters: 1.41%
    
    ValueHead(
      (dropout): Dropout(p=0.1, inplace=False)
      (summary): Linear(in_features=768, out_features=1, bias=True)
      (flatten): Flatten(start_dim=1, end_dim=-1)
    )


During PPO, only a few parameters will be updated. Specifically, the parameters of the `ValueHead`. More information about this class of models can be found in the [documentation](https://huggingface.co/docs/trl/main/en/models#trl.create_reference_model). The number of trainable parameters can be computed as $(n+1)*m$, where $n$ is the number of input units (here $n=768$) and $m$ is the number of output units (you have $m=1$). The $+1$ term in the equation takes into account the bias term.

Now create a frozen copy of the PPO which will not be fine-tuned - a reference model. The reference model will represent the LLM before detoxification. None of the parameters of the reference model will be updated during PPO training. This is on purpose.


```python
ref_model = create_reference_model(ppo_model)

print(f'Reference model parameters to be updated:\n{print_number_of_trainable_model_parameters(ref_model)}\n')
```

    Reference model parameters to be updated:
    
    trainable model parameters: 0
    all model parameters: 251117569
    percentage of trainable model parameters: 0.00%
    


Everything is set. It is time to prepare the reward model!

<a name='2.2'></a>
### 2.2 - Prepare Reward Model

**Reinforcement Learning (RL)** is one type of machine learning where agents take actions in an environment aimed at maximizing their cumulative rewards. The agent's behavior is defined by the **policy**. And the goal of reinforcement learning is for the agent to learn an optimal, or nearly-optimal, policy that maximizes the **reward function**. 

In the [previous section](#2.1) the original policy is based on the instruct PEFT model - this is the LLM before detoxification. Then you could ask human labelers to give feedback on the outputs' toxicity. However, it can be expensive to use them for the entire fine-tuning process. A practical way to avoid that is to use a reward model encouraging the agent to detoxify the dialogue summaries. The intuitive approach would be to do some form of sentiment analysis across two classes (`nothate` and `hate`) and give a higher reward if there is higher a chance of getting class `nothate` as an output. 

For example, we can mention that having human labelers for the entire finetuning process can be expensive. A practical way to avoid that is to use a reward model.

use feedback generated by a model

You will use [Meta AI's RoBERTa-based hate speech model](https://huggingface.co/facebook/roberta-hate-speech-dynabench-r4-target) for the reward model. This model will output **logits** and then predict probabilities across two classes: `nothate` and `hate`. The logits of the output `nothate` will be taken as a positive reward. Then, the model will be fine-tuned with PPO using those reward values.

Create the instance of the required model class for the RoBERTa model. You also need to load a tokenizer to test the model. Notice that the model label `0` will correspond to the class `nothate` and label `1` to the class `hate`.


```python
toxicity_model_name = "facebook/roberta-hate-speech-dynabench-r4-target"
toxicity_tokenizer = AutoTokenizer.from_pretrained(toxicity_model_name, device_map="auto")
toxicity_model = AutoModelForSequenceClassification.from_pretrained(toxicity_model_name, device_map="auto")
print(toxicity_model.config.id2label)
```


    Downloading tokenizer_config.json:   0%|          | 0.00/1.11k [00:00<?, ?B/s]



    Downloading vocab.json:   0%|          | 0.00/899k [00:00<?, ?B/s]



    Downloading merges.txt:   0%|          | 0.00/456k [00:00<?, ?B/s]



    Downloading (…)cial_tokens_map.json:   0%|          | 0.00/239 [00:00<?, ?B/s]



    Downloading config.json:   0%|          | 0.00/816 [00:00<?, ?B/s]



    Downloading model.safetensors:   0%|          | 0.00/499M [00:00<?, ?B/s]


    {0: 'nothate', 1: 'hate'}


Take some non-toxic text, tokenize it, and pass it to the model. Print the output logits, probabilities, and the corresponding reward that will be used for fine-tuning.


```python
non_toxic_text = "#Person 1# tells Tommy that he didn't like the movie."

toxicity_input_ids = toxicity_tokenizer(non_toxic_text, return_tensors="pt").input_ids

logits = toxicity_model(input_ids=toxicity_input_ids).logits
print(f'logits [not hate, hate]: {logits.tolist()[0]}')

# Print the probabilities for [not hate, hate]
probabilities = logits.softmax(dim=-1).tolist()[0]
print(f'probabilities [not hate, hate]: {probabilities}')

# get the logits for "not hate" - this is the reward!
not_hate_index = 0
nothate_reward = (logits[:, not_hate_index]).tolist()
print(f'reward (high): {nothate_reward}')
```

    logits [not hate, hate]: [3.114100694656372, -2.4896175861358643]
    probabilities [not hate, hate]: [0.9963293671607971, 0.003670616541057825]
    reward (high): [3.114100694656372]


Let's show a toxic comment.  This will have a low reward because it is more toxic.


```python
toxic_text = "#Person 1# tells that he will kill his friends."

toxicity_input_ids = toxicity_tokenizer(toxic_text, return_tensors="pt").input_ids

logits = toxicity_model(toxicity_input_ids).logits
print(f'logits [not hate, hate]: {logits.tolist()[0]}')

# Print the probabilities for [not hate, hate]
probabilities = logits.softmax(dim=-1).tolist()[0]
print(f'probabilities [not hate, hate]: {probabilities}')

# Get the logits for "not hate" - this is the reward!
nothate_reward = (logits[:, not_hate_index]).tolist() 
print(f'reward (low): {nothate_reward}')
```

    logits [not hate, hate]: [-0.6179004311561584, 0.3190023601055145]
    probabilities [not hate, hate]: [0.2815264165401459, 0.7184736728668213]
    reward (low): [-0.6179004311561584]


Setup Hugging Face inference pipeline to simplify the code for the toxicity reward model:


```python
device = 0 if torch.cuda.is_available() else "cpu"

sentiment_pipe = pipeline("sentiment-analysis", 
                          model=toxicity_model_name, 
                          device=device)
reward_logits_kwargs = {
    "top_k": None, # Return all scores.
    "function_to_apply": "none", # Set to "none" to retrieve raw logits.
    "batch_size": 16
}

reward_probabilities_kwargs = {
    "top_k": None, # Return all scores.
    "function_to_apply": "softmax", # Set to "softmax" to apply softmax and retrieve probabilities.
    "batch_size": 16
}

print("Reward model output:")
print("For non-toxic text")
print(sentiment_pipe(non_toxic_text, **reward_logits_kwargs))
print(sentiment_pipe(non_toxic_text, **reward_probabilities_kwargs))
print("For toxic text")
print(sentiment_pipe(toxic_text, **reward_logits_kwargs))
print(sentiment_pipe(toxic_text, **reward_probabilities_kwargs))
```

    Reward model output:
    For non-toxic text
    [{'label': 'nothate', 'score': 3.114100694656372}, {'label': 'hate', 'score': -2.4896175861358643}]
    [{'label': 'nothate', 'score': 0.9963293671607971}, {'label': 'hate', 'score': 0.003670616541057825}]
    For toxic text
    [{'label': 'hate', 'score': 0.3190023601055145}, {'label': 'nothate', 'score': -0.6179004311561584}]
    [{'label': 'hate', 'score': 0.7184736728668213}, {'label': 'nothate', 'score': 0.2815263867378235}]


The outputs are the logits for both `nothate` (positive) and `hate` (negative) classes. But PPO will be using logits only of the `nothate` class as the positive reward signal used to help detoxify the LLM outputs.


```python
print(sentiment_pipe(non_toxic_text, **reward_logits_kwargs))
print(sentiment_pipe(non_toxic_text, **reward_probabilities_kwargs))
```

    [{'label': 'nothate', 'score': 3.114100694656372}, {'label': 'hate', 'score': -2.4896175861358643}]
    [{'label': 'nothate', 'score': 0.9963293671607971}, {'label': 'hate', 'score': 0.003670616541057825}]



```python
print(sentiment_pipe(toxic_text, **reward_logits_kwargs))
print(sentiment_pipe(toxic_text, **reward_probabilities_kwargs))
```

    [{'label': 'hate', 'score': 0.3190023601055145}, {'label': 'nothate', 'score': -0.6179004311561584}]
    [{'label': 'hate', 'score': 0.7184736728668213}, {'label': 'nothate', 'score': 0.2815263867378235}]


<a name='2.3'></a>
### 2.3 - Evaluate Toxicity

To evaluate the model before and after fine-tuning/detoxification you need to set up the [toxicity evaluation metric](https://huggingface.co/spaces/evaluate-measurement/toxicity). The **toxicity score** is a decimal value between 0 and 1 where 1 is the highest toxicity.


```python
toxicity_evaluator = evaluate.load("toxicity", 
                                    toxicity_model_name,
                                    module_type="measurement",
                                    toxic_label="hate")
```


    Downloading builder script:   0%|          | 0.00/6.08k [00:00<?, ?B/s]


Try to calculate toxicity for the same sentences as in section [2.2](#2.2). It's no surprise that the toxicity scores are the probabilities of `hate` class returned directly from the reward model.


```python
toxicity_score = toxicity_evaluator.compute(predictions=[
    non_toxic_text
])

print("Toxicity score for non-toxic text:")
print(toxicity_score["toxicity"])

toxicity_score = toxicity_evaluator.compute(predictions=[
    toxic_text
])

print("\nToxicity score for toxic text:")
print(toxicity_score["toxicity"])
```

    Toxicity score for non-toxic text:
    [0.003670616541057825]
    
    Toxicity score for toxic text:
    [0.7184736728668213]


This evaluator can be used to compute the toxicity of the dialogues prepared in section [2.1](#2.1). You will need to pass the test dataset (`dataset["test"]`), the same tokenizer which was used in that section, the frozen PEFT model prepared in section [2.2](#2.2), and the toxicity evaluator. It is convenient to wrap the required steps in the function `evaluate_toxicity`. 


```python
def evaluate_toxicity(model, 
                      toxicity_evaluator, 
                      tokenizer, 
                      dataset, 
                      num_samples):
    
    """
    Preprocess the dataset and split it into train and test parts.

    Parameters:
    - model (trl model): Model to be evaluated.
    - toxicity_evaluator (evaluate_modules toxicity metrics): Toxicity evaluator.
    - tokenizer (transformers tokenizer): Tokenizer to be used.
    - dataset (dataset): Input dataset for the evaluation.
    - num_samples (int): Maximum number of samples for the evaluation.
        
    Returns:
    tuple: A tuple containing two numpy.float64 values:
    - mean (numpy.float64): Mean of the samples toxicity.
    - std (numpy.float64): Standard deviation of the samples toxicity.
    """

    max_new_tokens=100

    toxicities = []
    input_texts = []
    for i, sample in tqdm(enumerate(dataset)):
        input_text = sample["query"]

        if i > num_samples:
            break
            
        input_ids = tokenizer(input_text, return_tensors="pt", padding=True).input_ids
        
        generation_config = GenerationConfig(max_new_tokens=max_new_tokens,
                                             top_k=0.0,
                                             top_p=1.0,
                                             do_sample=True)

        response_token_ids = model.generate(input_ids=input_ids,
                                            generation_config=generation_config)
        
        generated_text = tokenizer.decode(response_token_ids[0], skip_special_tokens=True)
        
        toxicity_score = toxicity_evaluator.compute(predictions=[(input_text + " " + generated_text)])

        toxicities.extend(toxicity_score["toxicity"])

    # Compute mean & std using np.
    mean = np.mean(toxicities)
    std = np.std(toxicities)
        
    return mean, std
```

And now perform the calculation of the model toxicity before fine-tuning/detoxification:


```python
tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")

mean_before_detoxification, std_before_detoxification = evaluate_toxicity(model=ref_model, 
                                                                          toxicity_evaluator=toxicity_evaluator, 
                                                                          tokenizer=tokenizer, 
                                                                          dataset=dataset["test"], 
                                                                          num_samples=10)

print(f'toxicity [mean, std] before detox: [{mean_before_detoxification}, {std_before_detoxification}]')
```

    11it [00:27,  2.47s/it]

    toxicity [mean, std] before detox: [0.034846287136050785, 0.03141657928511457]


    


<a name='3'></a>
## 3 - Perform Fine-Tuning to Detoxify the Summaries
Optimize a RL policy against the reward model using Proximal Policy Optimization (PPO).

<a name='3.1'></a>
### 3.1 - Initialize `PPOTrainer`
 
For the `PPOTrainer` initialization, you will need a collator. Here it will be a function transforming the dictionaries in a particular way. You can define and test it:


```python
def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

test_data = [{"key1": "value1", "key2": "value2", "key3": "value3"}]
print(f'Collator input: {test_data}')
print(f'Collator output: {collator(test_data)}')
```

    Collator input: [{'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}]
    Collator output: {'key1': ['value1'], 'key2': ['value2'], 'key3': ['value3']}


Set up the configuration parameters. Load the `ppo_model` and the tokenizer. You will also load a frozen version of the model `ref_model`. The first model is optimized while the second model serves as a reference to calculate the KL-divergence from the starting point. This works as an additional reward signal in the PPO training to make sure the optimized model does not deviate too much from the original LLM.


```python
learning_rate=1.41e-5
max_ppo_epochs=1
mini_batch_size=4
batch_size=16

config = PPOConfig(
    model_name=model_name,    
    learning_rate=learning_rate,
    ppo_epochs=max_ppo_epochs,
    mini_batch_size=mini_batch_size,
    batch_size=batch_size
)

ppo_trainer = PPOTrainer(config=config, 
                         model=ppo_model, 
                         ref_model=ref_model, 
                         tokenizer=tokenizer, 
                         dataset=dataset["train"], 
                         data_collator=collator)
```

    Detected kernel version 4.14.330, which is below the recommended minimum of 5.5.0; this can cause the process to hang. It is recommended to upgrade the kernel to the minimum version or higher.


<a name='3.2'></a>
### 3.2 - Fine-Tune the Model

The fine-tuning loop consists of the following main steps:
1. Get the query responses from the policy LLM (PEFT model).
2. Get sentiments for query/responses from hate speech RoBERTa model.
3. Optimize policy with PPO using the (query, response, reward) triplet.

The operation is running if you see the following metrics appearing:
* `objective/kl`: minimize kl divergence,
* `ppo/returns/mean`: maximize mean returns,
* `ppo/policy/advantages_mean`: maximize advantages.

<img src="data:image/svg+xml;base64,Cjxzdmcgd2lkdGg9IjgwMCIgaGVpZ2h0PSIxMjUiIHZpZXdCb3g9IjAgMCA4MDAgMTI1IiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgogICAgPGRlZnM+CiAgICAgICAgPGxpbmVhckdyYWRpZW50IGlkPSJmYWRlR3JhZGllbnQiIHgxPSIwIiB4Mj0iMSI+CiAgICAgICAgICAgIDxzdG9wIG9mZnNldD0iMCUiIHN0b3AtY29sb3I9IiNGMEYwRjAiLz4KICAgICAgICAgICAgPHN0b3Agb2Zmc2V0PSIxMDAlIiBzdG9wLWNvbG9yPSIjRjBGMEYwIiBzdG9wLW9wYWNpdHk9IjAiLz4KICAgICAgICA8L2xpbmVhckdyYWRpZW50PgogICAgICAgIDxtYXNrIGlkPSJmYWRlTWFzayI+CiAgICAgICAgICAgIDxyZWN0IHg9IjAiIHk9IjAiIHdpZHRoPSI3NTAiIGhlaWdodD0iMTI1IiBmaWxsPSJ3aGl0ZSIvPgogICAgICAgICAgICA8cmVjdCB4PSI3NTAiIHk9IjAiIHdpZHRoPSI1MCIgaGVpZ2h0PSIxMjUiIGZpbGw9InVybCgjZmFkZUdyYWRpZW50KSIvPgogICAgICAgIDwvbWFzaz4KICAgIDwvZGVmcz4KICAgIDxwYXRoIGQ9Ik0zLDUwIEE1MCw1MCAwIDAgMSA1MywzIEw3OTcsMyBMNzk3LDk3IEw5Nyw5NyBMNTAsMTE1IEwzLDk3IFoiIGZpbGw9IiNGMEYwRjAiIHN0cm9rZT0iI0UwRTBFMCIgc3Ryb2tlLXdpZHRoPSIxIiBtYXNrPSJ1cmwoI2ZhZGVNYXNrKSIvPgogICAgPGNpcmNsZSBjeD0iNTAiIGN5PSI1MCIgcj0iMzAiIGZpbGw9IiM1N2M0ZjgiIHN0cm9rZT0iIzU3YzRmOCIgc3Ryb2tlLXdpZHRoPSIxIi8+CiAgICA8Y2lyY2xlIGN4PSI1MCIgY3k9IjUwIiByPSIyNSIgZmlsbD0iI0YwRjBGMCIvPgogICAgPGxpbmUgeDE9IjUwIiB5MT0iNTAiIHgyPSI1MCIgeTI9IjMwIiBzdHJva2U9IiM1N2M0ZjgiIHN0cm9rZS13aWR0aD0iMyIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+CiAgICA8bGluZSB4MT0iNTAiIHkxPSI1MCIgeDI9IjY1IiB5Mj0iNTAiIHN0cm9rZT0iIzU3YzRmOCIgc3Ryb2tlLXdpZHRoPSIzIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz4KICAgIDx0ZXh0IHg9IjEwMCIgeT0iMzQiIGZvbnQtZmFtaWx5PSJBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzMzMzMzMyI+VGhlIG5leHQgY2VsbCBtYXkgdGFrZSAyMC0zMCBtaW51dGVzIHRvIHJ1bi48L3RleHQ+Cjwvc3ZnPgo=" alt="Time alert open medium"/>


```python
output_min_length = 100
output_max_length = 400
output_length_sampler = LengthSampler(output_min_length, output_max_length)

generation_kwargs = {
    "min_length": 5,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True
}

reward_kwargs = {
    "top_k": None, # Return all scores.
    "function_to_apply": "none", # You want the raw logits without softmax.
    "batch_size": 16
}

max_ppo_steps = 10

for step, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    # Break when you reach max_steps.
    if step >= max_ppo_steps:
        break   

    prompt_tensors = batch["input_ids"]

    # Get response from FLAN-T5/PEFT LLM.
    summary_tensors = []

    for prompt_tensor in prompt_tensors:
        max_new_tokens = output_length_sampler()        
            
        generation_kwargs["max_new_tokens"] = max_new_tokens
        summary = ppo_trainer.generate(prompt_tensor, **generation_kwargs)
        
        summary_tensors.append(summary.squeeze()[-max_new_tokens:])
        
    # This needs to be called "response".
    batch["response"] = [tokenizer.decode(r.squeeze()) for r in summary_tensors]

    # Compute reward outputs.
    query_response_pairs = [q + r for q, r in zip(batch["query"], batch["response"])]    
    rewards = sentiment_pipe(query_response_pairs, **reward_kwargs)

    # You use the `nothate` item because this is the score for the positive `nothate` class.
    reward_tensors = [torch.tensor(reward[not_hate_index]["score"]) for reward in rewards]    

    # Run PPO step.
    stats = ppo_trainer.step(prompt_tensors, summary_tensors, reward_tensors)
    ppo_trainer.log_stats(stats, batch, reward_tensors)
    
    print(f'objective/kl: {stats["objective/kl"]}')
    print(f'ppo/returns/mean: {stats["ppo/returns/mean"]}')
    print(f'ppo/policy/advantages_mean: {stats["ppo/policy/advantages_mean"]}')
    print('-'.join('' for x in range(100)))
```

    0it [00:00, ?it/s]You're using a T5TokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
    1it [01:44, 104.73s/it]

    objective/kl: 31.117090225219727
    ppo/returns/mean: -0.5182645916938782
    ppo/policy/advantages_mean: -1.4600269437892166e-08
    ---------------------------------------------------------------------------------------------------


    2it [03:17, 97.80s/it] 

    objective/kl: 28.01114273071289
    ppo/returns/mean: -0.40017572045326233
    ppo/policy/advantages_mean: -7.117845868265249e-09
    ---------------------------------------------------------------------------------------------------


    3it [05:02, 101.06s/it]

    objective/kl: 30.86346435546875
    ppo/returns/mean: -0.4210904836654663
    ppo/policy/advantages_mean: -6.182602430016004e-09
    ---------------------------------------------------------------------------------------------------


    4it [06:37, 98.65s/it] 

    objective/kl: 31.220367431640625
    ppo/returns/mean: -0.6225794553756714
    ppo/policy/advantages_mean: 3.5729739167322805e-09
    ---------------------------------------------------------------------------------------------------


    5it [08:14, 98.03s/it]

    objective/kl: 29.64567756652832
    ppo/returns/mean: -0.4674806594848633
    ppo/policy/advantages_mean: 7.506906207765951e-09
    ---------------------------------------------------------------------------------------------------


    6it [09:46, 95.92s/it]

    objective/kl: 28.13495445251465
    ppo/returns/mean: -0.37669408321380615
    ppo/policy/advantages_mean: 1.786371939260789e-08
    ---------------------------------------------------------------------------------------------------


    7it [11:25, 96.89s/it]

    objective/kl: 28.820674896240234
    ppo/returns/mean: -0.46680739521980286
    ppo/policy/advantages_mean: 1.149169737146849e-08
    ---------------------------------------------------------------------------------------------------


    8it [12:55, 94.91s/it]

    objective/kl: 27.060161590576172
    ppo/returns/mean: -0.23012888431549072
    ppo/policy/advantages_mean: -2.648922858838887e-09
    ---------------------------------------------------------------------------------------------------


    9it [14:27, 94.00s/it]

    objective/kl: 21.506847381591797
    ppo/returns/mean: -0.1045561358332634
    ppo/policy/advantages_mean: -2.7389137624567184e-09
    ---------------------------------------------------------------------------------------------------


    10it [16:15, 97.53s/it]

    objective/kl: 35.32897186279297
    ppo/returns/mean: -0.6360522508621216
    ppo/policy/advantages_mean: 5.051141727108188e-09
    ---------------------------------------------------------------------------------------------------


    


<img src="data:image/svg+xml;base64,Cjxzdmcgd2lkdGg9IjgwMCIgaGVpZ2h0PSI1MCIgdmlld0JveD0iMCAwIDgwMCA1MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxkZWZzPgogICAgICAgIDxsaW5lYXJHcmFkaWVudCBpZD0iZmFkZUdyYWRpZW50IiB4MT0iMCIgeDI9IjEiPgogICAgICAgICAgICA8c3RvcCBvZmZzZXQ9IjAlIiBzdG9wLWNvbG9yPSIjRjBGMEYwIi8+CiAgICAgICAgICAgIDxzdG9wIG9mZnNldD0iMTAwJSIgc3RvcC1jb2xvcj0iI0YwRjBGMCIgc3RvcC1vcGFjaXR5PSIwIi8+CiAgICAgICAgPC9saW5lYXJHcmFkaWVudD4KICAgICAgICA8bWFzayBpZD0iZmFkZU1hc2siPgogICAgICAgICAgICA8cmVjdCB4PSIwIiB5PSIwIiB3aWR0aD0iNzUwIiBoZWlnaHQ9IjUwIiBmaWxsPSJ3aGl0ZSIvPgogICAgICAgICAgICA8cmVjdCB4PSI3NTAiIHk9IjAiIHdpZHRoPSI1MCIgaGVpZ2h0PSI1MCIgZmlsbD0idXJsKCNmYWRlR3JhZGllbnQpIi8+CiAgICAgICAgPC9tYXNrPgogICAgPC9kZWZzPgogICAgPHBhdGggZD0iTTI1LDUwIFEwLDUwIDAsMjUgTDUwLDMgTDk3LDI1IEw3OTcsMjUgTDc5Nyw1MCBMMjUsNTAgWiIgZmlsbD0iI0YwRjBGMCIgc3Ryb2tlPSIjRTBFMEUwIiBzdHJva2Utd2lkdGg9IjEiIG1hc2s9InVybCgjZmFkZU1hc2spIi8+Cjwvc3ZnPgo=" alt="Time alert close"/>

<a name='3.3'></a>
### 3.3 - Evaluate the Model Quantitatively

Load the PPO/PEFT model back in from disk and use the test dataset split to evaluate the toxicity score of the RL-fine-tuned model.


```python
mean_after_detoxification, std_after_detoxification = evaluate_toxicity(model=ppo_model, 
                                                                        toxicity_evaluator=toxicity_evaluator, 
                                                                        tokenizer=tokenizer, 
                                                                        dataset=dataset["test"], 
                                                                        num_samples=10)
print(f'toxicity [mean, std] after detox: [{mean_after_detoxification}, {std_after_detoxification}]')
```

    11it [00:23,  2.09s/it]

    toxicity [mean, std] after detox: [0.03039055831984363, 0.03521890552745902]


    


And compare the toxicity scores of the reference model (before detoxification) and fine-tuned model (after detoxification).


```python
mean_improvement = (mean_before_detoxification - mean_after_detoxification) / mean_before_detoxification
std_improvement = (std_before_detoxification - std_after_detoxification) / std_before_detoxification

print(f'Percentage improvement of toxicity score after detoxification:')
print(f'mean: {mean_improvement*100:.2f}%')
print(f'std: {std_improvement*100:.2f}%')
```

    Percentage improvement of toxicity score after detoxification:
    mean: 12.79%
    std: -12.10%


<a name='3.4'></a>
### 3.4 - Evaluate the Model Qualitatively

Let's inspect some examples from the test dataset. You can compare the original `ref_model` to the fine-tuned/detoxified `ppo_model` using the toxicity evaluator.

<img src="data:image/svg+xml;base64,Cjxzdmcgd2lkdGg9IjgwMCIgaGVpZ2h0PSIxMjUiIHZpZXdCb3g9IjAgMCA4MDAgMTI1IiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgogICAgPGRlZnM+CiAgICAgICAgPGxpbmVhckdyYWRpZW50IGlkPSJmYWRlR3JhZGllbnQiIHgxPSIwIiB4Mj0iMSI+CiAgICAgICAgICAgIDxzdG9wIG9mZnNldD0iMCUiIHN0b3AtY29sb3I9IiNGMEYwRjAiLz4KICAgICAgICAgICAgPHN0b3Agb2Zmc2V0PSIxMDAlIiBzdG9wLWNvbG9yPSIjRjBGMEYwIiBzdG9wLW9wYWNpdHk9IjAiLz4KICAgICAgICA8L2xpbmVhckdyYWRpZW50PgogICAgICAgIDxtYXNrIGlkPSJmYWRlTWFzayI+CiAgICAgICAgICAgIDxyZWN0IHg9IjAiIHk9IjAiIHdpZHRoPSI3NTAiIGhlaWdodD0iMTI1IiBmaWxsPSJ3aGl0ZSIvPgogICAgICAgICAgICA8cmVjdCB4PSI3NTAiIHk9IjAiIHdpZHRoPSI1MCIgaGVpZ2h0PSIxMjUiIGZpbGw9InVybCgjZmFkZUdyYWRpZW50KSIvPgogICAgICAgIDwvbWFzaz4KICAgIDwvZGVmcz4KICAgIDxwYXRoIGQ9Ik0zLDUwIEE1MCw1MCAwIDAgMSA1MywzIEw3OTcsMyBMNzk3LDk3IEw5Nyw5NyBMNTAsMTE1IEwzLDk3IFoiIGZpbGw9IiNGMEYwRjAiIHN0cm9rZT0iI0UwRTBFMCIgc3Ryb2tlLXdpZHRoPSIxIiBtYXNrPSJ1cmwoI2ZhZGVNYXNrKSIvPgogICAgPGNpcmNsZSBjeD0iNTAiIGN5PSI1MCIgcj0iMzAiIGZpbGw9IiM1N2M0ZjgiIHN0cm9rZT0iIzU3YzRmOCIgc3Ryb2tlLXdpZHRoPSIxIi8+CiAgICA8Y2lyY2xlIGN4PSI1MCIgY3k9IjUwIiByPSIyNSIgZmlsbD0iI0YwRjBGMCIvPgogICAgPGxpbmUgeDE9IjUwIiB5MT0iNTAiIHgyPSI1MCIgeTI9IjMwIiBzdHJva2U9IiM1N2M0ZjgiIHN0cm9rZS13aWR0aD0iMyIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+CiAgICA8bGluZSB4MT0iNTAiIHkxPSI1MCIgeDI9IjY1IiB5Mj0iNTAiIHN0cm9rZT0iIzU3YzRmOCIgc3Ryb2tlLXdpZHRoPSIzIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz4KICAgIDx0ZXh0IHg9IjEwMCIgeT0iMzQiIGZvbnQtZmFtaWx5PSJBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzMzMzMzMyI+VGhlIG5leHQgY2VsbCBtYXkgdGFrZSAyLTMgbWludXRlcyB0byBydW4uPC90ZXh0Pgo8L3N2Zz4K" alt="Time alert open medium"/>
​


```python
batch_size = 20
compare_results = {}

df_batch = dataset["test"][0:batch_size]

compare_results["query"] = df_batch["query"]
prompt_tensors = df_batch["input_ids"]

summary_tensors_ref = []
summary_tensors = []

# Get response from ppo and base model.
for i in tqdm(range(batch_size)):
    gen_len = output_length_sampler()
    generation_kwargs["max_new_tokens"] = gen_len
    
    summary = ref_model.generate(
        input_ids=torch.as_tensor(prompt_tensors[i]).unsqueeze(dim=0).to(device), 
        **generation_kwargs
    ).squeeze()[-gen_len:]
    summary_tensors_ref.append(summary)

    summary = ppo_model.generate(
        input_ids=torch.as_tensor(prompt_tensors[i]).unsqueeze(dim=0).to(device), 
        **generation_kwargs
    ).squeeze()[-gen_len:]
    summary_tensors.append(summary)

# Decode responses.
compare_results["response_before"] = [tokenizer.decode(summary_tensors_ref[i]) for i in range(batch_size)]
compare_results["response_after"] = [tokenizer.decode(summary_tensors[i]) for i in range(batch_size)]

# Sentiment analysis of query/response pairs before/after.
texts_before = [d + s for d, s in zip(compare_results["query"], compare_results["response_before"])]
rewards_before = sentiment_pipe(texts_before, **reward_kwargs)
compare_results["reward_before"] = [reward[not_hate_index]["score"] for reward in rewards_before]

texts_after = [d + s for d, s in zip(compare_results["query"], compare_results["response_after"])]
rewards_after = sentiment_pipe(texts_after, **reward_kwargs)
compare_results["reward_after"] = [reward[not_hate_index]["score"] for reward in rewards_after]
```

    100%|██████████| 20/20 [01:21<00:00,  4.06s/it]


<img src="data:image/svg+xml;base64,Cjxzdmcgd2lkdGg9IjgwMCIgaGVpZ2h0PSI1MCIgdmlld0JveD0iMCAwIDgwMCA1MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxkZWZzPgogICAgICAgIDxsaW5lYXJHcmFkaWVudCBpZD0iZmFkZUdyYWRpZW50IiB4MT0iMCIgeDI9IjEiPgogICAgICAgICAgICA8c3RvcCBvZmZzZXQ9IjAlIiBzdG9wLWNvbG9yPSIjRjBGMEYwIi8+CiAgICAgICAgICAgIDxzdG9wIG9mZnNldD0iMTAwJSIgc3RvcC1jb2xvcj0iI0YwRjBGMCIgc3RvcC1vcGFjaXR5PSIwIi8+CiAgICAgICAgPC9saW5lYXJHcmFkaWVudD4KICAgICAgICA8bWFzayBpZD0iZmFkZU1hc2siPgogICAgICAgICAgICA8cmVjdCB4PSIwIiB5PSIwIiB3aWR0aD0iNzUwIiBoZWlnaHQ9IjUwIiBmaWxsPSJ3aGl0ZSIvPgogICAgICAgICAgICA8cmVjdCB4PSI3NTAiIHk9IjAiIHdpZHRoPSI1MCIgaGVpZ2h0PSI1MCIgZmlsbD0idXJsKCNmYWRlR3JhZGllbnQpIi8+CiAgICAgICAgPC9tYXNrPgogICAgPC9kZWZzPgogICAgPHBhdGggZD0iTTI1LDUwIFEwLDUwIDAsMjUgTDUwLDMgTDk3LDI1IEw3OTcsMjUgTDc5Nyw1MCBMMjUsNTAgWiIgZmlsbD0iI0YwRjBGMCIgc3Ryb2tlPSIjRTBFMEUwIiBzdHJva2Utd2lkdGg9IjEiIG1hc2s9InVybCgjZmFkZU1hc2spIi8+Cjwvc3ZnPgo=" alt="Time alert close"/>

Store and review the results in a DataFrame


```python
pd.set_option('display.max_colwidth', 500)
df_compare_results = pd.DataFrame(compare_results)
df_compare_results["reward_diff"] = df_compare_results['reward_after'] - df_compare_results['reward_before']
df_compare_results_sorted = df_compare_results.sort_values(by=['reward_diff'], ascending=False).reset_index(drop=True)
df_compare_results_sorted
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>query</th>
      <th>response_before</th>
      <th>response_after</th>
      <th>reward_before</th>
      <th>reward_after</th>
      <th>reward_diff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Summarize the following conversation. #Person1#: How much are you asking for this? #Person2#: I'm offering them to you at 150 yuan a piece. Is that all right? #Person1#: Is tax already included in their price? #Person2#: Yes. Our price can't be matched. #Person1#: Would you consider a volume discount? #Person2#: If you buy 1, 000 or more, you'll get a 10 % discount. #Person1#: I'll accept your offer. Summary: &lt;/s&gt;</td>
      <td>&lt;pad&gt; #Person2# offers $14000 yuan for #Person1#'s order if #Person1# buys more than 1,000 yuan. #Person1# gives #Person2# the discount.&lt;/s&gt;</td>
      <td>&lt;pad&gt; #Person1# asked #Person2# to pay for the oranges with a volume discount.&lt;/s&gt;</td>
      <td>2.179607</td>
      <td>3.102147</td>
      <td>0.922540</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Summarize the following conversation. #Person1#: I'd like to have this cashed, please. #Person2#: Please put you name and address here. May I see your passport? #Person1#: Yes. #Person2#: How would you like it? #Person1#: Ten hundreds and ten twenties, and the rest in small change, please. #Person2#: OK. Here you are. Summary: &lt;/s&gt;</td>
      <td>&lt;pad&gt; #Person1# asks #Person2# for ten hundreds of hundreds and ten twenties. They take a fake bill but are set to double it when it gets there.&lt;/s&gt;</td>
      <td>&lt;pad&gt; #Person2# asks #Person1# to have the cash cashed and put #Person1#'s name and address and the net balance in small change.&lt;/s&gt;</td>
      <td>1.317862</td>
      <td>2.103319</td>
      <td>0.785457</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Summarize the following conversation. #Person1#: Could you help me, Sir? My flight got in 15 minutes ago. Everyone else has picked up the luggage but mine hasn't come through. #Person2#: I'm sorry, Madam, I'll go and find out if there is any more to come. Summary: &lt;/s&gt;</td>
      <td>&lt;pad&gt; #Person1# asks if #Person2# can help on his flight. Now the suggested list will be notified but #Person1# wants to&lt;/s&gt;</td>
      <td>&lt;pad&gt; #Person1#'s flight got in 15 minutes ago but the others haven't arrived yet. #Person2# will find out if there is a slight delay.&lt;/s&gt;</td>
      <td>2.193232</td>
      <td>2.873882</td>
      <td>0.680650</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Summarize the following conversation. #Person1#: Hello? #Person2#: Hello? #Person1#: Can I speak to Li Hong, please? #Person2#: Speaking. #Person1#: Hi, Li Hong. This is Alice. #Person2#: Hi, Alice. How are you? #Person1#: Not bad. Li Hong, I am sorry that I can't go to see Mrs. Brown with you tomorrow morning. My mother is ill. I must take care of her. #Person2#: I'm sorry to hear that. You'd better stay at home. After all, we can visit Mrs. Brown later #Person1#: OK. Bye - bye. #Person2#: ...</td>
      <td>&lt;pad&gt; Li Hong can't talk to Alice because his mother's ill. Alice mentions to Li Hong that they can visit Mrs. Brown later, and Li agrees to stay at home.&lt;/s&gt;</td>
      <td>&lt;pad&gt; Alice is seeing Li Hong on her way to see her mother. Li Hong invites Alice to visit her later.&lt;/s&gt;</td>
      <td>1.629495</td>
      <td>1.934588</td>
      <td>0.305093</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Summarize the following conversation. #Person1#: Here is the final draft of our contract. I'm glad that we have reached an agreement on almost every term in our trade. #Person2#: Yes, it seems to me we have come quite a long way. However, let me take a close look at the final draft. #Person1#: Do you have some points to bring up? #Person2#: Well, everything we've discussed seems to be here. #Person1#: Yes, including a description of the shirts you want to purchase this time, the total amount...</td>
      <td>&lt;pad&gt; #Person1# has a final draft of their contract. They are unhappy that they have reached an agreement. #Person1# gives some points to #Person2# who will be checking over their notes. #Person2# suggests signing the contract right now.&lt;/s&gt;</td>
      <td>&lt;pad&gt; #Person1# has sent away the draft of the contract. They look for it in the final draft. #Person2# likes it very much.&lt;/s&gt;</td>
      <td>2.805879</td>
      <td>3.090881</td>
      <td>0.285002</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Summarize the following conversation. #Person1#: Hello. I want to reconfirm our flight to London. #Person2#: Yes, sir. Did you call the airline? #Person1#: Yes, I did. But I couldn't communicate with them in English. They speak only Spanish. So I need your help. #Person2#: Certainly, sir. What is the flight number and when are you leaving? #Person1#: We are taking IB 385 to London tomorrow at 1 p. m. #Person2#: Oh, I see, sir. We have the airline office inside the hotel. They have an English...</td>
      <td>&lt;pad&gt; #Person1#'s problem can be reconfirmed with #Person2#'s assistance. #Person2# offers 36, their flight and tomorrow's departure times.&lt;/s&gt;</td>
      <td>&lt;pad&gt; #Person1# changes our flight reservation and #Person2# helps them. They spoke with the airline secretary in English.&lt;/s&gt;</td>
      <td>1.902260</td>
      <td>2.137513</td>
      <td>0.235253</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Summarize the following conversation. #Person1#: Could you help me figure out how to look for a job? #Person2#: We have lots of options, what type of job do you need? #Person1#: I want to work in an office. #Person2#: Do you want to work part-time or full-time? #Person1#: I want to work full-time. #Person2#: We have binders with local job listings or you can make use of the computers. OK? #Person1#: I am confused a bit but I am sure that I can figure it out. #Person2#: If you make an appoint...</td>
      <td>&lt;pad&gt; #Person1# asks #Person2# how to find a job. #Person2# tells #Person1# they have binders with local job listings and a counselor can give #Person1# information. #Person1# will make an appointment with a counselor.&lt;/s&gt;</td>
      <td>&lt;pad&gt; #Person1# wants to work in an office and some choices are dealt with. Assorted kinds of careers are offered in the job center. Hope you can help yourself and do the job center successfully.&lt;/s&gt;</td>
      <td>2.364130</td>
      <td>2.501021</td>
      <td>0.136891</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Summarize the following conversation. #Person1#: Let's take a coffee break, shall we? #Person2#: I wish I could, but I can't. #Person1#: What keeps you so busy? You've been sitting there for hours. You've got to walk around. You just can't stay on the computer forever. #Person2#: Well, I am up to my neck in work. I've got to finish this report. Sarah needs it by noon. I don't want to be scolded if I can't finish my work by the deadline. #Person1#: I understand that, but you'd feel better if ...</td>
      <td>&lt;pad&gt; #Person2# cannot take a coffee break because he has to finish a report. #Person1# asks #Person2# to take a short while before taking an earlier break so they can answer the question.&lt;/s&gt;</td>
      <td>&lt;pad&gt; #Person1# and #Person2# are taking a break because #Person2# is exhausted; they decide to take a coffee break.&lt;/s&gt;</td>
      <td>1.899473</td>
      <td>2.032474</td>
      <td>0.133001</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Summarize the following conversation. #Person1#: Oh, my God! What's this? #Person2#: What? #Person1#: Look! This window is open. #Person2#: Did you open it before we left? #Person1#: Are you kidding? It's winter. Why would I open it? #Person2#: I don't know. Wait. Is this yours? #Person1#: No! Oh, my God! Someone has broken into the house. #Person2#: It looks that way. That's probably why the door wasn't locked when we came in. #Person1#: I locked it when I left though. #Person2#: Yes, but t...</td>
      <td>&lt;pad&gt; #Person1# (Alan) and #Person2# are seating people in the house. Some-one broke into the house and was looking by the window and broke in with them. He recommends to look upstairs soon.&lt;/s&gt;</td>
      <td>&lt;pad&gt; There has been a burglar break in the house. Allen is not sure when the robber broke in and he gives Allen an explanation. Allen believes they could go upstairs.&lt;/s&gt;</td>
      <td>2.094149</td>
      <td>2.126117</td>
      <td>0.031968</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Summarize the following conversation. #Person1#: Today more and more families have personal computers. People have wider range of choice to communicate with the outside world. #Person2#: Right. With the establishment of Internet and a lot of web companies, people are getting more and more dependent on the web. #Person1#: One of the common uses of PC is that people can buy goods through it without going out to the physical stores. #Person2#: Can you tell me how it is done? #Person1#: If a cus...</td>
      <td>&lt;pad&gt; #Person2# and #Person1# have a wide range of choice to communicate with the outside world and travel for information thanks to the internet.&lt;/s&gt;</td>
      <td>&lt;pad&gt; #Person1# explains how a customer can buy a product on the Internet rather than going to physical stores and tells #Person2# how the procurement can be done.&lt;/s&gt;</td>
      <td>2.517467</td>
      <td>2.535873</td>
      <td>0.018407</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Summarize the following conversation. #Person1#: So how did you like the restaurant? #Person2#: Actually, it could have been better. #Person1#: What didn't you like about it? #Person2#: It is a new restaurant. I don't think they have their act together yet. #Person1#: What did you think about the food? #Person2#: I felt that the food was pretty mediocre. #Person1#: The service wasn't that great, either. #Person2#: I agree. The service was not good. #Person1#: Do you think that you want to tr...</td>
      <td>&lt;pad&gt; #Person1# thinks the restaurant should have been better and tries to concentrate on the service. But #Person2# complains that the food was quite mediocre and the service was barely good. #Person2# doesn't want to try another restaurant.&lt;/s&gt;</td>
      <td>&lt;pad&gt; #Person2# tells #Person1# that the restaurant is new, but the service is not good and #Person2# isn't satisfied with the restaurant.&lt;/s&gt;</td>
      <td>2.140657</td>
      <td>2.126348</td>
      <td>-0.014309</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Summarize the following conversation. #Person1#: I would like to order some internet today. #Person2#: What kind would you like? #Person1#: What kind of internet is there? #Person2#: You can get DEL or dial-up. #Person1#: Which of those two is best? #Person2#: I would recommend DEL. #Person1#: So that one better? #Person2#: It's better because it doesn't tie up the phone. #Person1#: What do you mean by that? #Person2#: DEL isn't connected through your phone line, but dial-up is. #Person1#: S...</td>
      <td>&lt;pad&gt; #Person2# recommends DEL because it doesn't tie up the phone. #Person1# couldn't use #Person1#s phone if #Person1# did both.&lt;/s&gt;</td>
      <td>&lt;pad&gt; #Person2# recommends DEL because it doesn't tie up the phone.&lt;/s&gt;</td>
      <td>2.446063</td>
      <td>2.366349</td>
      <td>-0.079714</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Summarize the following conversation. #Person1#: What can I do for you, madam? #Person2#: I'd like to buy a toy car for my son. #Person1#: How about this one? #Person2#: It looks nice. How much is it? #Person1#: They're three hundred dollars. #Person2#: Oh, I'm afraid it's too expensive. Can you show me something cheaper? #Person1#: OK, This one is one hundred and twenty. It's the cheapest here. #Person2#: OK, I'll take it. Here's the money. #Person1#: Thank you very much. Summary: &lt;/s&gt;</td>
      <td>&lt;pad&gt; #Person1#reaches #Person2# asking about the toy car though #Person2#'d like it worth three hundred dollars. #Person1# goes and finds two more with good reviews.&lt;/s&gt;</td>
      <td>&lt;pad&gt; #Person2# wants to buy a toy car. #Person1# offers #Person2# sell-sale one hundred and twenty. Their price is the cheapest.&lt;/s&gt;</td>
      <td>1.416189</td>
      <td>1.263100</td>
      <td>-0.153090</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Summarize the following conversation. #Person1#: Mom, I just finished my paper. Can you proofread it before I hand it in? #Person2#: Sure, let's take a look. Sweetie, this is terrific. Your ideas are so original. #Person1#: Thanks. #Person2#: I can tell you worked hard on it. #Person1#: I really did! I started thinking about what I wanted to say three weeks ago. #Person2#: Well, it was definitely worth all the time. #Person1#: Let's just hope my teacher agrees. Summary: &lt;/s&gt;</td>
      <td>&lt;pad&gt; Mrs. Butler has just finished her paper and leaves it as proofread. Mother loved her creative ideas and sends it back to her.&lt;/s&gt;</td>
      <td>&lt;pad&gt; #Person1# is reading #Person1#'s paper and looks at it, and she appreciates her original ideas.&lt;/s&gt;</td>
      <td>2.743418</td>
      <td>2.587080</td>
      <td>-0.156338</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Summarize the following conversation. #Person1#: I'm forming a music band. #Person2#: Do you already know how to play an instrument? #Person1#: Uh... Yeah! I'Ve told you a thousand times that I'm learning to play the drums. Now that I know how to play well, I would like to form a rock band. #Person2#: Aside from yourself, who are the other members of the band? #Person1#: We have a guy who plays guitar, and another who plays bass. Although we still haven't found anyone to be our singer. You t...</td>
      <td>&lt;pad&gt; #Person1# wants to form a rock rock band mainly because #Person1# is learning to play the drums, including the others, especially the guitars and bass. But #Person1# says they haven't found anybody to be their singer in the music band. #Person1# invites #Person2# to audition this weekend.&lt;/s&gt;</td>
      <td>&lt;pad&gt; #Person1# is forming a rock band. #Person2# prepares the music for #Person1# and tells him about what insects are in the band. They are considering auditioning external speakers and microphones for auditions.&lt;/s&gt;</td>
      <td>2.948375</td>
      <td>2.783118</td>
      <td>-0.165258</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Summarize the following conversation. #Person1#: Where shall I register, please? #Person2#: Here. Do you have a registration card? #Person1#: Yes. Here you are. #Person2#: Please register your information here and pay for it. And I'll make a medical record for you. #Person1#: OK. How much do I need to pay for the registration? #Person2#: Please pay ten yuan for the registration. #Person1#: Here is my money. #Person2#: This is your registration card. Please don't lose it and bring it whenever...</td>
      <td>&lt;pad&gt; #Person1# registers in the medical clinic and #Person2# asks #Person1# to register. #Person1# notes #Person1#'s ID card and asks how to get to the consulting room at the pharmacy.&lt;/s&gt;</td>
      <td>&lt;pad&gt; #Person1#'ll submit to theinstitution administration to get medical records at the clinic. #Person2# instructs #Person1# to walk down to the pharmacy and visit the training room to get medical records.&lt;/s&gt;</td>
      <td>1.770025</td>
      <td>1.491990</td>
      <td>-0.278035</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Summarize the following conversation. #Person1#: Amanda, how do you like this peaked cap? #Person2#: Didn't you say you want to buy a top hat? #Person1#: But I think this one fits me Well. Why don't you try on the sombrero in black? #Person2#: I don't like caps at all. Summary: &lt;/s&gt;</td>
      <td>&lt;pad&gt; Amanda doesn't like a cap but has some friends who like it but she doesn't like caps. Anna gives her a new one and she doesn't really like the cap.&lt;/s&gt;</td>
      <td>&lt;pad&gt; Amanda has become jealous of the peaked cap made by #Person1# but not her fellow customers.&lt;/s&gt;</td>
      <td>1.403095</td>
      <td>1.040424</td>
      <td>-0.362671</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Summarize the following conversation. #Person1#: Excuse me, could you tell me how to get to the Cross Bakery building? #Person2#: The Cross Bakery building? Oh sure. You're actually walking in the opposite direction. #Person1#: Oh, you're kidding! I thought I was heading east. #Person2#: No, east is the other direction. To get to the Bakery, you need to turn around and go three blocks to Broadway. When you get to the intersection of Broadway and Elm, you hang a left. Go straight down that st...</td>
      <td>&lt;pad&gt; #Person1# takes #Person2# turn around and go three blocks to Broadway and hang a left. #Person2# offers #Person1# the route to the Cross Bakery building in the opposite direction.&lt;/s&gt;</td>
      <td>&lt;pad&gt; #Person1# needs to go outside Broadway and Elm and tells #Person2# how to get to the Cross Bakery. #Person2# will show #Person1# the way.&lt;/s&gt;</td>
      <td>3.017745</td>
      <td>2.628376</td>
      <td>-0.389369</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Summarize the following conversation. #Person1#: It smells like an ashtray in here! #Person2#: Hi honey! What's wrong? Why do you have that look on your face? #Person1#: What's wrong? I thought we agreed that you were gonna quit smoking. #Person2#: No! I said I was going to cut down which is very different. You can't just expect me to go cold turkey overnight! #Person1#: Look, there are other ways to quit. You can try the nicotine patch, or nicotine chewing gum. We spend a fortune on cigaret...</td>
      <td>&lt;pad&gt; #Person1# goes away after her boyfriend starts a cheating. She tells #Person1# that she doesn't have the willpower to quit smoking and wants a divorce.&lt;/s&gt;</td>
      <td>&lt;pad&gt; #Person1# is confronted with #Person2#'s ashtray. #Person2#'s not going to quit smoking because #Person2# doesn't have the willpower to quit, so she won't be allowed.&lt;/s&gt;</td>
      <td>1.228137</td>
      <td>0.446271</td>
      <td>-0.781865</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Summarize the following conversation. #Person1#: Judy, what is everybody talking about? #Person2#: Haven't you heard? Richard was fired by our manager. #Person1#: You're kidding. It can't be true. #Person2#: Believe it or not. Everybody is talking about it in the company. #Person1#: Really? I'm surprised. #Person2#: Me too. Summary: &lt;/s&gt;</td>
      <td>&lt;pad&gt; Judy is wondering what everybody is talking about Richard who was fired in the company. Judy finds out it prints on the news. They are surprised.&lt;/s&gt;</td>
      <td>&lt;pad&gt; Judy says everyone talking about Richard fired because Richard was fired by their manager. Judy sounds surprised about that.&lt;/s&gt;</td>
      <td>1.809330</td>
      <td>0.996536</td>
      <td>-0.812794</td>
    </tr>
  </tbody>
</table>
</div>



Looking at the reward mean/median of the generated sequences you can observe a significant difference!
