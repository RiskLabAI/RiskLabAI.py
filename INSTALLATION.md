**Step 1: Create and activate the new environment**
First, just create the environment with the Python version you want.

```bash
conda create -n risklab python=3.9
conda activate risklab
```

**Step 2: Install PyTorch (Important)**
Next, install PyTorch from its dedicated channel. This is crucial for getting the correct version, especially if you need GPU (CUDA) support.

  * **For GPU (CUDA) support:** (This is the most common for ML)

    ```bash
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    ```

  * **For CPU-only:**

    ```bash
    conda install pytorch torchvision torchaudio cpuonly -c pytorch
    ```

**Step 3: Install from your `requirements.txt`**
Finally, now that you're inside the `risklab` environment, use `pip` to install everything listed in your file. `pip` will automatically find packages like `yfinance` and `ta` from PyPI (the Python Package Index) and will be smart enough to see that `torch` is already installed.

```bash
pip install -r requirements.txt
```