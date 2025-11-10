### 1\. Create the New Environment

Open your terminal or Anaconda Prompt. This command will create a new, empty environment named `risklab` using a stable Python version (e.g., 3.10).

```bash
conda create -n risklab python=3.10 -y
```

### 2\. Activate the Environment

You must activate the environment to install packages into it and use it.

```bash
conda activate risklab
```

Your terminal prompt should now change to show `(risklab)` at the beginning.

### 3\. Install Your Project's Dependencies

Navigate to the root directory of your `RiskLabAI.py` project (the one containing `requirements.txt`). This command will read your `requirements.txt` file and install all the necessary packages.

```bash
# Navigate to your project folder first
cd /path/to/your/RiskLabAI.py

# Install all packages from your requirements file
pip install -r requirements.txt
```

### 4\. Install Your Library in "Editable" Mode

This is a crucial step for development and testing. It links your `RiskLabAI` source code to the environment, which allows your test suite to import your library as if it were officially installed.

From the same root directory (where your `pyproject.toml` is), run:

```bash
pip install -e .
```

### 5\. Run Your Tests

Now you are all set. The standard way to run your test suite is by using `pytest`. If `pytest` wasn't included in your `requirements.txt`, you can install it:

```bash
pip install pytest
```

Then, simply run the following command from your project's root directory:

```bash
pytest
```
If you want to be fast, you can tell `pytest` to ignore that specific directory when you run your tests:

```bash
pytest --ignore=RiskLabAI/pde
```