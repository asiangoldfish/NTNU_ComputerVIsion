# Installation

To ensure the installation for your operating system goes smoothly, follow the
appropriate instructions.

- [Linux](#linux)
- [Windows](#windows)

# Linux
## Python
Some of the project's dependencies require Python version 3.9. While some make
it simple to install older versions, this doesn't apply to them all. If your 
package manager of choice doesn't contain Python3.9, then we suggest using
[pyenv](https://github.com/pyenv/pyenv).

1. Follow the [installation instructions](https://github.com/pyenv/pyenv?tab=readme-ov-file#automatic-installer) for pyenv.
2. Install Python3.9:
    ```
    pyenv install 3.9
    ```

    The project includes a [.python-version](../.python-version) file. This
    enables pyenv to automatically detect this project's local python version.
    Ensure the correct Python version is installed:
    ```sh
    pyenv local
    # 3.9

    python --version
    # Python 3.9.18
    ```
3. Once Python is installed, activate the virtual environment and install all
   dependencies. Please note this will take some time.
   ```
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

   *Optionally*, you can install the dependencies manually.
   ```
   pip install numpy pandas numpydoc keras torch tensorflow pillow opencv-python scikit-learn
   ```

# Windows
