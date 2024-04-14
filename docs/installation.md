# Installation

To ensure a successful installation, follow please follow the appropriate instructions for your platform.

- [Linux](#linux)
- [Windows](#windows)

We do not support MacOS.

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
   ```

   Install the dependencies.
   ```
   pip install numpy pandas numpydoc keras torch tensorflow pillow opencv-python scikit-learn ultralytics
   ```

   We do not provide *requirements.txt*, because dependencies differ across platforms and system configurations (like graphics adaptors).

# Windows

Some of the project's dependencies require Python version 3.9.
Follow the guide below for installation.

## Prerequisites

- Git installed on your Windows system. You can download and install Git from [git-scm.com](https://git-scm.com/).

## Step-by-Step Installation Guide for Python 3.9

1. **Install pyenv**:
   - Open a command prompt (cmd) as an administrator.
   - Run the following command to install pyenv-win:
     ```
     git clone https://github.com/pyenv-win/pyenv-win.git "%USERPROFILE%\.pyenv"
     ```
   - Add `%USERPROFILE%\.pyenv\pyenv-win\bin` to your system's PATH environment variable.

2. **Install Python 3.9 with pyenv**:
   - Open a new command prompt.
   - Run the following command to install Python 3.9:
     ```
     pyenv update
     pyenv install 3.9
     ```

3. **Set Python 3.9 as the local version**:
   - Navigate to your project directory in the command prompt.
   - Run the following command to set Python 3.9 as the local version for your project:
     ```
     pyenv local 3.9
     ```

4. **Verify Python installation**:
   - Run the following command to verify that Python 3.9 is installed correctly:
     ```
     python --version
     ```

     If the Python version is not 3.9.x, then it may be necessary to add the new
     Python version to PATH. Thus, add `%USERPROFILE%\.pyenv\pyenv-win\shims`
     to PATH. You should now be able to verify the Python version as 3.9.x:
     ```
     python39 --version
     ```

5. **Create and activate a virtual environment**:
   - Navigate to your project directory in the command prompt.
   - Run the following commands to create and activate a virtual environment named `.venv`:
     ```ps1
     python -m venv .venv # or python39
     .venv\Scripts\activate
     ```

6. **Install project dependencies**:
   - Ensure you're still in your project directory and your virtual environment is activated.
   - Run the following command to install the dependencies:
   
      ```
      pip install numpy pandas numpydoc keras torch tensorflow pillow opencv-python scikit-learn ultralytics
      ```

      We do not provide *requirements.txt*, because dependencies differ across platforms and system configurations (like graphics adaptors).