# Session 0 — Setting Up: Colab, Python, an IDE, and Virtual Environments

*From Pixels to Patients · AAPM 2026 (Session 0 — the setup prequel)*

Sessions 1–3 assume you can open a notebook and run a cell. This guide gets you there.
It is written for the person who has never installed Python, and it is safe to skip
entirely if `python --version` already prints something like `3.11.x` and you have an
environment you trust.

There are two ways to run everything in this repository:

| Path | What it costs you | Best for |
|------|-------------------|----------|
| **Google Colab** — run in the browser, nothing installed | A Google account. Files vanish when the runtime ends. | Following along live during the talk; trying Session 1 in five minutes. |
| **Local install** — Python + an IDE on your own machine | 20–30 minutes once. | Real work, large downloads, your own data, anything you want to keep. |

Start with Colab. Come back for the local install when you want the work to persist.

---

## Learning objectives

By the end of this guide you should be able to:

1. **Open** any notebook in this repository in Google Colab, straight from GitHub.
2. **Install** Python 3.10 or later on Windows, macOS, or Linux, and verify it.
3. **Install** an IDE — VS Code or PyCharm — and point it at your Python.
4. **Explain** what a virtual environment is, why every project should have one, and
   create, activate, and populate one.

---

## What's in this folder

| File | What it is |
|------|------------|
| `Guide.md` | This document. There is no notebook — this session *is* the setup. |

---

## Part 1 — Google Colab: run the notebooks online

Colab is a free Jupyter notebook environment that runs on Google's machines. Python is
already installed, the common scientific packages are already there, and you can request a
GPU. Nothing to install; you just need a Google account.

### Open a notebook from this repository

The fastest route is a direct link. Colab can open any public GitHub notebook by URL — the
pattern is `colab.research.google.com/github/<owner>/<repo>/blob/<branch>/<path>`:

- **Session 1** — [NSCLC_Radiomics_DICOM_to_NIfTI.ipynb](https://colab.research.google.com/github/brianmanderson/AAPM2026_PixelsToPatients/blob/main/Session1/NSCLC_Radiomics_DICOM_to_NIfTI.ipynb)
- **Session 2** — [NSCLC_NIfTI_to_PyTorch.ipynb](https://colab.research.google.com/github/brianmanderson/AAPM2026_PixelsToPatients/blob/main/Session2/NSCLC_NIfTI_to_PyTorch.ipynb)

Or navigate there yourself, which is worth doing once so you know how:

1. Go to <https://colab.research.google.com> and sign in.
2. **File → Open notebook**.
3. Choose the **GitHub** tab.
4. Paste `brianmanderson/AAPM2026_PixelsToPatients` into the search box and press Enter.
   (You do *not* need to authorize Colab against GitHub for a public repo.)
5. Pick a notebook from the list.

### Save your own copy — do this first

The notebook you just opened is read-only; it is GitHub's copy, not yours. Before you
change anything, **File → Save a copy in Drive**. That creates
`Copy of <notebook>.ipynb` in your Google Drive, and that copy is what you edit and keep.

### Running cells

Click a cell and press **Shift+Enter** to run it and advance. **Runtime → Run all** does
the whole notebook top to bottom. The first cell in our notebooks installs dependencies
with `pip` — expect it to take a minute, and expect a *"You must restart the runtime…"*
notice. Restart when asked (**Runtime → Restart session**), then carry on from the next
cell.

### Getting a GPU

Session 2 trains a model, which is much happier on a GPU. **Runtime → Change runtime
type → Hardware accelerator → GPU**, then Save. Free-tier GPUs are subject to
availability and usage limits; the Session 2 demo is sized to run on CPU too, just slower.

### What to watch out for in Colab

> **The filesystem is temporary.** When the runtime disconnects — you close the tab, or it
> idles out after ~90 minutes — every file you downloaded or generated is gone. Only the
> notebook in your Drive survives. Anything you want to keep must be written to Drive or
> downloaded before you leave.

A few consequences worth knowing up front:

- **Session 1's TCIA download** lands on the temporary disk. Fine for the 3-patient
  default; not where you want a 36 GB cohort.
- **Session 2 needs Session 1's output.** In Colab that means running Session 1 first *in
  the same runtime*, or mounting your Drive and pointing the notebook at a copy you saved
  there:
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  ```
- **`pip install` doesn't persist either.** Every fresh runtime reinstalls. This is why the
  first cell of each notebook is an install cell.
- **Never upload patient data to Colab.** It is a public cloud environment outside your
  institution's control. The public NSCLC-Radiomics cohort is fine — that is the whole
  reason these sessions use it. Your clinic's DICOM is not.

That last point is the honest limit of the Colab path: it is excellent for learning the
pipeline and useless for running it on real data. For that, install locally.

---

## Part 2 — Install Python

**Get Python 3.10 or later.** Anything from 3.10 through 3.12 will run this repository
comfortably. Avoid 3.9 and earlier (some dependencies have dropped it), and be a little
cautious with a version released in the last few months — PyTorch and SimpleITK sometimes
lag the newest release by a few months. If you want a boring, safe answer: **3.11**.

> **macOS and Linux already ship a `python3`.** Don't use it for your work and don't
> modify it — the operating system depends on it, and it is usually an old version.
> Install your own alongside it, which is what the instructions below do.

### Windows

1. Go to <https://www.python.org/downloads/windows/> and download the **Windows installer
   (64-bit)** for your chosen version.
2. Run it. On the first screen, **check "Add python.exe to PATH"** before clicking
   Install Now. This is the single most common setup mistake — without it, your terminal
   won't find `python` and nothing else in this guide works.
3. Open a new PowerShell window and verify:
   ```powershell
   python --version
   pip --version
   ```

If `python` opens the Microsoft Store instead of running, the PATH checkbox was missed.
Re-run the installer, choose **Modify**, and add it — or disable the Store alias under
*Settings → Apps → Advanced app settings → App execution aliases*.

### macOS

Either download the **macOS 64-bit universal2 installer** from
<https://www.python.org/downloads/macos/>, or use [Homebrew](https://brew.sh):

```bash
brew install python@3.11
```

Verify in Terminal:

```bash
python3 --version
pip3 --version
```

> On macOS and Linux the command is `python3`, not `python`. Once you activate a virtual
> environment (Part 4), plain `python` works and points at the right interpreter — which is
> one more small reason to always work inside one.

### Linux (Debian/Ubuntu)

```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip
python3.11 --version
```

The `python3.11-venv` package is easy to forget and you will need it in Part 4.

### Did it work?

Whatever your platform, this is the test:

```bash
python --version      # or python3 --version
```

You want to see `Python 3.10.x` or higher. If you see "command not found" or a 2.x
version, PATH is the culprit — revisit the platform steps above. **Close and reopen your
terminal after installing**; a shell started before the install won't have the new PATH.

---

## Part 3 — Install an IDE

You can technically run everything from a terminal, but an IDE gives you a debugger,
autocomplete, notebook rendering, and — most usefully — a visible indicator of *which*
Python environment you're in. Two good free options; pick one.

| | **VS Code** | **PyCharm** |
|---|---|---|
| **Cost** | Free | Community Edition free; Professional paid |
| **Feel** | Light, general-purpose editor; add what you need | Full Python IDE; batteries included |
| **Notebooks** | Excellent, via the Jupyter extension | Good in Community, best in Professional |
| **Environments** | Interpreter picker in the status bar | Strong built-in venv management |
| **Pick it if** | You want one editor for Python, Markdown, YAML, and everything else | You want Python tooling that just works without assembly |

Either is a fine choice and this repository assumes neither. If you're undecided, take
**VS Code** — it is what most of the examples and screenshots in the wild will match.

### VS Code

1. Download from <https://code.visualstudio.com/> and install.
2. Open the **Extensions** panel (Ctrl+Shift+X / Cmd+Shift+X) and install:
   - **Python** (Microsoft) — language support, debugging, environment selection
   - **Jupyter** (Microsoft) — run `.ipynb` notebooks right in the editor
3. **File → Open Folder** and choose your clone of this repository.
4. Open a notebook. Click **Select Kernel** at the top right and choose your interpreter
   (after Part 4, choose the one inside `.venv`).

The current interpreter shows in the bottom-right status bar. If a cell can't find a
package you know you installed, that indicator is the first thing to check — you are
almost certainly in the wrong environment.

### PyCharm

1. Download **Community Edition** from <https://www.jetbrains.com/pycharm/download/> and
   install.
2. **Open** your clone of this repository as a project.
3. PyCharm will usually detect a `.venv` in the project root and offer to use it — accept.
   To set it by hand: *Settings → Project → Python Interpreter → Add Interpreter → Add
   Local Interpreter*.
4. Open a notebook and run cells with the green arrows.

The active interpreter shows in the bottom-right corner here too.

---

## Part 4 — Virtual environments

This is the part people skip and later regret, so it gets the most space.

### The problem it solves

Install packages straight into your system Python and every project shares one pile of
dependencies. Session 1 wants one version of SimpleITK; some other project wants a
different one. `pip install` for the second quietly breaks the first. Six months later
your notebook fails and nothing about it changed — the environment underneath it did.

Worse for us specifically: a result you can't reproduce isn't a result. "It worked on my
machine in March" is not a methods section.

### The solution

A **virtual environment** is a self-contained folder holding its own Python interpreter and
its own `site-packages`. Activate it and `python` and `pip` resolve to *that* copy, not the
system one. Install what you like — nothing leaks in or out. Delete the folder and the
environment is gone without a trace.

One environment per project. They are cheap. Make them freely.

### Create one

From the repository root:

```bash
python -m venv .venv
```

That reads as: run the `venv` module with the current Python, and build an environment in a
new folder called `.venv`. The name is a convention, not a rule — but it's the one this
repo's `.gitignore` already excludes, so stick with it.

### Activate it

This is the step that differs by platform, and the one worth bookmarking:

| Shell | Command |
|-------|---------|
| **Windows PowerShell** | `.venv\Scripts\Activate.ps1` |
| **Windows cmd.exe** | `.venv\Scripts\activate.bat` |
| **macOS / Linux (bash, zsh)** | `source .venv/bin/activate` |
| **Git Bash on Windows** | `source .venv/Scripts/activate` |

Your prompt gains a `(.venv)` prefix. That prefix is the whole point — it's how you know
where you are. Confirm it took:

```bash
python -c "import sys; print(sys.executable)"
```

The path printed should be inside your `.venv` folder. If it isn't, activation didn't work.

> **PowerShell may refuse to run the activate script** with an execution-policy error.
> Allow signed local scripts for your user account, once:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
> Then activate again.

### Use it

With the environment active, install normally:

```bash
python -m pip install --upgrade pip
pip install DicomRTTool tcia_utils SimpleITK pandas matplotlib nibabel   # Session 1
pip install torch nibabel numpy pandas matplotlib                        # Session 2
```

Everything lands inside `.venv/`. Your system Python is untouched.

`pip install` inside a notebook cell installs into whatever environment the notebook's
kernel is running — which is why picking the right kernel (Part 3) matters. Our notebooks
install their own dependencies in the first cell, so if you've selected the `.venv` kernel,
running that cell does the right thing and you can skip the manual `pip install` above.

### Leave it

```bash
deactivate
```

The `(.venv)` prefix disappears. Nothing is lost — activate again whenever you come back.

### Record what's in it

To let someone else — or you, next year — rebuild the same environment:

```bash
pip freeze > requirements.txt
```

And to rebuild from that file:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Commit `requirements.txt`. **Never commit `.venv/`** — it's hundreds of megabytes of
platform-specific binaries that won't work on anyone else's machine anyway. This repo's
`.gitignore` already excludes it.

### Habits worth forming

- **Activate before you install.** If you forget, packages go to your system Python and the
  isolation you set up buys you nothing. Check for the `(.venv)` prefix — every time, until
  it's automatic.
- **One environment per project**, living in the project folder.
- **Delete and rebuild when confused.** An environment is disposable: `rm -rf .venv` (or
  delete the folder), recreate, reinstall. This is usually faster than debugging it.
- **`.venv` is not the only tool.** `conda`/`miniconda` is common in scientific Python and
  better at non-Python dependencies (CUDA, ITK); `uv` is a fast modern alternative that's
  gaining ground quickly. Both solve the same problem. `venv` ships with Python and needs
  nothing else installed, which is why it's what this guide teaches.

---

## Putting it together

The full local setup, start to finish:

```bash
git clone https://github.com/brianmanderson/AAPM2026_PixelsToPatients.git
cd AAPM2026_PixelsToPatients

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
```

Then open the folder in your IDE, select the `.venv` interpreter, open
`Session1/NSCLC_Radiomics_DICOM_to_NIfTI.ipynb`, and run the first cell. If it installs
without complaint, you're set up. Go to [`Session1/Guide.md`](../Session1/Guide.md).

> **Don't have git?** Install it from <https://git-scm.com/downloads>, or just download
> the repository as a ZIP from the GitHub page's green **Code** button. Git is worth
> learning — it's how you'll get updates to these materials — but it isn't required to
> follow along.

---

## When something goes wrong

| Symptom | Almost always |
|---------|---------------|
| `python: command not found` | Python isn't on PATH, or you're on macOS/Linux and want `python3`. Reopen your terminal first. |
| `python` opens the Microsoft Store | The "Add to PATH" box was unchecked at install (Part 2). |
| `ModuleNotFoundError` for something you just installed | Wrong environment. Check for `(.venv)`; in a notebook, check the selected kernel. |
| PowerShell won't run `Activate.ps1` | Execution policy — see the `Set-ExecutionPolicy` note in Part 4. |
| Colab: "you must restart the runtime" | Expected after the install cell. Restart, then continue from the next cell. |
| Colab: files I downloaded are gone | The runtime recycled. Expected — see Part 1. |
| `pip install` fails building a wheel | Often a too-new Python. Try 3.11. |

---

## Links

- **Google Colab** — <https://colab.research.google.com> · [welcome notebook](https://colab.research.google.com/notebooks/intro.ipynb)
- **Python downloads** — <https://www.python.org/downloads/>
- **VS Code** — <https://code.visualstudio.com/> · [Python in VS Code](https://code.visualstudio.com/docs/python/python-tutorial)
- **PyCharm Community** — <https://www.jetbrains.com/pycharm/download/>
- **Git** — <https://git-scm.com/downloads>
- **`venv` documentation** — <https://docs.python.org/3/library/venv.html>
- **Next:** Session 1 turns raw DICOM-RT into an analysis-ready dataset — [`Session1/Guide.md`](../Session1/Guide.md).
