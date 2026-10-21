## Quick start

### 1. Install the skill

- Install Create continual learning Model skill here [create continual learning model skill](create-cl-model.zip)
- Extract the .zip -> Copy the `create-cl-model` folder into your Claude Code skills directory.

**Personal (all your projects)**

```bash
# macOS / Linux
mkdir -p ~/.claude/skills
cp -r pytorch-sqlite-memory ~/.claude/skills/
```

```powershell
# Windows (PowerShell)
New-Item -ItemType Directory -Force "$env:USERPROFILE\.claude\skills"
Copy-Item -Recurse pytorch-sqlite-memory "$env:USERPROFILE\.claude\skills\"
```

**One project only:** copy it to `<your-repo>/.claude/skills/` instead and commit it.

Claude Code picks up new skills in a running session. If the `skills` directory did not exist when the session started, restart Claude Code once. See the [Claude Code skills docs](https://code.claude.com/docs/en/skills) for details.

Requirements for the generated code: **PyTorch 2.x** and **NumPy**. SQLite comes with Python.

### 2. Ask for a model

A request that already contains the spec is built straight away:

> Create an MLP in PyTorch for a tabular dataset with 10 features and 3 classes. It should save its weights in SQLite under the memory name `churn_v1` and load them back later.

A vague request is answered with questions, in a single message, with a default offered for each:

> Create a continual learning model in PyTorch using external memory.
>
> *Claude:* Before I build anything, I need: which architecture (MLP for flat feature rows, LSTM for ordered windows, Transformer if you want one), classification or regression, the input and output sizes (or a path to your data), and a memory name (default `<task>_v1`). Reply "defaults" to accept my suggestions.

### 3. Run what it generated

Claude finishes by printing the full path of the folder. From there:

```bash
cd <the path it printed>
python verify_mlp.py     # checks save/load/versioning on throwaway data
python train.py          # after replacing the data placeholder with your own
```

`train.py` contains a placeholder (`X, y = ..., ...`) that you replace with your data loading.

---

