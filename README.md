# pytorch-sqlite-memory

A [Claude Code](https://code.claude.com/docs/en/skills) skill that builds **PyTorch models with a persistent, named weight memory stored in SQLite**.

--------------------------------------------------------------------------
<img width="800" height="400" alt="✳ Continual learning model skill 21_10_2026 06_41_50" src="https://github.com/user-attachments/assets/360da0e4-969b-45eb-be51-8db9eb9865b8" />

--------------------------------------------------------------------------

Ask Claude Code to build continual learning Pytorch model, for example: MLP, Transformer or LSTM. It generates a small project in which the trained weights are saved under a *memory name* after training inside an SQlite database, and restored later by that name, so a model can skip retraining, keep refining the same weights, or start a new task from old knowledge.

It is inspired by the memory design of the NumPy-based **AbstractIntegratedModule** framework, but generates only the parts needed for a PyTorch model, not the whole framework.

- Note:
  - Create continual learning Model skill can be downloaded here [create continual learning model skill](create-cl-model.zip)
  - Follow instruction below


```python
database_init()
model = create_mlp_with_sqlite_memory(input_dim=10, output_dim=3)

if not load_memory(model, "churn_v1"):       # no memory yet: train once, then save
    model.fit_scaler(X_train)
    train_mlp(model, X_train, y_train)
    save_memory(model, "churn_v1")           # weights land in SQLite, after training
                                             # next run: load_memory() restores them, no retraining
```

---

## Contents

- [What the skill does](#what-the-skill-does)
- [Quick start](#quick-start)
- [The memory API](#the-memory-api)
- [Supported architectures](#supported-architectures)
- [How it works](#how-it-works)
- [Strengths](#strengths)
- [Tradeoffs and limitations](#tradeoffs-and-limitations)
- [Repository layout](#repository-layout)
- [Verifying a generated project](#verifying-a-generated-project)
- [Extending the skill](#extending-the-skill)
- [FAQ](#faq)
- [Status](#status)

---

## What the skill does

When you ask Claude Code for a PyTorch model with persistent weights, the skill makes it:

1. **Ask first.** It confirms five things before writing anything: architecture, task (classification or regression), input size, output size, and a memory name. Each can come from your message or from a dataset you point to. A vague request such as "create a continual learning model in PyTorch" gets questions, not code.
2. **Build only what you asked for.** One architecture per request, never all three "for completeness".
3. **Copy tested files, not retyped code.** The storage code and the model templates are bundled in the skill and copied into your project, then adapted to your data.
4. **Tell you where everything went.** The final message gives the absolute path of the new folder, lists the files, says where the database will be created, and gives the commands to run.

The generated project (default folder name `<memory_name>_<arch>/`) contains:

| File | Purpose |
|---|---|
| `memory.py` | `database_init`, `save_memory`, `load_memory`, `list_memories` |
| `model.py` | the PyTorch model and its `create_<arch>_with_sqlite_memory()` factory |
| `train.py` | training loop plus a load, train-if-needed, save, predict flow |
| `verify_<arch>.py` | smoke test for the memory behaviour |

---

## Quick start

### 1. Install the skill

- Install Create continual learning Model skill here [create continual learning model skill](create-cl-model.zip)
- Extract the .zip -> Copy the `create-cl-model` folder into your Claude Code skills directory.

**Personal (all your projects)**

```bash
# macOS / Linux
mkdir -p ~/.claude/skills
cp -r create-cl-model ~/.claude/skills/
```

```powershell
# Windows (PowerShell)
New-Item -ItemType Directory -Force "$env:USERPROFILE\.claude\skills"
Copy-Item -Recurse create-cl-model "$env:USERPROFILE\.claude\skills\"
```

**One project only:** copy it to `<your-folder>/.claude/skills/` instead and commit it.

> Note: pass the create-clm-model that has no recursion, meaning if inside the create-clm-model theres another create-clm-model folder, pass the create-clm-model folder (without another create-clm-model folder) that contains SKILL.md directly to `<your-folder>/.claude/skills/`.


Claude Code picks up new skills in a running session. If the `skills` directory did not exist when the session started, restart Claude Code once. See the [Claude Code skills docs](https://code.claude.com/docs/en/skills) for details.

Requirements for the generated code: **PyTorch 2.x** and **NumPy**. SQLite comes with Python.

- Quick start file can be downloaded here: [Quick start](Quick_start_cl_model.md)

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

## The memory API

All four functions live in `memory.py`.

| Function | What it does |
|---|---|
| `database_init()` | Creates `activity_log.db` and the weight table if missing. Safe to call on every run. |
| `save_memory(model, name)` | Stores the weights under `name`. Call it **once, after training finishes**. Returns the new row id. |
| `load_memory(model, name, version=None)` | Restores the active version, or a specific one with `version=<id>`. Returns `True` if loaded, `False` if no memory exists yet. |
| `list_memories(name=None, model_type=None)` | Lists saved versions, newest first, without loading weights. |

`MemoryIncompatibleError` is raised when a memory exists but does not match the model you built.

### Common patterns

**Train once, reuse later** (the flow in `train.py`)

```python
database_init()
model = create_mlp_with_sqlite_memory(input_dim=10, output_dim=3)

if not load_memory(model, "churn_v1"):
    model.fit_scaler(X_train)                 # only on a fresh model
    train_mlp(model, X_train, y_train)
    save_memory(model, "churn_v1")
```

**Keep refining the same task:** set `CONTINUE_TRAINING = True` in `train.py`. A new version is saved and the old one stays in history.

**Start a new task from old knowledge** (warm start)

```python
load_memory(model, "task_A")        # start from task A's weights
train_mlp(model, X_b, y_b)
save_memory(model, "task_B")        # task A stays exactly as it was
```

The two tasks must share the same architecture config.

**See and restore earlier versions**

```python
for v in list_memories("churn_v1"):
    print(v["id"], v["is_active"], v["created_at"], v["size_bytes"])

load_memory(model, "churn_v1", version=3)   # load that exact old version
save_memory(model, "churn_v1")              # rollback: stores it as the new active version
```

**Inspect the database directly**

```bash
sqlite3 activity_log.db "SELECT id, memory_name, model_type, is_active, created_at FROM torch_weight_storage;"
```

---

## Supported architectures

Exactly three, kept deliberately plain.

| | **MLP** | **Transformer** | **LSTM** |
|---|---|---|---|
| Built from | `nn.Linear` stack | `nn.TransformerEncoder` + linear head | `nn.LSTM` + linear head |
| Input `X` | `(N, F)` | `(N, F)` (each feature is a token) or `(N, T, F)` (each step is a token) | `(N, T, F)`, or `(N, T)` for one feature |
| Tasks | classification, regression | classification, regression | classification, regression |
| Initialisation | He or Xavier/Glorot, chosen by activation | Xavier/Glorot | Xavier input weights, orthogonal recurrent weights, forget-gate bias 1 |
| Good for | tabular features | small tabular or windowed data when you specifically want attention | ordered sequences, time series, sensor windows |
| Reference | [`mlp.md`](pytorch-sqlite-memory/references/mlp.md) | [`transformer.md`](pytorch-sqlite-memory/references/transformer.md) | [`lstm.md`](pytorch-sqlite-memory/references/lstm.md) |

Common to all three: scaler statistics are stored with the weights, regression predictions come back in original units, and `predict()` returns class probabilities for classification.

CNNs, GRUs, graph networks and other architectures are not supported. If you ask for one, the skill says so rather than silently substituting.

---

## How it works

**Memory name is the key.** One name per task. Training a new task under a new name leaves earlier memories untouched. Re-using a name means "same task, keep refining".

**Save and load are separate calls.** `load_memory` returns `False` when nothing is saved yet, and the script then trains and saves. Weights are saved once after training, never per epoch, so a half-trained model can never become the memory a later run trusts.

**Nothing is deleted.** Each save marks the previous version `is_active = 0` and inserts a new active row. The full history stays in the file.

**One small SQLite table.**

```
torch_weight_storage(id, memory_name, model_type, config, weights, is_active, created_at)
```

`weights` is a `torch.save` of the `state_dict`. `config` is JSON describing the structure those weights belong to.

**Loading is checked and all-or-nothing.** `config` holds structure only (sizes, layers, heads, task, class names). If it differs from the model you built, `load_memory` raises `MemoryIncompatibleError` and leaves the model untouched, instead of returning `False`. Returning `False` would send the script back into training, and the following save would quietly retire a perfectly good memory just because someone changed a hidden size. Shapes and parameter names are checked before any weight is copied.

**Safety details**

- `torch.load(..., weights_only=True)`, so a database file cannot execute pickled code.
- `save_memory` refuses NaN or Inf weights, so a diverged run cannot overwrite a good memory.
- All SQL uses `?` placeholders.
- The database file sits **next to `memory.py`**, not in the current directory, so launching a script from another folder cannot create a second, empty database.
- Scaler mean and std are registered buffers, so they travel inside the `state_dict` and cannot drift away from the weights.

**The skill's own guardrails**

- A hard "confirm the spec first" gate before any file is created.
- One architecture per request, with explicit handling when you name two.
- A new folder by default, and never overwriting an existing `memory.py`, `model.py` or `train.py`.
- Bundled, tested files are copied with `cp`, not regenerated from memory.

---

## Strengths

- **Small and readable.** A few hundred lines of plain PyTorch and `sqlite3`. No framework to learn, no extra dependencies beyond PyTorch and NumPy.
- **Auditable.** Every version is a row you can list, inspect, load or roll back to, with plain SQL if you like.
- **Safe by default.** Incompatible, corrupt or diverged memories are refused loudly instead of being half-loaded or silently overwritten.
- **Consistent across architectures.** The same four functions and the same `config` contract serve the MLP, Transformer and LSTM.
- **Predictable generation.** Because the code is bundled and copied, two runs of the skill give the same storage code, and the verify scripts check it.
- **Standard PyTorch models.** Nothing proprietary, so you can swap in your own layers, optimiser or loss.
- **Per-task memories.** Separate names make it easy to keep one set of weights per task or customer, and to warm-start one from another.
- **Self-contained database.** One `.db` file next to your code. Copy it to move your memories.

---

## Tradeoffs and limitations

Read these before relying on it.

**It is persistence, not a continual-learning algorithm.**
SQLite memory preserves *saved* weights. It does not stop a model from forgetting task A while it trains on task B under the same memory name. The per-task memory name is what protects old knowledge. There is no replay buffer, regularisation (such as EWC) or other anti-forgetting method.

**Weights only.**
Optimiser state is not saved, so continued training restarts the optimiser (the template lowers the learning rate when continuing). Training history and metrics are not stored either.

**Structure must match exactly.**
Changing hidden sizes, layer counts, heads, input or output size means a new memory name. This is deliberate, and it fails loudly, but it is less flexible than loading weights loosely.

**Every save is a full copy.**
Each version stores the complete `state_dict` as an uncompressed BLOB. There is no pruning or compression, so the database grows with every save. This suits small and medium models. SQLite's default limit for one BLOB is about 1 GB, and storing very large models in SQLite is the wrong tool.

**Single-user design.**
There is no locking strategy beyond SQLite's defaults. It is not designed for many processes writing the same database at once.

**Not interchangeable with the NumPy framework.**
The model structures differ, and the weights live in a separate table (`torch_weight_storage`), so the two do not read each other's data.

**The models are intentionally plain.**
- *Transformer:* numeric `X` only (no tokenizer, vocabulary or embedding table), a fixed sequence length, no padding mask. It is data hungry, so on small tabular datasets an MLP often matches or beats it. The framework's alpha blending, fixed and dynamic backward modes, and learned-attention bookkeeping were deliberately dropped.
- *LSTM:* one prediction per sequence from the final hidden state. No padding or masking for ragged sequences, and no per-step outputs.
- *MLP:* no QAT or geometric weight shaping from the framework, only standard He or Xavier initialisation.

**Framework features that are not ported.**
P2P agents, async prediction managers, ensembles, explainability, SNN, kNN-augmented transformers, uncertainty estimates for the LSTM, and cross-session export.

**It is a prompt-based skill.**
Claude follows the instructions in `SKILL.md`, and behaviour can still vary between runs and models. The bundled code and the verify scripts are the safety net, not the prompt.

**It asks before it builds.**
That costs one extra turn on vague requests. It was a deliberate trade for not getting arbitrary files and setups.

---

## Repository layout

```
.
├── README.md
└── pytorch-sqlite-memory/            # the skill: copy this folder into .claude/skills/
    ├── SKILL.md                      # instructions Claude Code follows
    ├── references/                   # explanations: contract, parameters, gotchas
    │   ├── mlp.md
    │   ├── transformer.md
    │   └── lstm.md
    ├── templates/                    # tested code that gets copied into projects
    │   ├── memory.py                 # shared by all architectures
    │   ├── mlp/          model_mlp.py, training_mlp.py
    │   ├── transformer/  model_transformer.py, train_tf.py
    │   └── lstm/         model_lstm.py, training_lstm.py
    └── scripts/                      # smoke tests
        ├── verify_mlp.py
        ├── verify_tf.py
        └── verify_lstm.py
```

---

## Verifying a generated project

Each `verify_<arch>.py` uses a throwaway database, so it never touches your real memories. It checks that:

- weights, including scaler statistics, survive a save and load exactly;
- a different seed or dropout still loads (they are not structure);
- a structural mismatch raises `MemoryIncompatibleError` and leaves the model untouched;
- saving twice keeps history with exactly one active version;
- `list_memories` and `load_memory(version=...)` work, and a rollback makes the old weights active;
- regression predictions come back in original units;
- a wrong input shape raises instead of being sliced or padded.

Run it from the folder that holds `memory.py`, `model.py` and `train.py`:

```bash
python verify_mlp.py        # or verify_tf.py / verify_lstm.py
```

A quick way to try the MLP yourself is scikit-learn's `make_classification`: train once, run the script again, and the second run should print `memory loaded, skipping training` with the same accuracy.

---

## Extending the skill

- **Change a default or a model:** edit the file under `templates/<arch>/`. Claude copies from there.
- **Add an architecture:** add `templates/<arch>/model.py` and `training_<arch>.py` (the model must expose `model.config` with `architecture` and structure-only keys), a `references/<arch>.md`, and a `scripts/verify_<arch>.py`, then add it to the routing table in `SKILL.md`.
- **Move the database:** change `DB_PATH` in `memory.py`, or pass `db_path=` to any of the four functions.

Keep the model's `config` to structure only. Training hyperparameters such as learning rate or dropout belong outside it, so you can tune them between runs and still load the same memory.

---

## FAQ

**Where is my database?**
`activity_log.db`, created next to `memory.py` the first time `database_init()` runs. Claude's final message tells you the folder.

**Why did Claude ask me questions instead of building?**
Your request did not state the architecture, task, sizes and memory name. This is intended. Answer "defaults" to accept its suggestions (for example an MLP for flat feature rows, with a name like `<task>_v1`).

**Why did loading fail with `MemoryIncompatibleError`?**
The model you built differs structurally from the one that was saved (for example a different hidden size). Use a new memory name, or rebuild the model with the saved config, which `list_memories()` shows.

**Can I keep several tasks in one database?**
Yes. Each memory name is independent, and the same name can hold different architectures separately.

**Does it work with a GPU?**
Yes, for training. The model is built and loaded on CPU. Pass `device="cuda"` to the training function (for example `train_mlp(model, X, y, device="cuda")`); it moves the model to the GPU for training and back to CPU afterwards, so saving always works from CPU tensors. The default is `"cpu"`.

**I changed `memory.py` and now the verify script fails on import.**
The files are meant to be used together. Copy `memory.py`, `model.py`, `train.py` and the verify script from the same version of the skill.

---

## Status

Developed and tested by the author on Windows using synthetic data. The storage logic also has unit tests against a stand-in for PyTorch. There is no CI, and the models have not been benchmarked on real-world datasets. Run the verify scripts after generating a project, and treat the templates as a starting point for your own data.

## Credits

Inspired by the memory design of the NumPy-based **AbstractIntegratedModule** framework.
