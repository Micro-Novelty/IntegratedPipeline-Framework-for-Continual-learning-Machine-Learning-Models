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
