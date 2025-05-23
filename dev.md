## Notes for the development

### Project initialization

```bash
[🎩 lgao@lins-p1 CLIver]$ uv venv
Using CPython 3.12.8
Creating virtual environment at: .venv
Activate with: source .venv/bin/activate
[🎩 lgao@lins-p1 CLIver]$ uv sync --all-extras --dev --locked
Resolved 15 packages in 1.37s
      Built cliver @ file:///home/lgao/sources/personal/CLIver
Prepared 1 package in 554ms
Installed 1 package in 0.81ms
 + cliver==0.1.0 (from file:///home/lgao/sources/personal/CLIver)
```

#### Ruff formatter

- `uv run ruff format --check`

#### Ruff linter

- `uv run ruff check`

### Run application using uv

```bash
uv run cliver
```

### Test application

```bash
uv run pytest
```

### Release application

### Add a new Command

- Create a `xxx.py` under `src/cliver/commands/`
- Define a click.Group having the same name as the file:
  like the `config` in the `config.py`:

```python
@click.group(name="config", help="Manage configuration settings.")
def config():
    """
    Configuration command group.
    This group contains commands to manage configuration settings.
    """
    pass
```

- If there is a `def post_group()` defined, it will be called.
