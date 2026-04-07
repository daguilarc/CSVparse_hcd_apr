# Module 1 — Basic, Chapter 2: `pathlib.Path` and file locations

This chapter ties Python’s filesystem API to patterns you will see in `acs_v2.py`: building paths next to the script, checking whether files exist, reading text, and writing outputs. Everything here uses **`pathlib`** only (no `os.path` in the examples).

## Learning outcomes

By the end of this chapter you should be able to:

1. Build paths with the **`/` operator** on `Path` objects (and know when to wrap a string in `Path(...)` first).
2. Anchor paths to the **directory that contains the current module** using `Path(__file__).resolve().parent`.
3. Use **`.exists()`** before optional work and **`.read_text(...)`** to load file contents as a string.
4. **Write outputs** by composing a `Path` and calling **`.write_text(...)`** or passing that `Path` to libraries that accept path-like objects (as `acs_v2.py` does with pandas).

## Paths as objects: joining with `/`

`pathlib.Path` represents a filesystem path. The most readable way to combine a directory with a filename is the **`/` operator**, which returns a new `Path`:

```python
from pathlib import Path

root = Path("fake_project")
data_file = root / "inputs" / "table.csv"
# data_file is Path('fake_project/inputs/table.csv') on POSIX-style display
```

You cannot divide a string by a string; you must start from a `Path` (or use `Path("a") / "b"`). This matches how `acs_v2.py` builds names like `place_county_relationship.csv` next to the script (see below).

## Anchoring to the script: `Path(__file__).resolve().parent`

When you run a script, **relative paths** depend on the **current working directory**, which can change. Code in `acs_v2.py` avoids that ambiguity by anchoring to the file’s own folder.

- **`__file__`** is the path to the current `.py` file (as a string).
- **`.resolve()`** turns it into an absolute path with symlinks normalized.
- **`.parent`** is the directory containing that file.

So **`Path(__file__).resolve().parent`** is “the folder where this module lives,” which is ideal for config files, caches, and CSVs that ship with the project.

In the **`if __name__ == "__main__":`** block, the gazetteer (place–county relationship) file is defined exactly this way:

```3849:3851:work/CAY/CSVparse_hcd_apr/TableA2-ACSv2/acs_v2.py
    gazetteer_path = Path(__file__).resolve().parent / "place_county_relationship.csv"
    if (file_exists := gazetteer_path.exists()):
        df_rel = pd.read_csv(gazetteer_path, dtype=str)
```

A few lines later, when data is freshly downloaded, the same `gazetteer_path` is written back with **`df_rel.to_csv(gazetteer_path, index=False)`**—pandas accepts a `Path`, so the variable stays one coherent object from “exists?” through “read” and “save.”

## Checking existence and reading text: `.exists()` and `.read_text()`

**`.exists()`** returns `True` if the path points to something on disk (file or directory, depending on the path). The gazetteer logic uses it to choose between loading a cached CSV and downloading from the Census.

**`.read_text(encoding=..., errors=...)`** reads the whole file into a **single string**. That is the first real step in loading Table A2: the pipeline repairs quoting *before* pandas sees the CSV, so it needs the raw bytes decoded as text.

```471:471:work/CAY/CSVparse_hcd_apr/TableA2-ACSv2/acs_v2.py
    raw_text = Path(filepath).read_text(encoding="utf-8", errors="replace")
```

Even when callers pass a `Path` (for example `apr_path` built beside `__file__`), wrapping with **`Path(filepath)`** is harmless and ensures the object is a `Path`. The commented-out debug lines in the same function show **`.parent`** used to place sibling outputs such as `before_quote_fix.csv` next to the input file:

```484:486:work/CAY/CSVparse_hcd_apr/TableA2-ACSv2/acs_v2.py
    # affected_before.to_csv(Path(filepath).parent / "before_quote_fix.csv", index=False)
    # affected_after.to_csv(Path(filepath).parent / "after_quote_fix.csv", index=False)
    # pd.DataFrame([("rows_parsed_before_fix", len(df_before)), ("rows_parsed_after_fix", len(df_after)), ("affected_before", len(affected_before)), ("affected_after", len(affected_after)), ("opener_replacements", n_op), ("closer_replacements", n_cl)], columns=["metric", "value"]).to_csv(Path(filepath).parent / "recovery_summary.csv", index=False)
```

Tiny standalone examples (pathlib only, fake locations):

```python
from pathlib import Path

cfg = Path("fake_app") / "config.toml"
if cfg.exists():
    text = cfg.read_text(encoding="utf-8")
else:
    text = ""

report = Path("fake_app") / "out" / "summary.txt"
report.parent.mkdir(parents=True, exist_ok=True)
report.write_text("ok\n", encoding="utf-8")
```

Here **`.parent.mkdir(parents=True, exist_ok=True)`** creates `fake_app/out` if needed—a common companion pattern when writing outputs (the script uses the same idea in spirit when it ensures data lives next to the module).

## Writing outputs with `Path`

You can write strings with **`.write_text(data, encoding="utf-8")`**. For tables, libraries like pandas expose **`to_csv(path)`** where `path` may be a `Path`; the gazetteer block does that after download. Elsewhere in `acs_v2.py`, figures and CSVs are saved with paths built as **`Path(__file__).resolve().parent / "filename.ext"`**, keeping artifacts next to the source for this workflow.

Rule of thumb: **one `Path` variable** per artifact (input or output), built from **`Path(__file__).resolve().parent / "name"`** when the file belongs with the script, or from **`Path(filepath).parent / "sidecar.csv"`** when the output should sit beside a given input.

### Encoding and “where am I?”

Always pass an explicit **`encoding`** for text (`read_text` / `write_text`). The APR loader uses **`errors="replace"`** so a bad byte does not crash the whole run—use that only when you prefer a complete pass over strict fidelity.

Do not confuse **script directory** (`Path(__file__).resolve().parent`) with **process working directory** (`Path.cwd()`). Opening `"tablea2.csv"` with no folder means “look in whatever folder the terminal was in,” which breaks easily. `acs_v2.py` builds **`apr_path = Path(__file__).resolve().parent / "tablea2.csv"`** and passes that to **`load_a2_csv`**, so the location is stable.

## Chapter summary

- Use **`Path` / `"segment"`** to build paths clearly.
- Use **`Path(__file__).resolve().parent`** so file locations do not depend on where the shell was when you ran Python.
- Use **`.exists()`** to branch; use **`.read_text(...)`** when you need the full file as text (as in `load_a2_csv`).
- Use **`.write_text(...)`** or library writers (`to_csv`, etc.) with a composed **`Path`** for outputs.

## Exercises

1. **Joining** — Given `base = Path("fake_lab")`, write a single expression that produces `fake_lab/runs/run_01/metrics.json` using only `pathlib` (no string concatenation).

2. **Anchor** — Write a function `def sibling_data(name: str) -> Path:` that returns `Path(__file__).resolve().parent / name`. In a REPL, `__file__` is undefined; explain why you would test this function by running a small script file instead of typing it interactively.

3. **Read vs write** — Using only pathlib and fake paths under `Path("fake_demo")`, write a short script that: (a) checks whether `fake_demo/in.txt` exists; (b) if it exists, reads it with UTF-8 and writes the same content to `fake_demo/out.txt`; (c) if it does not exist, creates `fake_demo` if needed and writes `"missing\n"` to `out.txt`. (Hint: `mkdir(parents=True, exist_ok=True)`.)

---

*Course anchor file: `acs_v2.py` — patterns: `load_a2_csv` (`Path(filepath).read_text`, `Path(filepath).parent` in comments); `__main__` gazetteer (`gazetteer_path`, `.exists()`, `to_csv(gazetteer_path, ...)`).*
