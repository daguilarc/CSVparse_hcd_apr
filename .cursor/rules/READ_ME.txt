Private OMNI master rule (local only, gitignored)
=================================================

Before coding in this repo, read:
  .cursor/rules/omni-rule.mdc

That file is listed in .gitignore so it is never pushed to GitHub.
Cursor still loads it via alwaysApply: true in the file frontmatter.

Agents: if search does not find omni-rule.mdc (gitignored), read it
directly at the path above. Do not rely on stale User Rules snapshots.

Canonical global copy (outside this repo, for Cursor User Rules sync):
  ~/.cursor/rules/omni-rule.mdc
