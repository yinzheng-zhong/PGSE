# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project rules

- **Comments say what the code does, not why.** No rationale, history, or bug/environment
  background in comments or docstrings — that belongs in the commit message or the reply to the
  user. Applies to `.py`, `.toml`, `.yml` and the shell scripts alike. Most existing code predates
  this rule and does the opposite; follow the rule in new code, and do not mass-rewrite old
  comments unless asked.
  Comments in functions need a brief description and clear Args.
- **Bump `version` in `pyproject.toml` for every fix.** Patch bump for a fix, minor for a feature.
  Nothing publishes the package today, so the bump is for history, not release.
- **Typing** Type everything

