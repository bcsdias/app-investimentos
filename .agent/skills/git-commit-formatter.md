
---
name: git-commit-formatter
description: Formats git commit messages using Conventional Commits and Gitmoji. Use this when the user asks to commit changes or write a commit message.
---

# Git Commit Formatter Skill

When writing a git commit message, you MUST strictly follow the Conventional Commits specification integrated with Gitmojis.

## Format
`<type>[optional scope]: <emoji> <description>`

## Allowed Types & Gitmojis
* **feat**: ✨ (A new feature)
* **fix**: 🐛 (A bug fix)
* **docs**: 📝 (Documentation only changes)
* **style**: 💄 (Changes that do not affect the meaning of the code - white-space, formatting, etc)
* **refactor**: ♻️ (A code change that neither fixes a bug nor adds a feature)
* **perf**: ⚡️ (A code change that improves performance)
* **test**: ✅ (Adding missing tests or correcting existing tests)
* **build**: 📦 (Changes that affect the build system or external dependencies - e.g., npm, pip)
* **ci**: 👷 (Changes to CI configuration files and scripts - e.g., GitHub Actions, Jenkins)
* **chore**: 🔧 (Changes to the build process or auxiliary tools and libraries)
* **revert**: ⏪️ (Reverts a previous commit)
* **security**: 🔒 (Fixes security vulnerabilities)

## Instructions
1. **Deep Analysis:** Examine the `git diff` to understand *exactly* what was added or changed. Identify the core purpose of the changes.
2. **Specific Scope:** Identify the most relevant scope (e.g., `agent`, `db`, `auth`, `ui`).
3. **Detailed Description:** Write the final description in **Portuguese (pt-BR)** using the imperative mood.
   - **CRITICAL:** The emoji must come immediately after the colon (`:`), followed by a single space.
   - **AVOID:** Vague terms like "ajustes", "update", or "atualiza arquivos".
   - **BETTER:** Use specific phrases like "adiciona regras de automação de commit", "configura workflow de deploy", or "corrige validação de ACL no CMTS".
4. **Imperative Mood:** Always start the Portuguese description with a verb in the imperative (e.g., "configura", "adiciona", "remove", "ajusta").
5. **Breaking Changes:** If the change breaks compatibility, add `BREAKING CHANGE:` in the footer with a detailed explanation in Portuguese.
6. **Semantic Versioning (SemVer):** The commit MUST suggest the next project version based on the changes:
   - **PATCH (x.y.Z):** Increment Z for bug fixes (`fix`). E.g., 1.6.8 -> 1.6.9.
   - **MINOR (x.Y.z):** Increment Y for new features (`feat`) and reset Z. E.g., 1.6.8 -> 1.7.0.
   - **MAJOR (X.y.z):** Increment X for incompatible changes (`BREAKING CHANGE`) and reset Y and Z. E.g., 1.6.8 -> 2.0.0.
7. **Changelog Sync:** Ensure that the `README.md` file's changelog section (`## 📝 Changelog (Recente)`) is updated to reflect the changes being committed and the suggested version.

## Examples
- `feat(auth): ✨ adiciona suporte a login com Google e sugere versão v1.2.0`
- `fix(api): 🐛 corrige timeout na rota de relatórios e sugere versão v1.2.1`
- `docs(readme): 📝 atualiza instruções de instalação`
- `security(headers): 🔒 implementa políticas de CORS restritivas`