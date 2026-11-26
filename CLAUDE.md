# CLAUDE.md

## Project context
This repository supports a research study on Solar Energetic Particle (SEP) forecasting.

## My paper (SWLG-Paper)
- Never edit my paper files (`SWLG-Paper/main.tex`, `SWLG-Paper/text/*.tex`, or any other text I write myself).
- When I ask you to reword, paraphrase, or "translate" something, give me the text in chat only. I will place it myself.
- Only write to a paper file if I explicitly tell you to put the text into that file.
- make text red that are a direct copy from another paper
## Citations
- **Never invent a page number, equation number, or citation.** If you cannot verify the location in the source, mark it `UNVERIFIED — <what you believe the source is>` so I can check it.
- If something comes from somewhere other than Devore (a paper, docs, etc.), cite that instead. Use author, year, and page, section, or URL.

## Notes vault (Obsidian-style)
`notes/` is an Obsidian vault. Treat it as my second brain.

### When to write a note
- Whenever I ask a conceptual or methodological question (statistics, physics, SEP forecasting, a modeling choice), create or update a note answering it.
- Do not write notes for trivial requests (fixing typos, renaming variables, running a command).

### How to write notes
- One concept per `.md` file. Use lowercase-hyphenated filenames (e.g. `notes/statistics/confidence-intervals.md`).
- Organize into subdirectories by topic (e.g. `notes/statistics/`, `notes/sep/`, `notes/ml/`). Create new subdirectories as needed.
- Begin each note with YAML frontmatter:
  ```yaml
  ---
  tags: [topic, subtopic]
  sources: [Devore 2012 p. XX]
  created: YYYY-MM-DD
  ---
  ```
- Include the question I asked, a concise answer, and any equations in LaTeX (`$...$` / `$$...$$`).
- The same citation rules apply: no invented page numbers; mark anything unverified.

### Linking, not duplicating
- Before writing a note, search `notes/` for an existing note on the concept.
- If a concept is already explained, link to it with `[[note-name]]` instead of re-explaining it.
- If the existing note is incomplete, update that note rather than creating a duplicate.

## Format
- All figures and text are written latex and be inline with the AGU style guide provided here 
-https://www.agu.org/publications/authors/journals/text-graphics-requirements#guidelines
- https://libguides.humboldt.edu/ocn/agu
- https://www.agu.org/publications/authors/journals/grammar-style-guide
-all figures and equations will be place in their own seperate folder in accordance to the section.