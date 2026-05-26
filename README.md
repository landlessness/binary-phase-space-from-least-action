# Code and figures for "Binary Phase Space from Least Action"

> **Working title — subject to change.** The manuscript title and content are
> still being settled; treat everything here as provisional.

Repository for the manuscript *Binary Phase Space from Least Action*
(B. S. Mulloy, 2026).

This repository will contain everything needed to reproduce the paper: the
code that generates the data figures, the OmniGraffle source for the schematic
figures, and the LaTeX source of the manuscript.

## Structure

- `python/` — Python project (managed with [uv](https://docs.astral.sh/uv/))
- `omnigraffle/` — OmniGraffle source for the schematic figures
- `tex/` — LaTeX manuscript source (`main.tex`, `main.bib`, `supplement.tex`),
  with included figures under `tex/figures/`

## Requirements

- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/) for environment management
- LaTeX with the `revtex4-2` class (for the manuscript)
- OmniGraffle (optional, only to edit the schematic figures)

## Building the manuscript

```bash
cd tex
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

## Citation

This release is archived on Zenodo.

> **TODO:** add the Zenodo concept DOI here after the first release.

Suggested citation (update once a DOI exists):

> Mulloy, B. S. (2026). *Code and figures for "Binary Phase Space from Least
> Action"*. Zenodo. https://doi.org/10.5281/zenodo.XXXXXXXX

A BibTeX entry is available from the Zenodo deposit page, or from the
"Cite this repository" button on GitHub (populated from `CITATION.cff`).

For citing the paper itself, see the published manuscript.

## License

- **Code** (`python/`): MIT License — see `LICENSE` in the repo root.
- **Manuscript and figures** (`tex/`, `omnigraffle/`): Creative Commons
  Attribution 4.0 International (CC BY 4.0) — see `tex/LICENSE` and
  `omnigraffle/LICENSE`.

## Contact

Brian S. Mulloy — bmulloy@umich.edu