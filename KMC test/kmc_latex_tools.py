#!/usr/bin/env python3
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text).replace("\n", " ")


def latex_path(value: object) -> str:
    text = str(value).replace("|", "/")
    return rf"\path|{text}|"


def tex_itemize(items: list[object]) -> str:
    lines = ["\\begin{itemize}"]
    lines.extend(f"\\item {latex_escape(item)}" for item in items)
    lines.append("\\end{itemize}")
    return "\n".join(lines)


def tex_enumerate(items: list[object]) -> str:
    lines = ["\\begin{enumerate}"]
    lines.extend(f"\\item {latex_escape(item)}" for item in items)
    lines.append("\\end{enumerate}")
    return "\n".join(lines)


def tex_metric_strip(metrics: list[tuple[object, object]]) -> str:
    cells = []
    for label, value in metrics:
        cells.append(
            "\\centering\\arraybackslash "
            rf"{{\Large\bfseries\color{{kmcblue}} {latex_escape(value)}}}\par"
            rf"{{\small\color{{kmcmuted}} {latex_escape(label)}}}"
        )
    return "\n".join(
        [
            "\\begin{center}",
            "\\renewcommand{\\arraystretch}{1.45}",
            "\\begin{tabularx}{\\linewidth}{|" + "|".join(["Y"] * len(metrics)) + "|}",
            "\\hline",
            " & ".join(cells) + r" \\",
            "\\hline",
            "\\end{tabularx}",
            "\\end{center}",
        ]
    )


def tex_table(headers: list[object], rows: list[list[object]], colspec: str) -> str:
    out = [
        "\\begin{center}",
        "\\small",
        "\\renewcommand{\\arraystretch}{1.30}",
        "\\arrayrulecolor{kmcline}",
        rf"\begin{{tabularx}}{{\linewidth}}{{{colspec}}}",
        "\\hline",
        "\\rowcolor{kmclight}",
        " & ".join(rf"\textbf{{{latex_escape(header)}}}" for header in headers) + r" \\",
        "\\hline",
    ]
    for row in rows:
        out.append(" & ".join(latex_escape(cell) for cell in row) + r" \\")
        out.append("\\hline")
    out.extend(["\\end{tabularx}", "\\arrayrulecolor{black}", "\\end{center}"])
    return "\n".join(out)


def tex_figure(path: str, caption: str, width: str = "0.92\\linewidth") -> str:
    return "\n".join(
        [
            "\\begin{figure}[H]",
            "\\centering",
            rf"\includegraphics[width={width}]{{{path}}}",
            rf"\caption{{{latex_escape(caption)}}}",
            "\\end{figure}",
        ]
    )


def tex_figure_note(text: str) -> str:
    return "\n".join(
        [
            "\\begin{center}",
            "\\begin{minipage}{0.90\\linewidth}",
            rf"{{\small\color{{kmcmuted}}\textbf{{图示说明：}}{latex_escape(text)}}}",
            "\\end{minipage}",
            "\\end{center}",
        ]
    )


def tex_document(title: str, subtitle: str, body: str, header: str) -> str:
    return rf"""\documentclass[UTF8,11pt,fontset=none]{{ctexart}}
\usepackage{{fontspec}}
\setmainfont{{Arial Unicode MS}}
\setsansfont{{Arial Unicode MS}}
\setCJKmainfont{{Arial Unicode MS}}
\setCJKsansfont{{Arial Unicode MS}}
\usepackage[a4paper,margin=1.65cm]{{geometry}}
\usepackage{{array}}
\usepackage{{booktabs}}
\usepackage{{caption}}
\usepackage{{enumitem}}
\usepackage{{fancyhdr}}
\usepackage{{float}}
\usepackage{{graphicx}}
\usepackage{{hyperref}}
\usepackage{{longtable}}
\usepackage{{tabularx}}
\usepackage{{titlesec}}
\usepackage{{url}}
\usepackage[table]{{xcolor}}

\definecolor{{kmcblue}}{{HTML}}{{2F5D97}}
\definecolor{{kmcdark}}{{HTML}}{{1F2D3D}}
\definecolor{{kmcmuted}}{{HTML}}{{5B6778}}
\definecolor{{kmclight}}{{HTML}}{{EEF3F8}}
\definecolor{{kmcline}}{{HTML}}{{BFCBDC}}

\hypersetup{{colorlinks=true,linkcolor=kmcblue,urlcolor=kmcblue}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{0.45em}}
\renewcommand{{\arraystretch}}{{1.18}}
\newcolumntype{{Y}}{{>{{\raggedright\arraybackslash}}X}}
\newcolumntype{{L}}[1]{{>{{\raggedright\arraybackslash}}p{{#1}}}}
\titleformat{{\section}}{{\Large\bfseries\color{{kmcblue}}}}{{}}{{0pt}}{{}}
\titleformat{{\subsection}}{{\large\bfseries\color{{kmcdark}}}}{{}}{{0pt}}{{}}
\captionsetup{{font=small,labelfont=bf}}
\setlist[itemize]{{leftmargin=1.65em,itemsep=0.12em,topsep=0.15em}}
\pagestyle{{fancy}}
\fancyhf{{}}
\lhead{{\color{{kmcmuted}}\small {latex_escape(header)}}}
\rhead{{\color{{kmcmuted}}\small \thepage}}
\renewcommand{{\headrulewidth}}{{0.25pt}}
\renewcommand{{\footrulewidth}}{{0pt}}

\begin{{document}}

{{\Huge\bfseries\color{{kmcblue}} {latex_escape(title)}}}\par
\vspace{{0.35em}}
{{\large\color{{kmcmuted}} {latex_escape(subtitle)}}}\par
\vspace{{1.0em}}

{body}

\end{{document}}
"""


def compile_pdf(tex_path: Path) -> Path:
    tectonic = shutil.which("tectonic")
    if tectonic is None:
        raise RuntimeError("tectonic compiler is required to build PDF reports")
    tex_path = tex_path.resolve()
    subprocess.run(
        [tectonic, tex_path.name, "--outdir", "."],
        cwd=tex_path.parent,
        check=True,
    )
    pdf_path = tex_path.with_suffix(".pdf")
    if not pdf_path.exists() or pdf_path.stat().st_size == 0:
        raise RuntimeError(f"PDF was not produced: {pdf_path}")
    return pdf_path
