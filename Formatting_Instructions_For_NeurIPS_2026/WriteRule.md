# NeurIPS 2026 Writing Rules

## 说明

- 本文件基于官方模板 `neurips_2026.tex` 中从 Abstract 开始的写作与排版规则整理而成。
- 原始 LaTeX 命令在 Markdown 预览中容易出现反斜杠、数学符号和转义显示混乱，因此这里改写成可读的 Markdown 结构。
- 文档分为英文整理版和中文对照版两部分。
- 示例命令、图表和参考文献格式保留为代码块，便于直接查看与复制。

---

## English Version

### Abstract

- The abstract paragraph should be indented `1/2 inch` (`3 picas`) on both the left and right margins.
- Use `10-point` type with `11-point` vertical spacing.
- The word `Abstract` must be centered, bold, and set in `12-point` type.
- Two line spaces should precede the abstract.
- The abstract must be limited to one paragraph.

### Submission of Papers to NeurIPS 2026

Please read the instructions carefully and follow them faithfully.

#### Style

- Papers submitted to NeurIPS 2026 must follow these instructions.
- Papers may be up to `nine pages` long, including figures.
- Papers exceeding the page limit will not be reviewed or otherwise considered for presentation.
- Additional pages containing acknowledgments, references, checklist, and optional technical appendices do not count as content pages.
- The 2026 margins are the same as in previous years.
- Authors must use the current NeurIPS LaTeX style files from the official website.
- Using outdated files or modifying the style files may result in desk rejection.

#### Retrieval of Style Files

- NeurIPS style files and conference information are available at:

  <https://neurips.cc>

- The only supported style file for NeurIPS 2026 is `neurips_2026.sty`, rewritten for LaTeX2e.
- Previous style files for LaTeX 2.09, Microsoft Word, and RTF are no longer supported.

The style file includes three optional arguments:

- `final`: creates a camera-ready copy.
- `preprint`: creates a preprint version, for example for arXiv.
- `nonatbib`: prevents automatic loading of `natbib` if it conflicts with other packages.

#### Preprint Option

- If you want to post a preprint online, use the `preprint` option.
- This creates a non-anonymized version with the footer text `Preprint. Work in progress.`
- This version may be distributed freely, as long as it does not state which conference it was submitted to.
- Do not use the `final` option unless the paper has been accepted.

At submission time:

- Omit both `final` and `preprint`.
- The paper will be anonymized automatically.
- Line numbers will be added for review.
- Do not refer to those line numbers in the paper, because they will be removed in the camera-ready version.

The file `neurips_2026.tex` can be used as a shell document. Replace the title, author, abstract, and body text with your own content.

The formatting rules are summarized in Sections `General formatting instructions`, `Headings`, and `Citations, figures, tables, references`.

### General Formatting Instructions

- The text must fit inside a rectangle `5.5 inches` (`33 picas`) wide and `9 inches` (`54 picas`) long.
- The left margin is `1.5 inches` (`9 picas`).
- Use `10-point` type with `11-point` vertical spacing.
- Times New Roman is the preferred typeface and is selected by default.
- Paragraphs are separated by `1/2` line space (`5.5 points`) with no indentation.

Title requirements:

- The paper title should be `17-point`, bold, centered, and in initial caps/lower case.
- It should appear between two horizontal rules.
- The top rule should be `4 points` thick.
- The bottom rule should be `1 point` thick.
- Leave `1/4 inch` space above and below the rules.
- All pages should start `1 inch` (`6 picas`) from the top of the page.

Author block in the final version:

- Authors' names are set in boldface.
- Each name is centered above the corresponding address.
- The lead author's name should be listed first, at the left.
- Co-authors follow after that, especially when they have different affiliations.
- If there is only one co-author, list the author and co-author side by side.

Pay special attention to the rules about figures, tables, acknowledgments, and references.

### Headings

#### First-Level Headings

- All headings should be lower case except for the first word and proper nouns.
- Headings should be flush left and bold.
- First-level headings should be `12-point`.

#### Second-Level Headings

- Second-level headings should be `10-point`.

#### Third-Level Headings

- Third-level headings should be `10-point`.

#### Paragraph Headings

- The `\paragraph` command is available.
- It produces a bold, flush-left heading that is inline with the text.
- The heading is followed by `1 em` of space.

### Citations, Figures, Tables, and References

These instructions apply to everyone.

#### Citations Within the Text

- The `natbib` package is loaded by default.
- Citations may use either author-year style or numeric style.
- The chosen citation style must be consistent throughout the paper.
- Any reference formatting style is acceptable as long as it is used consistently.

Documentation for `natbib`:

- <http://mirrors.ctan.org/macros/latex/contrib/natbib/natnotes.pdf>

Useful command example:

```tex
\citet{hasselmo} investigated\dots
```

Output:

```text
Hasselmo, et al. (1995) investigated...
```

If you need to pass options to `natbib`, add this before loading `neurips_2026`:

```tex
\PassOptionsToPackage{options}{natbib}
```

If `natbib` conflicts with another package, use:

```tex
\usepackage[nonatbib]{neurips_2026}
```

Double-blind rule:

- Refer to your own published work in the third person.
- Use wording like `In the previous work of Jones et al. [4]`, not `In our previous work [4]`.
- If citing your own papers that are not widely available, use anonymous author names such as `A. Anonymous` and include the anonymized paper in the supplementary material.

#### Footnotes

- Use footnotes sparingly.
- If needed, indicate them with a number in the text.
- Footnotes should appear at the bottom of the same page.
- A horizontal rule of `2 inches` (`12 picas`) should precede the footnotes.
- Footnotes should appear after punctuation marks.

Sample footnote usage:

```tex
number\footnote{Sample of the first footnote.}
marks.\footnote{As in this example.}
```

#### Figures

- Artwork must be neat, clean, and legible.
- Lines should be dark enough for reproduction.
- Figure numbers and captions always appear after the figure.
- Leave one line space before and after the caption.
- Figure captions should be lower case except for the first word and proper nouns.
- Figures should be numbered consecutively.
- Color figures are allowed, but captions and text should remain legible in both color and black-and-white printing.

Sample figure:

```tex
\begin{figure}
  \centering
  \fbox{\rule[-.5cm]{0cm}{4cm} \rule[-.5cm]{4cm}{0cm}}
  \caption{Sample figure caption. Explain what the figure shows and add a key take-away message to the caption.}
\end{figure}
```

#### Tables

- Tables must be centered, neat, clean, and legible.
- The table number and title always appear before the table.
- Leave one line space before the title, one line space after the title, and one line space after the table.
- Table titles should be lower case except for the first word and proper nouns.
- Tables should be numbered consecutively.
- Publication-quality tables should not contain vertical rules.
- The `booktabs` package is strongly recommended.

Booktabs package:

- <https://www.ctan.org/pkg/booktabs>

Sample table:

```tex
\begin{table}
  \caption{Sample table caption. Explain what the table shows and add a key take-away message to the caption.}
  \label{sample-table}
  \centering
  \begin{tabular}{lll}
    \toprule
    \multicolumn{2}{c}{Part}                   \\
    \cmidrule(r){1-2}
    Name     & Description     & Size ($\mu$m) \\
    \midrule
    Dendrite & Input terminal  & $\approx$100     \\
    Axon     & Output terminal & $\approx$10      \\
    Soma     & Cell body       & up to $10^6$  \\
    \bottomrule
  \end{tabular}
\end{table}
```

#### Math

- Do not use bare TeX display-math commands if you want correct line numbering in submission mode.
- Use LaTeX or AMSTeX display-math commands instead.
- Avoid using `$$ ... $$`.

References:

- <https://tex.stackexchange.com/questions/503/why-is-preferable-to>
- <https://tex.stackexchange.com/questions/40492/what-are-the-differences-between-align-equation-and-displaymath>

#### Final Instructions

- Do not change the formatting parameters in the style files.
- Do not modify the width or length of the text area.
- Do not change font sizes.
- Pages should be numbered.

### Preparing PDF Files

- Submission files must use `US Letter` paper size, not `A4`.
- The PDF must contain only Type 1 or embedded TrueType fonts.

Important notes:

- Generate PDF directly with `pdflatex`.
- You can inspect PDF fonts in Acrobat Reader via `File > Document Properties > Fonts > Show All Fonts`.
- You can also use `pdffonts`, which is available on most Linux machines.
- `xfig` patterned shapes use bitmap fonts, so use `solid` shapes instead.
- The `\bbold` package almost always uses bitmap fonts. Prefer AMS fonts.

Recommended replacement:

```tex
\usepackage{amsfonts}
```

Examples:

```tex
\mathbb{R}
\mathbb{N}
\mathbb{C}
```

Possible workaround:

```tex
\newcommand{\RR}{I\!\!R} % real numbers
\newcommand{\Nat}{I\!\!N} % natural numbers
\newcommand{\CC}{I\!\!\!\!C} % complex numbers
```

If your file contains Type 3 fonts or non-embedded TrueType fonts, you will be asked to fix it.

#### Margins in LaTeX

- Margin problems often come from manually positioned figures using `\special` or similar commands.
- Use `\includegraphics` from the `graphicx` package instead.
- Always specify figure width as a multiple of `\linewidth`.

Example:

```tex
\usepackage[pdftex]{graphicx}
\includegraphics[width=0.8\linewidth]{myfile.pdf}
```

Reference:

- <http://mirrors.ctan.org/macros/latex/required/graphics/grfguide.pdf>

If LaTeX cannot hyphenate a line correctly, provide hyphenation hints with `\-`.

### Acknowledgments

- Use an unnumbered first-level heading for acknowledgments.
- Place acknowledgments at the end of the paper before the references.
- You must declare funding supporting the submitted work.
- You must also declare competing interests related to financial activities outside the submitted work.

Funding disclosure information:

- <https://neurips.cc/Conferences/2026/PaperInformation/FundingDisclosure>

Important rule:

- Do not include the acknowledgments section in the anonymized submission.
- Include it only in the final version.
- You may use the `ack` environment to hide the section automatically in the anonymized submission.

### References

- References come after acknowledgments in the camera-ready paper.
- Use an unnumbered first-level heading for references.
- Any citation style is acceptable if used consistently.
- It is permissible to reduce the font size to `small` (`9-point`) for the reference list.
- The references section does not count toward the page limit.

Official sample references:

```text
[1] Alexander, J.A. & Mozer, M.C. (1995) Template-based algorithms for
connectionist rule extraction. In G. Tesauro, D.S. Touretzky and T.K. Leen
(eds.), Advances in Neural Information Processing Systems 7,
pp. 609--616. Cambridge, MA: MIT Press.

[2] Bower, J.M. & Beeman, D. (1995) The Book of GENESIS: Exploring
Realistic Neural Models with the GEneral NEural SImulation System.
New York: TELOS/Springer--Verlag.

[3] Hasselmo, M.E., Schnell, E. & Barkai, E. (1995) Dynamics of learning and
recall at excitatory recurrent synapses and cholinergic modulation in rat
hippocampal region CA3. Journal of Neuroscience 15(7):5249-5262.
```

### Technical Appendices and Supplementary Material

- Technical appendices with additional results, figures, graphs, and proofs may be submitted before the full submission deadline.
- You may upload a ZIP file for videos or code.
- Do not upload a separate PDF file for the appendix.
- There is no page limit for the technical appendices.
- The appendix should be optional reading for reviewers.
- The paper must stand alone without the appendix.
- Do not move critical experiments that support the main claims into the appendix.

Checklist inclusion in the template:

```tex
\newpage
\input{checklist.tex}
```

---

## 中文版本

### 摘要

- 摘要段落左右两侧都应各缩进 `1/2 英寸`（`3 picas`）。
- 正文使用 `10 磅` 字体，行距为 `11 磅`。
- `Abstract` 一词必须居中、加粗，并使用 `12 磅` 字号。
- 摘要前应空两行。
- 摘要必须限制为一个自然段。

### 向 NeurIPS 2026 提交论文

请仔细阅读并严格遵守以下说明。

#### 样式要求

- 投稿到 NeurIPS 2026 的论文必须遵守这些排版说明。
- 正文最多 `9 页`，图也计入正文页数。
- 超过页数限制的论文将不会被审稿，也不会被考虑接收。
- 致谢、参考文献、checklist 和可选技术附录所占页面不计入正文页数。
- 2026 年的页边距与往年保持一致。
- 作者必须使用 NeurIPS 官网提供的最新 LaTeX 样式文件。
- 使用旧版本样式文件，或者自行修改样式文件，可能会导致直接拒稿。

#### 样式文件获取

- NeurIPS 样式文件和会议信息见官网：

  <https://neurips.cc>

- NeurIPS 2026 唯一支持的样式文件是 `neurips_2026.sty`，它基于 LaTeX2e 重写。
- 旧版 LaTeX 2.09、Microsoft Word 和 RTF 样式均不再支持。

样式文件有三个可选参数：

- `final`：生成 camera-ready 终稿版本。
- `preprint`：生成预印本版本，例如用于 arXiv。
- `nonatbib`：在与其他宏包冲突时，禁止自动加载 `natbib`。

#### 预印本选项

- 如果要在线发布预印本，应使用 `preprint` 选项。
- 该选项会生成一个非匿名版本，并在页脚显示 `Preprint. Work in progress.`。
- 这个版本可以自行分发，但不能写明它投稿到了哪个会议。
- 未被接收之前，不要使用 `final` 选项。

正式投稿时：

- 不要使用 `final` 和 `preprint`。
- 模板会自动匿名化。
- 模板会自动添加行号，方便审稿。
- 不要在论文正文中引用这些行号，因为终稿生成时它们会被删除。

`neurips_2026.tex` 可以作为写论文的骨架文件，你只需要把标题、作者、摘要和正文替换成自己的内容。

规则重点集中在以下三部分：

- `General formatting instructions`
- `Headings`
- `Citations, figures, tables, references`

### 一般排版要求

- 正文必须落在 `5.5 英寸` 宽、`9 英寸` 高的矩形版心内。
- 左边距是 `1.5 英寸`。
- 使用 `10 磅` 字体和 `11 磅` 行距。
- 默认字体为 Times New Roman。
- 段落之间以 `1/2` 行距（`5.5 points`）分隔，不使用首行缩进。

标题要求：

- 论文标题使用 `17 磅`、加粗、居中，采用首字母大写/其余小写风格。
- 标题位于两条水平线之间。
- 上方水平线粗 `4 points`。
- 下方水平线粗 `1 point`。
- 水平线与标题之间上下各留 `1/4 英寸` 空白。
- 每页顶部距页面上边缘应为 `1 英寸`。

终稿作者栏要求：

- 作者姓名加粗。
- 每位作者姓名应居中放在对应单位上方。
- lead author 应列在最前、最左侧。
- 其余作者按顺序排列在后面，尤其在单位不同的情况下更要如此。
- 如果只有一个合作者，则作者与合作者并排放置。

还需要特别注意图、表、致谢和参考文献的写法。

### 标题层级

#### 一级标题

- 除第一个词和专有名词外，标题应使用小写。
- 标题左对齐并加粗。
- 一级标题字号为 `12 磅`。

#### 二级标题

- 二级标题字号为 `10 磅`。

#### 三级标题

- 三级标题字号为 `10 磅`。

#### 段落标题

- 可以使用 `\paragraph` 命令。
- 它会生成左对齐、加粗、与正文同一行的标题。
- 标题后跟 `1 em` 间距。

### 引用、图、表与参考文献

以下规则适用于所有投稿者。

#### 文中引用

- 模板默认加载 `natbib`。
- 文中引用可以使用作者-年份制，也可以使用数字制。
- 整篇文章必须保持一致。
- 参考文献条目本身的格式可以自由选择，但必须前后一致。

`natbib` 文档：

- <http://mirrors.ctan.org/macros/latex/contrib/natbib/natnotes.pdf>

示例命令：

```tex
\citet{hasselmo} investigated\dots
```

效果：

```text
Hasselmo, et al. (1995) investigated...
```

如果要给 `natbib` 传参，应在加载 `neurips_2026` 之前加入：

```tex
\PassOptionsToPackage{options}{natbib}
```

如果 `natbib` 与别的宏包冲突，可以写：

```tex
\usepackage[nonatbib]{neurips_2026}
```

双盲投稿要求：

- 引用自己已经发表的工作时，必须使用第三人称。
- 应写成 `In the previous work of Jones et al. [4]`，而不是 `In our previous work [4]`。
- 如果引用自己尚未公开发表的论文，应使用匿名作者名，例如 `A. Anonymous`，并将匿名稿放入补充材料。

#### 脚注

- 脚注应尽量少用。
- 如确需使用，应在正文中用编号标记。
- 脚注应放在同一页页脚。
- 脚注上方应有一条 `2 英寸` 宽的水平线。
- 脚注应放在标点符号之后。

示例：

```tex
number\footnote{Sample of the first footnote.}
marks.\footnote{As in this example.}
```

#### 图

- 所有图必须清晰、整洁、可辨认。
- 线条应足够深，便于印刷复制。
- 图号和图注总是放在图的后面。
- 图注前后各留一行空白。
- 图注应使用小写，只有首词和专有名词首字母大写。
- 图应连续编号。
- 允许彩色图，但图注和正文在彩色与黑白打印下都应保持清晰可读。

示例图：

```tex
\begin{figure}
  \centering
  \fbox{\rule[-.5cm]{0cm}{4cm} \rule[-.5cm]{4cm}{0cm}}
  \caption{Sample figure caption. Explain what the figure shows and add a key take-away message to the caption.}
\end{figure}
```

#### 表

- 所有表格都必须居中、整洁、清晰、可读。
- 表号和表题总是放在表格前面。
- 表题前一行空白，表题后也空一行，表格结束后再空一行。
- 表题使用小写，只有首词和专有名词首字母大写。
- 表格应连续编号。
- 高质量学术表格不应包含竖线。
- 强烈建议使用 `booktabs` 宏包。

`booktabs` 链接：

- <https://www.ctan.org/pkg/booktabs>

示例表格：

```tex
\begin{table}
  \caption{Sample table caption. Explain what the table shows and add a key take-away message to the caption.}
  \label{sample-table}
  \centering
  \begin{tabular}{lll}
    \toprule
    \multicolumn{2}{c}{Part}                   \\
    \cmidrule(r){1-2}
    Name     & Description     & Size ($\mu$m) \\
    \midrule
    Dendrite & Input terminal  & $\approx$100     \\
    Axon     & Output terminal & $\approx$10      \\
    Soma     & Cell body       & up to $10^6$  \\
    \bottomrule
  \end{tabular}
\end{table}
```

#### 数学公式

- 如果希望投稿版的行号正确，不要使用裸 TeX 的 display math 命令。
- 应使用 LaTeX 或 AMSTeX 的数学环境。
- 不建议使用 `$$ ... $$`。

参考说明：

- <https://tex.stackexchange.com/questions/503/why-is-preferable-to>
- <https://tex.stackexchange.com/questions/40492/what-are-the-differences-between-align-equation-and-displaymath>

#### 最终说明

- 不要修改样式文件中的任何排版参数。
- 不要修改版心宽度和高度。
- 不要修改字体大小。
- 页面应保留页码。

### PDF 文件准备

- 投稿 PDF 必须使用 `US Letter` 纸型，而不是 `A4`。
- PDF 中只能包含 Type 1 字体或嵌入式 TrueType 字体。

重要说明：

- 应直接用 `pdflatex` 生成 PDF。
- 可以在 Acrobat Reader 中通过 `File > Document Properties > Fonts > Show All Fonts` 检查字体。
- 也可以使用 `pdffonts` 命令。
- `xfig` 的 patterned 形状会调用位图字体，应改用 `solid` 形状。
- `\bbold` 宏包通常也会产生位图字体，应优先使用 AMS 字体。

推荐替代写法：

```tex
\usepackage{amsfonts}
```

示例：

```tex
\mathbb{R}
\mathbb{N}
\mathbb{C}
```

备选写法：

```tex
\newcommand{\RR}{I\!\!R} % real numbers
\newcommand{\Nat}{I\!\!N} % natural numbers
\newcommand{\CC}{I\!\!\!\!C} % complex numbers
```

如果 PDF 含有 Type 3 字体或未嵌入的 TrueType 字体，组委会会要求修正。

#### LaTeX 中的边距问题

- 边距问题通常来自手工使用 `\special` 或类似命令定位图片。
- 应使用 `graphicx` 包中的 `\includegraphics`。
- 图片宽度应写成 `\linewidth` 的倍数。

示例：

```tex
\usepackage[pdftex]{graphicx}
\includegraphics[width=0.8\linewidth]{myfile.pdf}
```

参考文档：

- <http://mirrors.ctan.org/macros/latex/required/graphics/grfguide.pdf>

如果 LaTeX 无法正确断词，可以用 `\-` 提供断词提示。

### 致谢

- 致谢应使用不编号的一级标题。
- 致谢放在正文末尾、参考文献之前。
- 必须声明支撑论文的资金来源。
- 还必须声明与投稿无关但可能构成利益冲突的相关财务活动。

资金披露说明：

- <https://neurips.cc/Conferences/2026/PaperInformation/FundingDisclosure>

重要规则：

- 匿名投稿版本中不得出现致谢。
- 只有终稿版才能包含致谢。
- 可以使用 `ack` 环境，让模板在匿名版自动隐藏致谢部分。

### 参考文献

- 在终稿中，参考文献位于致谢之后。
- 参考文献标题使用不编号的一级标题。
- 引用风格可以自由选择，但必须一致。
- 参考文献部分允许缩小到 `small`（`9 磅`）。
- 参考文献页数不计入正文限制。

官方示例参考文献：

```text
[1] Alexander, J.A. & Mozer, M.C. (1995) Template-based algorithms for
connectionist rule extraction. In G. Tesauro, D.S. Touretzky and T.K. Leen
(eds.), Advances in Neural Information Processing Systems 7,
pp. 609--616. Cambridge, MA: MIT Press.

[2] Bower, J.M. & Beeman, D. (1995) The Book of GENESIS: Exploring
Realistic Neural Models with the GEneral NEural SImulation System.
New York: TELOS/Springer--Verlag.

[3] Hasselmo, M.E., Schnell, E. & Barkai, E. (1995) Dynamics of learning and
recall at excitatory recurrent synapses and cholinergic modulation in rat
hippocampal region CA3. Journal of Neuroscience 15(7):5249-5262.
```

### 技术附录与补充材料

- 含额外结果、图、证明等内容的技术附录可以在正式投稿截止前一并提交。
- 视频或代码可以打包成 ZIP 上传。
- 不要单独上传附录 PDF。
- 技术附录没有页数限制。
- 附录应被视为审稿人的可选阅读材料。
- 论文本体必须能够独立成立，不能依赖附录。
- 支撑主张的关键实验不应被放到附录中。

模板中的 checklist 引入方式：

```tex
\newpage
\input{checklist.tex}
```
