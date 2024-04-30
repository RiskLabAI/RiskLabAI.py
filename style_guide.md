# Style Guide for Julia and Python

Most of the styles we follow are in line with QuantLib’s conventions. More styles and variable naming conventions in Julialang. More styles for Python.

## Indentation

Use four spaces per indentation level.

## Notebooks

Each notebook will have one section (`#`) and several subsections (`##`), subsubsections (`###`), etc. Each sub/sub/section should be followed by one descriptive paragraph with one to ten sentences. You don't need to assign a sectioning for each cell. In fact, each sub/sub/section could consist of a series of cells following the same purpose.

## Naming Conventions

- File names, folder names: UpperCamelCase [julia] or Python packages should also have short, all-lowercase names although the use of underscores is discouraged. [python]
- Names of variables, functions, and macros: lowerCamelCase [julia] or snake_case [python] Exceptions: `Δprice` lowercase with words separated by underscores as necessary to improve readability [python]
- The arguments of function (including constructors) are aligned with the indent of the first argument.
- Always use `self` for the first argument to instance methods. Always use `cls` for the first argument to class methods. If a function argument’s name clashes with a reserved keyword, it is generally better to append a single trailing underscore rather than use an abbreviation or spelling corruption. Thus `class_` is better than `clss`.
- Use of underscores is discouraged unless the name would be hard to read otherwise.
- Names of Types, Modules, Classes, Structs, Dictionaries: UpperCamelCase
- Functions that write to their arguments have names that end in `!`. These are sometimes called "mutating" or "in-place" functions because they are intended to produce changes in their arguments after the function is called, not just return a value.
- A constant or a static variable: ALL_CAPS_AND_UNDERSCORES
- Refrain from using `get` in the name of a function (e.g. use `PCA` instead of `getPCA` or `get_PCA`).
- Conciseness is valued, but avoid abbreviation (`indexin` rather than `indxin`) as it becomes difficult to remember whether and how particular words are abbreviated.
- List of abbreviations: `math`, `cov`, `corr`, `vol`, `max`, `min`.
- Number of something is named `nSomething` [julia] or `n_something` [python] (not `numSomething`, `numberSomething`, etc.)

## Commenting

- Each line is commented.
- Use the settings of your VS Code to hide comments unless your cursor is on a line.
- Files -> Preferences -> Settings -> search for “editor.token” -> Edit in settings.json -> add the following lines:
  ```json
  "editor.renderWhitespace": "none",
  "workbench.colorCustomizations": {
      "editor.lineHighlightBackground": "#2C3E50"
  },
  "editor.tokenColorCustomizations": {
      "comments": "#201c1c"
  }
  ```

## Copying variables

- The `=` operator does not make a copy. It is assigning a new variable (i.e., way to refer to an object) the same reference (actual object in memory). If you want a copy, you can use `copy`.

## Dataframes

- No use of pandas. Use `DataFrames` and `TimeArray`.

## Figures

- Should be transparent.

## Time Series

- All time-series data (pd.Series, pd.DataFrame, etc.) in Python should be converted to `TimeArray` structs in Julia. We strongly advise against using the Pandas Julia wrapper or the Julia DataFrame. `TimeArray` structs can be converted to DataFrames in the code blocks and then converted back to `TimeArrays` for convenience.

## Spacing

- `=`, `>`, `=>`, `<`, `=<`, `+=`, `-=`, `*=`, `==`, `===`, `&&`, `||`: leave a space before and after
- `*`, `/`, `^`: do not leave spaces before and after
- `+`, `-`: leave spaces before and after
- `;`: put a space after (and not before) comma and semicolon