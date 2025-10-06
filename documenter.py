import ast
import pathlib
import sys
from typing import TextIO

# --- Part 1: File Tree Generation (Adapted from tree.py) ---

def generate_file_tree(
    root_dir: pathlib.Path,
    file_handle: TextIO,
    ignore_list: list[str],
    prefix: str = "",
):
    """Generates and writes a tree-like structure for a directory to a file."""
    paths = sorted(
        [p for p in root_dir.iterdir() if p.name not in ignore_list],
        key=lambda p: (not p.is_dir(), p.name.lower()),
    )

    for i, path in enumerate(paths):
        is_last = i == len(paths) - 1
        connector = "└── " if is_last else "├── "
        if path.is_dir():
            print(f"{prefix}{connector}📁 {path.name}/", file=file_handle)
            new_prefix = prefix + ("    " if is_last else "│   ")
            generate_file_tree(path, file_handle, ignore_list, prefix=new_prefix)
        else:
            print(f"{prefix}{connector}📄 {path.name}", file=file_handle)

# --- Part 2: Docstring Extraction ---

def extract_docstrings(module_path: pathlib.Path, file_handle: TextIO):
    """Parses a Python file and writes its structure and docstrings to a file."""
    try:
        with open(module_path, "r", encoding="utf-8") as f:
            source = f.read()
        
        tree = ast.parse(source)
        
        # Write the file path as a header
        print(f"\n### 📄 `{module_path}`\n", file=file_handle)

        # Extract module-level docstring
        module_docstring = ast.get_docstring(tree)
        if module_docstring:
            print(f"> {module_docstring}\n", file=file_handle)

        # Extract functions and classes
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                print_function_doc(node, file_handle)
            elif isinstance(node, ast.ClassDef):
                print_class_doc(node, file_handle)

    except Exception as e:
        print(f"Could not parse {module_path}: {e}", file=file_handle)

def print_function_doc(node: ast.FunctionDef, file_handle: TextIO):
    """Formats and prints a function's signature and docstring."""
    signature = f"def {node.name}{ast.unparse(node.args)}:"
    print(f"#### `function {node.name}`\n", file=file_handle)
    print(f"```python\n{signature}\n```", file=file_handle)
    docstring = ast.get_docstring(node)
    if docstring:
        print(f"\n> {docstring}\n", file=file_handle)

def print_class_doc(node: ast.ClassDef, file_handle: TextIO):
    """Formats and prints a class and its methods' signatures and docstrings."""
    print(f"#### `class {node.name}`\n", file=file_handle)
    docstring = ast.get_docstring(node)
    if docstring:
        print(f"> {docstring}\n", file=file_handle)
    
    for method in node.body:
        if isinstance(method, ast.FunctionDef):
            signature = f"def {method.name}{ast.unparse(method.args)}:"
            print(f"##### `method {method.name}`\n", file=file_handle)
            print(f"```python\n{signature}\n```", file=file_handle)
            method_docstring = ast.get_docstring(method)
            if method_docstring:
                print(f"\n> {method_docstring}\n", file=file_handle)

# --- Part 3: Main Execution ---

if __name__ == "__main__":
    # The source code directory to scan (e.g., 'RiskLabAI')
    if len(sys.argv) < 2:
        print("Usage: python documenter.py <path_to_library>")
        sys.exit(1)
        
    start_path_str = sys.argv[1]
    start_path = pathlib.Path(start_path_str)
    
    output_filename = "DOCUMENTATION.md"
    ignore = ["__pycache__", ".git", ".github", ".vscode", "tests", "test"]

    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            # --- Main Title ---
            f.write(f"# Documentation for `{start_path.name}` Library\n\n")

            # --- File Structure Section ---
            f.write("## 🌳 File Structure\n\n")
            f.write("```\n")
            print(f"📁 {start_path.name}/", file=f)
            generate_file_tree(start_path, file_handle=f, ignore_list=ignore)
            f.write("```\n")

            # --- Code Reference Section ---
            f.write("\n## 📄 Module & Function Reference\n")
            py_files = sorted(start_path.rglob("*.py"))
            for py_file in py_files:
                # Exclude __init__.py files from detailed documentation for brevity
                if py_file.name != "__init__.py":
                    extract_docstrings(py_file, file_handle=f)
        
        print(f"✅ Documentation successfully generated in '{output_filename}'")

    except FileNotFoundError:
        print(f"Error: Directory not found at '{start_path_str}'")
    except Exception as e:
        print(f"An error occurred: {e}")