"""
Generates a markdown file representing the project's file structure.

Usage:
    python tree.py [root_directory] [output_filename]
"""

import pathlib
import sys
from typing import TextIO, List, Optional

def generate_file_structure(
    root_dir: pathlib.Path,
    file_handle: TextIO,
    ignore_list: List[str],
    prefix: str = "",
):
    """
    Generates a tree-like structure for a directory and writes it to a file.

    Args:
        root_dir (pathlib.Path): The path to the root directory.
        file_handle (TextIO): The file object to write the output to.
        ignore_list (List[str]): A list of names to ignore.
        prefix (str): The prefix for drawing the tree structure.
    """
    try:
        paths = sorted(
            [p for p in root_dir.iterdir() if p.name not in ignore_list],
            key=lambda p: (not p.is_dir(), p.name.lower()),
        )
    except FileNotFoundError:
        print(f"Warning: Directory {root_dir} not found.", file=sys.stderr)
        return

    for i, path in enumerate(paths):
        is_last = i == len(paths) - 1
        connector = "└── " if is_last else "├── "
        
        if path.is_dir():
            print(f"{prefix}{connector}📁 {path.name}/", file=file_handle)
            new_prefix = prefix + ("    " if is_last else "│   ")
            generate_file_structure(path, file_handle, ignore_list, new_prefix)
        else:
            print(f"{prefix}{connector}📄 {path.name}", file=file_handle)


if __name__ == "__main__":
    # --- Configuration ---
    DEFAULT_IGNORE = [
        "__pycache__", ".git", ".github", ".vscode", "build", 
        "dist", ".mypy_cache", ".pytest_cache", "venv", ".venv"
    ]
    
    # The directory to scan (defaults to current directory)
    start_path_str = sys.argv[1] if len(sys.argv) > 1 else "."
    
    # The output filename (defaults to STRUCTURE.md)
    output_filename = sys.argv[2] if len(sys.argv) > 2 else "STRUCTURE.md"
    
    # --- Execution ---
    start_path = pathlib.Path(start_path_str)
    root_name = start_path.resolve().name
    
    if not start_path.is_dir():
        print(f"Error: '{start_path_str}' is not a valid directory.")
        sys.exit(1)

    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            # Write the Markdown header and opening code block
            f.write("# Project File Structure\n\n```\n")
            
            # Write the root directory and start the tree
            print(f"📁 {root_name}/", file=f)
            generate_file_structure(
                start_path, 
                file_handle=f, 
                ignore_list=DEFAULT_IGNORE
            )
            
            # Write the closing code block
            f.write("```\n")
        
        print(f"✅ File structure successfully saved to '{output_filename}'")

    except IOError as e:
        print(f"Error writing to file: {e}")