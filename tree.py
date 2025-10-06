import pathlib
import sys
from typing import TextIO

def generate_file_structure(
    root_dir: str,
    file_handle: TextIO,
    ignore_list: list[str] | None = None,
    prefix: str = "",
):
    """
    Generates a tree-like structure for a directory and writes it to a file.

    Args:
        root_dir (str): The path to the root directory.
        file_handle (TextIO): The file object to write the output to.
        ignore_list (list[str], optional): A list of names to ignore.
        prefix (str, optional): The prefix for drawing the tree structure.
    """
    if ignore_list is None:
        ignore_list = ["__pycache__", ".git", ".github", ".vscode", "build", "dist"]

    dir_path = pathlib.Path(root_dir)
    if not dir_path.is_dir():
        print(f"Error: '{dir_path}' is not a valid directory.")
        return

    paths = sorted(
        [p for p in dir_path.iterdir() if p.name not in ignore_list],
        key=lambda p: (not p.is_dir(), p.name.lower()),
    )

    for i, path in enumerate(paths):
        is_last = i == len(paths) - 1
        connector = "└── " if is_last else "├── "
        
        if path.is_dir():
            print(f"{prefix}{connector}📁 {path.name}/", file=file_handle)
            new_prefix = prefix + ("    " if is_last else "│   ")
            generate_file_structure(path, file_handle, ignore_list, prefix=new_prefix)
        else:
            print(f"{prefix}{connector}📄 {path.name}", file=file_handle)


if __name__ == "__main__":
    # The directory to scan (defaults to current directory)
    start_path = sys.argv[1] if len(sys.argv) > 1 else "."
    
    # The output filename (defaults to STRUCTURE.md)
    output_filename = sys.argv[2] if len(sys.argv) > 2 else "STRUCTURE.md"
    
    root_name = pathlib.Path(start_path).resolve().name

    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            # Write the Markdown header and opening code block
            f.write("# Project File Structure\n\n```\n")
            
            # Write the root directory and start the tree
            print(f"📁 {root_name}/", file=f)
            generate_file_structure(start_path, file_handle=f)
            
            # Write the closing code block
            f.write("```\n")
        
        # Print a confirmation message to the console
        print(f"✅ File structure successfully saved to '{output_filename}'")

    except IOError as e:
        print(f"Error writing to file: {e}")