
import os
import re
import pathlib
import sys 
import click


output_dir = pathlib.Path("D:/Vino/coding/book-manager/bookmanager-kmp/book-manager-kmp/server/src/main/resources/static")

def extract_file_path(line):
    if line.startswith('//'):
        # Remove leading comment markers and extract the file path
        line = line[2:].strip()
    return line

def extract_code_blocks(file_path):
    """Extract all code blocks with their content and line numbers"""
    code_blocks = []
    current_block = None
    
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    for line_num, line in enumerate(lines, 1):
        if '```' in line:
            if current_block is None:
                # Start of code block
                filename = extract_file_path(lines[line_num]).strip()
                language = line.strip()[3:].strip()  # Extract language if specified
                current_block = {
                    'start_line': line_num,
                    'language': language,
                    'content': [],
                    'filename': filename if filename else None
                }
            else:
                # End of code block
                current_block['end_line'] = line_num
                code_blocks.append(current_block)
                current_block = None
        elif current_block is not None:
            # Inside a code block
            current_block['content'].append(line.rstrip('\n'))
    
    # Handle unclosed code block
    if current_block is not None:
        current_block['end_line'] = None
        code_blocks.append(current_block)
    
    return code_blocks

# Usage
# markdown_path = 'example.md'

def do_create_files(blocks):
    for i, block in enumerate(blocks, 1):
        click.echo(f"Code Block {i}:")
        click.echo(f"  Lines: {block['start_line']} to {block['end_line'] or 'end of file'}")
        click.echo(f"  Language: {block['language'] or 'none specified'}")
        click.echo(f"  Filename: {block['filename'] or 'not specified'}")
        click.echo(f"  Content preview: {block['content'][:2]}")  # First 2 lines
        click.echo("  ...") if len(block['content']) > 2 else None
        click.echo(f"last 2 line: {block['content'][-2:]}") if block['content'] else None
        create_file = click.prompt("should I create a file for this code block? y/n", type=bool)
        if create_file:
            # Create the file
            file_path = block['filename']
            file_content = "\n".join(block['content'])
            end_file_path = pathlib.Path(output_dir / file_path)
            click.echo(f"Creating file: {end_file_path}")
            confirm = click.prompt("should I create a file? y/n", type=bool)
            end_file_path.parent.mkdir(parents=True, exist_ok=True)  #
            if confirm:
                with open(end_file_path, "w", encoding="utf-8") as f:
                    f.write(file_content.strip("\n") + "\n")

        # Remove JavaScript/other comment markers before the actual path
        # e.g., "// js/app.js" → "js/app.js"
        # if filepath.startswith("//"):
            # filepath = filepath[2:].strip()

        # # Normalize and create directories if needed
        # full_path = os.path.join(output_dir, filepath)
        # os.makedirs(os.path.dirname(full_path), exist_ok=True)

        # with open(full_path, "w", encoding="utf-8") as f:
        #     f.write(file_content.strip("\n") + "\n")

        # print(f"Created: {full_path}")


@click.command()
@click.argument('markdown_path')
def main(markdown_path):
    print("Arguments:", markdown_path)
    blocks = extract_code_blocks(markdown_path)
    do_create_files(blocks)

if __name__ == "__main__":
    main()