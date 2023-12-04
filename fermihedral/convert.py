import os
import subprocess

def convert_svg_to_pdf(input_file, output_file):
    subprocess.run(["inkscape", input_file, "--export-type=pdf", f"--export-filename={output_file}"])

def batch_convert_svg_to_pdf(input_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith(".svg"):
            input_path = os.path.join(input_dir, filename)
            output_filename = os.path.splitext(filename)[0] + ".pdf"
            output_path = os.path.join(input_dir, output_filename)
            convert_svg_to_pdf(input_path, output_path)
            print(f"Converted {filename} to {output_filename}")

# Use the current directory as the input directory
input_directory = os.getcwd()

batch_convert_svg_to_pdf(input_directory)
