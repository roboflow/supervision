def write_text_to_file(content: str, output_path: str) -> None:
    with open(output_path, "w") as file:
        file.write(content)
