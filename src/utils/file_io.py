import re

def extract_id_from_filename(filename: str) -> str:
    match = re.search(r'(\d+)\.txt$', filename)
    if match:
        return match.group(1)
    else:
        raise ValueError("ファイルパスにnovel_idが見つかりませんでした。")