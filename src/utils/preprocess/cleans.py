import re

def replace_symbols(text):
    text = text.replace("｜", "")
    text = text.replace('\u3000', '')
    text = re.sub(r"※［＃.*?］", r"", text)
    text = re.sub(r"［＃.*?］", r"", text)
    text = re.sub(r"《.*?》", r"", text)
    return text.strip()

def remove_header(text):
    match = re.search(r'-{5,}\n[\s\S]*?-{5,}\n([\s\S]*)', text)
    text = match.group(1)
    return text.strip()

def remove_footer(text):
    text = '\n'.join(x for x in text.split('\n\n')[:-1] if len(x) > 5)
    return text