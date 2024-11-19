import re

def format_text(text):
  """Lowercases and strips punctuation."""
  text = text.lower().strip()
  if text == "n/a" or text == "?":
    text = "EMPTY"

  text = re.sub(r"[^\w\d]+", " ", text).replace("_", " ")
  text = " ".join(text.split())
  text = text.strip()
  if text:
    return text
  return "EMPTY"