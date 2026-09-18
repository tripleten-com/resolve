from resolve.llm import ask, MODEL_REVIEW
out = ask('Grade this support reply for tone on a 1 to 5 scale: "Hi, the export link lasts 7 days. Quillstack Support".\n'
          'Reply with JSON only, keys in this order: {"reasoning": "<one sentence>", "score": <1-5>}',
          model=MODEL_REVIEW, max_tokens=120)
print(repr(out))
