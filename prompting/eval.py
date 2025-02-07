import json
from pathlib import Path

from ssc.pipeline import evaluate_files

output = Path("data/output_classify")
for dir in (
        output / "Model(name='llama3.1:405b', context_size=409, json=True, openai=False)",
        output / "Model(name='gpt-4o-mini', context_size=409, json=False, openai=True)",
        output / "Model(name='gpt-4o', context_size=409, json=False, openai=True)",
        output / "Model(name='o1-mini', context_size=409, json=False, openai=True)",
        output / "Model(name='o1', context_size=409, json=False, openai=True, openrouter=False)",
        output / "Model(name='deepseek/deepseek-r1', context_size=409, json=True, openai=False, openrouter=True)",
):
    print(dir)
    result = evaluate_files([file for file in dir.iterdir() if ".xmi" in file.name], tolerance=3, coarse=True)
    print(result)
    (dir / "results.json").write_text(json.dumps(result, indent=4))