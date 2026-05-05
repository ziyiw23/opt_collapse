from pathlib import Path


def test_readme_mentions_pipeline():
    text = Path('README.md').read_text()
    assert 'optimized embedding' in text.lower()
    assert 'gen_embed.py' in text
