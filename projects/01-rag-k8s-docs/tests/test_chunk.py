from rag.chunk import chunk_text, parse_markdown


def word_count(text: str) -> int:
    return len(text.split())


def test_parse_strips_frontmatter_and_shortcodes():
    text = "---\ntitle: Pods\ndraft: false\n---\n\n# Pods\n\nA Pod is small.\n\n{{< note >}}\nEphemeral.\n{{< /note >}}\n"
    chunks = parse_markdown(text, "concepts/pods.md", "release-1.34")
    assert len(chunks) == 1
    assert "title:" not in chunks[0].text
    assert "{{<" not in chunks[0].text
    assert "A Pod is small." in chunks[0].text
    assert chunks[0].url == "https://kubernetes.io/docs/concepts/pods/#pods"
    assert chunks[0].heading == "Pods"


def test_parse_skips_draft_pages():
    text = "---\ntitle: Draft\ndraft: true\n---\n\n# Draft\n\nHidden.\n"
    assert parse_markdown(text, "concepts/draft.md", "release-1.34") == []


def test_parse_tracks_nested_heading():
    text = "# Pods\n\nIntro.\n\n## Lifecycle\n\nBody.\n"
    chunks = parse_markdown(text, "concepts/pods.md", "release-1.34")
    headings = [c.heading for c in chunks]
    assert "Pods > Lifecycle" in headings


def test_chunk_text_keeps_code_fence_intact():
    text = "Before.\n\n```yaml\nkind: Pod\n\nmetadata:\n  name: demo\n```\n\nAfter."
    chunks = chunk_text(text, max_tokens=20, overlap=0.1, count_tokens=word_count)
    fenced = [c for c in chunks if "```yaml" in c]
    assert len(fenced) == 1
    assert "metadata:" in fenced[0]


def test_chunk_text_overlaps_blocks():
    text = "one two three four\n\nfive six seven eight\n\nnine ten"
    chunks = chunk_text(text, max_tokens=4, overlap=0.5, count_tokens=word_count)
    assert len(chunks) >= 2
    assert set(chunks[1].split()) & set(chunks[0].split())
