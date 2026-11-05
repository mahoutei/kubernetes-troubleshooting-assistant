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
    chunks = chunk_text(text, max_tokens=6, overlap=0.5, count_tokens=word_count)
    assert len(chunks) >= 2
    assert set(chunks[1].split()) & set(chunks[0].split())


def test_parse_ignores_headings_inside_fences():
    text = "# Pods\n\nExample manifest:\n\n```yaml\n# pod name\nkind: Pod\nmetadata:\n  name: demo\n```\n\nAfter the fence.\n"
    chunks = parse_markdown(text, "concepts/pods.md", "release-1.34")
    assert len(chunks) == 1
    assert "```yaml\n# pod name\nkind: Pod" in chunks[0].text
    assert "After the fence." in chunks[0].text
    assert not any(c.url.endswith("#pod-name") for c in chunks)


def test_preamble_is_preserved_before_first_heading():
    chunks = parse_markdown("Important introduction.\n\n# Details\n\nMore details.", "concepts/pods.md", "v")
    assert chunks[0].text == "Important introduction."
    assert chunks[0].url == "https://kubernetes.io/docs/concepts/pods/"
    assert chunks[1].text == "More details."


def test_longer_and_tilde_fences_hide_code_headings():
    for fence in ("````", "~~~"):
        text = f"# Actual heading\n\n{fence}text\n```\n# Inside code\n```\n{fence}\n\nAfter."
        chunks = parse_markdown(text, "concepts/pods.md", "v")
        assert len(chunks) == 1
        assert "# Inside code" in chunks[0].text
        assert "After." in chunks[0].text


def test_indented_fence_preserves_code_immediately_after_heading():
    text = "# Actual heading\n\n   ```yaml\n# Inside code\nkind: Pod\n   ```\n\nAfter."
    chunks = parse_markdown(text, "concepts/pods.md", "v")
    assert len(chunks) == 1
    assert "# Inside code" in chunks[0].text
    assert "After." in chunks[0].text


def test_identical_repeated_sections_do_not_duplicate_chunk_ids():
    chunks = parse_markdown("# Pods\n\nSame content.\n\n# Pods\n\nSame content.", "concepts/pods.md", "v")
    assert len(chunks) == 1


def test_normal_prose_and_overlap_stay_within_budget():
    text = " ".join(f"word{i}" for i in range(60)) + "\n\nAnother paragraph with several words."
    chunks = chunk_text(text, 10, 0.2, word_count)
    assert all(word_count(chunk) <= 10 for chunk in chunks)
    assert all(f"word{i}" in " ".join(chunks) for i in range(60))


def test_oversized_code_is_preserved_for_explicit_truncation_reporting():
    text = "```yaml\n" + "value: example\n" * 20 + "```"
    assert chunk_text(text, 10, 0.2, word_count) == [text]
