"""Record the actual local recorded-demo UI; optional Playwright dependency, outside CPU checks."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output-dir", default="docs/media")
    args = parser.parse_args()
    from playwright.sync_api import sync_playwright

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context(viewport={"width": 1440, "height": 1000}, record_video_dir=str(output / "raw"), record_video_size={"width": 1440, "height": 1000})
        page = context.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.base_url)
        page.wait_for_selector("#demo option[value=crashloop]", state="attached")
        for ident in ("crashloop", "imagepull", "pending"):
            page.select_option("#demo", ident)
            page.click("#load-demo")
            page.wait_for_function("document.getElementById('status').textContent.startsWith('Showing recorded demo')")
            page.locator("#result").get_by_text("RECORDED / AUTHORED EXAMPLE", exact=True).wait_for()
            if ident == "crashloop":
                page.screenshot(path=str(output / "troubleshooting-demo.png"), full_page=True)
            page.wait_for_timeout(1200)
        page.check("#github")
        page.click("#load-demo")
        page.wait_for_function("document.getElementById('result').textContent.includes('recorded snapshot')")
        page.wait_for_timeout(1500)
        video = page.video
        context.close()
        video.save_as(str(output / "troubleshooting-demo.webm"))
        browser.close()
        if errors:
            raise RuntimeError("Browser errors: " + "; ".join(errors))
    print(f"Saved real browser demo screenshot/video in {output}")


if __name__ == "__main__":
    main()
