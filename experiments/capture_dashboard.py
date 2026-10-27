"""Capture evidence screenshots of the running dashboard (needs API on :8000, Streamlit on :8501).
Uses Playwright with the locally installed Microsoft Edge."""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "docs" / "evidence")
OUT.mkdir(parents=True, exist_ok=True)
URL = "http://127.0.0.1:8501"


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1500, "height": 1100})
        page.goto(URL)
        page.get_by_text("Replay next segments").wait_for(timeout=60000)

        page.get_by_role("button", name="Replay next segments").click()
        page.get_by_text("Recent segment verdicts").wait_for(timeout=60000)
        page.wait_for_timeout(2000)
        page.screenshot(path=OUT / "dashboard_telemetry.png", full_page=True)

        page.get_by_role("tab", name="Physics twin").click()
        page.get_by_text("Orbital period").wait_for(timeout=60000)
        # inject a 50% solar-panel loss (slider step 0.1) so twin and measurement diverge
        panel = (page.locator('[data-testid="stSlider"]').filter(has_text="panel loss")
                 .locator('input[type="range"]'))
        panel.focus()
        for _ in range(5):
            panel.press("ArrowRight")
        page.wait_for_timeout(4000)
        page.screenshot(path=OUT / "dashboard_physics.png", full_page=True)

        page.get_by_role("tab", name="Downlink schedule").click()
        page.get_by_role("button", name="Plan downlinks").click()
        page.get_by_text("Weighted MB").wait_for(timeout=120000)
        page.wait_for_timeout(2000)
        page.screenshot(path=OUT / "dashboard_schedule.png", full_page=True)
        browser.close()
    print("saved to", OUT)


if __name__ == "__main__":
    main()
