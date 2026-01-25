from playwright.sync_api import sync_playwright, expect
import time

def verify_frontend():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Navigate to the app
        page.goto("http://localhost:5000")

        # 1. Verify Top 30 List has Region Info (if not Fortaleza)
        # We need to wait for data to load
        expect(page.locator("#top-areas")).to_be_visible(timeout=20000)

        # Wait for list items
        page.wait_for_selector("#top-areas a.list-group-item")

        # Check if we can find a city name suffix.
        # Since the data is dynamic (real or random fallback), we might not see "Sobral" unless it's in top 30.
        # But we added "Sobral" via test previously, if app.py is running on same env, it might have it?
        # The app.py started fresh, so it reloaded from file.
        # If EXOGENOUS_FILE has the Sobral event, risk might be high there.

        # Let's take a screenshot of the Dashboard
        page.screenshot(path="verification/dashboard_initial.png")

        # 2. Open Exogenous Modal
        page.click("button[data-bs-target='#exogenousModal']")
        expect(page.locator("#exogenousModal")).to_be_visible()

        # 3. Enter Text
        page.fill("#ciops-text", "01 - TEST - CONFLITO - TESTE - NATUREZA - ENV - SOBRAL - DP - TIME")

        # 4. Click Process
        # Note: This will fail if our mock parsing in finding coordinates fails.
        # 'Sobral' should be found by our new region logic?
        # find_node_coordinates uses name match or caches.
        # Sobral is in the cache.

        # We need to mock the alert to not block execution
        page.on("dialog", lambda dialog: dialog.accept())

        page.click("#btn-process-ciops")

        # Wait for loading overlay
        expect(page.locator("#loading-overlay")).to_be_visible()

        # Wait for it to disappear
        expect(page.locator("#loading-overlay")).not_to_be_visible(timeout=30000)

        # Take Screenshot after processing
        page.screenshot(path="verification/dashboard_after_save.png")

        browser.close()

if __name__ == "__main__":
    verify_frontend()
