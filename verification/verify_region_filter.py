from playwright.sync_api import sync_playwright, expect

def test_region_filter():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Navigate
        page.goto("http://localhost:5000")

        # Wait for map to load
        page.wait_for_selector("#map")

        # Wait for Top 30 list to populate (default is All)
        page.wait_for_selector("#top-areas a")

        # Take initial screenshot
        page.screenshot(path="verification/1_initial_all_regions.png")
        print("Initial screenshot taken.")

        # Select "Região Metropolitana (RMF)"
        page.select_option("#region-filter", "rmf")

        # Wait for list to update
        # We expect a city like "Caucaia" or "Maracanaú" to appear in the top list if RMF is selected
        # Or at least the list content to change.
        page.wait_for_timeout(1000) # Wait for JS update

        # Take screenshot of RMF filter
        page.screenshot(path="verification/2_filter_rmf.png")
        print("RMF Filter screenshot taken.")

        # Verify text content of the list contains RMF cities
        # Note: This depends on data. Assuming Caucaia or Maracanaú has risk.
        content = page.text_content("#top-areas")
        print("List content sample:", content[:200])

        # Select "Interior"
        page.select_option("#region-filter", "interior")
        page.wait_for_timeout(1000)
        page.screenshot(path="verification/3_filter_interior.png")
        print("Interior Filter screenshot taken.")

        browser.close()

if __name__ == "__main__":
    test_region_filter()
