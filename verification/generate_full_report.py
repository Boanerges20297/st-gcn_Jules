from playwright.sync_api import Page, expect, sync_playwright
import time
import os

SCREENSHOT_DIR = "verification/report"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

def run_report_simulation(page: Page):
    # 1. Arrange: Go to the app.
    page.goto("http://localhost:5000")

    # Wait for map to load
    page.wait_for_selector(".leaflet-container")
    print("Map loaded.")

    # Wait for polygons
    page.wait_for_selector("path.leaflet-interactive", timeout=15000)
    print("Polygons loaded.")

    # 2. Screenshot 1: Full Screen Initial (CVLI)
    page.screenshot(path=f"{SCREENSHOT_DIR}/1_initial_cvli.png")
    print("Saved 1_initial_cvli.png")

    # Switch to CVP
    page.locator("label[for='mode-switch']").click()
    time.sleep(1)
    page.screenshot(path=f"{SCREENSHOT_DIR}/1_initial_cvp.png")
    print("Saved 1_initial_cvp.png")

    # Switch back to CVLI
    page.locator("label[for='mode-switch']").click()
    time.sleep(1)

    # 3. Screenshot 2: Zoom Interaction
    # Click on the first item in Top 30 list
    print("Clicking top critical area...")
    top_area = page.locator("#top-areas a").first
    top_area.click()
    time.sleep(2) # Wait for flyTo
    page.screenshot(path=f"{SCREENSHOT_DIR}/2_zoomed_area.png")
    print("Saved 2_zoomed_area.png")

    # Double click to zoom out (reset)
    print("Double clicking to zoom out...")
    map_el = page.locator("#map")
    map_el.dblclick()
    time.sleep(2)
    page.screenshot(path=f"{SCREENSHOT_DIR}/2_zoom_out_reset.png")

    # 4. Screenshot 3: Suppression Simulation
    print("Activating Suppression Simulation...")
    page.get_by_role("button", name="Equipe").click()

    # Click on map (center-ish)
    map_el.click(position={"x": 400, "y": 300})
    time.sleep(0.5)
    map_el.click(position={"x": 420, "y": 320}) # Second bike

    # Wait for processing
    time.sleep(2)
    page.screenshot(path=f"{SCREENSHOT_DIR}/3_simulation_suppression.png")
    print("Saved 3_simulation_suppression.png")

    # Clear simulation
    page.get_by_role("button", name="Limpar Simulação").click()
    time.sleep(2)

    # 5. Screenshot 4: Exogenous Simulation
    print("Activating Exogenous Conflict Simulation...")
    page.get_by_role("button", name="Conflito").click()

    # Click on map (different area)
    map_el.click(position={"x": 350, "y": 350})

    # Wait for processing
    time.sleep(2)
    page.screenshot(path=f"{SCREENSHOT_DIR}/4_simulation_exogenous.png")
    print("Saved 4_simulation_exogenous.png")

if __name__ == "__main__":
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        # Set viewport to a standard desktop size
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        try:
            run_report_simulation(page)
            print("Report generation finished.")
        except Exception as e:
            print(f"Report failed: {e}")
            page.screenshot(path=f"{SCREENSHOT_DIR}/error.png")
        finally:
            browser.close()
