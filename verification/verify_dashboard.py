from playwright.sync_api import sync_playwright, expect

def test_dashboard(page):
    # 1. Acessar o dashboard (assumindo que o flask está rodando na porta 5000)
    page.goto("http://localhost:5000")

    # 2. Verificar Título
    expect(page).to_have_title("Monitoramento de Inteligência - Segurança Pública")

    # 3. Verificar se o Mapa carregou
    # O leaflet cria uma div com classe leaflet-container
    expect(page.locator(".leaflet-container")).to_be_visible()

    # 4. Verificar se os KPIs estão visíveis
    expect(page.locator("#high-risk-count")).to_be_visible()
    expect(page.locator("#medium-risk-count")).to_be_visible()
    expect(page.locator("#low-risk-count")).to_be_visible()

    # 5. Esperar um pouco para os dados carregarem (chamada AJAX)
    page.wait_for_timeout(2000)

    # 6. Screenshot
    page.screenshot(path="verification/dashboard.png")

if __name__ == "__main__":
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            test_dashboard(page)
        except Exception as e:
            print(f"Erro na verificação: {e}")
        finally:
            browser.close()
