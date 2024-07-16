from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def download_all_step_files(base_url):
    # Set up Selenium Chrome driver
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode, if desired
    service = ChromeService(executable_path='/Users/modeh/Downloads/chromedriver-mac-arm64/chromedriver')  # Replace with your chromedriver path
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        # Navigate to the base URL
        driver.get(base_url)
        print(f"Opened base URL: {base_url}")

        # Function to recursively find and download .STEP files
        def find_and_download_step_files(url):
            # Find all download links with .step in href
            step_links = driver.find_elements(By.XPATH, '//a[contains(@href, ".step")]')

            # Iterate through each link and download
            for link in step_links:
                try:
                    # Click each link to download
                    link.click()
                    print(f"Downloading file: {link.get_attribute('href')}")

                    # Optionally, handle download dialog or wait for download completion
                    # Additional code can be added here as needed

                except Exception as e:
                    print(f"Failed to download file: {link.get_attribute('href')}. Error: {str(e)}")

            # Find next page link and continue recursively
            next_page_link = driver.find_element(By.XPATH, '//a[text()="Next"]')
            if next_page_link:
                next_page_url = next_page_link.get_attribute('href')
                driver.get(next_page_url)
                print(f"Navigating to next page: {next_page_url}")
                find_and_download_step_files(next_page_url)

        # Start recursion from the base URL
        find_and_download_step_files(base_url)

    finally:
        driver.quit()

# Example base URL:
base_url = 'https://www.mcmaster.com/mvC/Library/M4/'
download_all_step_files(base_url)
