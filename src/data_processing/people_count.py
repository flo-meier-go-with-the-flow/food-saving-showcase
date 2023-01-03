import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup


def get_people_count(url="https://live.xovis.cloud/de/e8f4d56a-75aa-419b-a285-38cb55840209", delay=0.1):
    """ Scrape the Zühlke Xovis webpage for the Schlieren office and return the current people count as
    an integer.
    Args:
        url: The URL to the Xovis page for the Schlieren office.
        delay: The time in seconds to wait until the Xovis page is loaded.
    Return:
        The current people count in the Schlieren office
    """
    options = Options()
    options.add_argument('--headless')  # Run in headless mode, so that the browser is not opened.
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    time.sleep(delay)  # Needs this delay, otherwise the people count is not loaded yet.
    html_source = driver.page_source

    parsed_html = BeautifulSoup(html_source, 'lxml')

    people_count = parsed_html.body.find('span', {'class': 'people-value'}).text

    return int(people_count)


if __name__ == '__main__':
    print(get_people_count())