import asyncio
import time
import urllib.parse
import os
import sys
import random
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig, BrowserConfig
from pathlib import Path

PROJECT_ROOT= str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


from src.utils.db_manager import JobDB
from src.utils.logger import logger

#=======================================================
#Scraper
async def main(db_path:str = None, csv_path:str = None):
    if not db_path or not csv_path:
        base_data_path= os.path.join(PROJECT_ROOT, 'data')
        db_path = os.path.join(base_data_path, "jobs.db")
        csv_path = os.path.join(base_data_path, "jobs.csv")

    logger.info("Starting crawler")
    db = JobDB(db_path,csv_path)

    base_url="https://www.linkedin.com/jobs/search"
    search_keywords = [
        "AI Intern",
        "Machine Learning Intern",
        "Data Science Intern",
        "Data Engineer Intern",
        "Data Analyst Intern"
    ]
    chosen_keyword = random.choice(search_keywords)
    params={
        "keywords": chosen_keyword,
        "location":"Ho Chi Minh City Metropolitan Area",
        "trk":"public_jobs_jobs-search-bar_search-submit",
        "position":"1",
        "pagenumber":"0",
    }
    encoded_params = urllib.parse.urlencode(params)

    query_string = '?' + encoded_params
    encoded_url = urllib.parse.urljoin(base_url, "")+query_string

    browser_config = BrowserConfig(
        headless=True,
        user_agent_mode="random",
        java_script_enabled=True,
        viewport_width=1920,
        viewport_height=1080,
    )
    async with AsyncWebCrawler(config=browser_config) as crawler:

        base_config = CrawlerRunConfig(
            css_selector="ul.jobs-search__results-list",
            wait_for="css:ul.jobs-search__results-list",
            delay_before_return_html= 5.0,
            cache_mode=CacheMode.BYPASS
            #verbose=False,
            )
        try:
            t0=time.perf_counter()
            result= await crawler.arun(encoded_url,config=base_config)
            elapsed = time.perf_counter()-t0

            if result.success:
                logger.info(f"Crawler ran successfully | Time : {elapsed:.2f} seconds")

                await asyncio.sleep(random.uniform(2,5))

                soup = BeautifulSoup(result.html, "html.parser")

                job_cards = soup.select ("div.base-card")
                jobs_to_save=[]

                for card in job_cards:
                    title_el=card.select_one("h3.base-search-card__title")
                    comp_el= card.select_one("h4.base-search-card__subtitle")
                    time_el= card.select_one("time.job-search-card__listdate")
                    link_el= card.select_one("a.base-card__full-link")
                    loc_el= card.select_one("span.job-search-card__location")

                    logger.debug(f"Raw card HTML:\n{card.prettify()}")
                    logger.debug(f"title_el: {title_el}")
                    logger.debug(f"comp_el: {comp_el}")
                    logger.debug(f"time_el: {time_el}")
                    logger.debug(f"link_el: {link_el}")
                    logger.debug(f"loc_el: {loc_el}")

                    raw_link = link_el["href"] if link_el else "N/A"

                    clean_link= raw_link.replace("https://vn.linkedin.com", "https://www.linkedin.com")
                    parsed_job= {
                        "title": title_el.get_text(strip=True) if title_el else "N/A",
                        "company": comp_el.get_text(strip=True) if comp_el else "N/A",
                        "location": loc_el.get_text(strip=True) if loc_el else "N/A",
                        "time": time_el.get_text(strip=True) if time_el else "N/A",
                        "link": clean_link,
                    }
                    if parsed_job["title"] != "N/A" and parsed_job["link"] != "N/A":
                        jobs_to_save.append(parsed_job)

                if jobs_to_save:
                    #update db
                    new_jobs_count= db.upsert_jobs(jobs_to_save)
                    #update csv
                    db.export_to_csv(csv_path)
                    logger.info(f"Data Sync: found {len(jobs_to_save)} total jobs. {new_jobs_count} new jobs saved to .db and .csv")
                else:
                    logger.warning("Data sync: No jobs found. Check if selectors have changed of if blocked.")

            else:
                logger.error(f"Crawl Status: FAILED | Problem: {result.error_message}")

        except Exception as e:
            logger.critical(f"System crash: {str(e)}",exc_info=True)

        finally:
            logger.info("End Crawler")

if __name__ == "__main__":
    asyncio.run(main())