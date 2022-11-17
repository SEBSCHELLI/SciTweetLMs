import os
import ast
import pandas as pd
import numpy as np
from tqdm import tqdm
from newspaper import Article, Config
from playwright.sync_api import sync_playwright
import requests

tqdm.pandas()

if 'SciTweetLMs' not in os.getcwd():
    os.chdir('/home/schellsn/python/SciTweetLMs')

class WebsiteDownloader:
    def __init__(self):
        user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.134 Safari/537.36 Edg/103.0.1264.71'
        self.newspaper_config = Config()
        self.newspaper_config.browser_user_agent = user_agent
        self.newspaper_config.fetch_images = False
        self.newspaper_config.request_timeout = 3

        self.page = None

        self.playwright_domains = ['nytimes.com']

        self.session = requests.Session()
        self.session.headers = {'User-Agent': user_agent}

    def download_websites(self, tweets):
        tweets = tweets.sample(n=len(tweets), replace=False)
        assert 'processed_urls' in tweets.columns, "DataFrame needs column 'processed urls' that contains list of resolved urls"

        if type(tweets['processed_urls'].iloc[0]) == str:
            tweets['processed_urls'] = tweets['processed_urls'].apply(lambda urls: ast.literal_eval(urls))

        with sync_playwright() as p:
            browser = p.firefox.launch(timeout=3000)
            self.page = browser.new_page()
            tweets['htmls'] = tweets[['processed_urls']].progress_apply(lambda urls: self.extract_article_htmls(*urls), axis=1)
            browser.close()

        return tweets

    def _use_playwright(self, url):
        for pdo in self.playwright_domains:
            if pdo in url:
                return True

        return False

    def _check_not_404(self, url):
        try:
            r = self.session.head(url, allow_redirects=False, timeout=3)
            return r.status_code != 404
        except Exception as exc:
            print(f'Could not check status for {url} because of the following exception:\n', exc, '\n')
            return False


    def extract_article_htmls(self, urls):
        htmls = []
        for url in urls:

            if self._check_not_404(url):
                use_playwright = self._use_playwright(url)

                if use_playwright:
                    article_html, exc = self._extract_htmls_playwright(url)
                    if exc is None:
                        htmls.append(article_html)
                    else:
                        htmls.append(f'Exception: {exc}')
                        print(f'Could not extract html for {url} because of the following exception:\n', exc, '\n')

                else: #use newspaper
                    article_html, exc = self._extract_htmls_newspaper(url)
                    if exc is None:
                        htmls.append(article_html)
                    else:
                        htmls.append(f'Exception: {exc}')
                        print(f'Could not extract html for {url} because of the following exception:\n', exc, '\n')
            else:
                htmls.append(f'Exception: Timeout')

        return htmls

    def _extract_htmls_playwright(self, url):
        try:
            self.page.goto(url)
            article_html = self.page.content()
            article_html = article_html.replace('"', "").replace("'", "")

            return article_html, None

        except Exception as e:
            return e, e

    def _extract_htmls_newspaper(self, url):
        try:
            article = Article(url, config=self.newspaper_config)
            article.download()
            article_html = article.html.replace('"', "").replace("'", "")
            exc = article.download_exception_msg
            if article_html == "" and exc is not None:
                article_html = exc

            return article_html, exc

        except Exception as e:
            return e, e

tweets_newsoutlet = pd.read_csv('tweets_newsoutlet.tsv', sep='\t')
website_dl = WebsiteDownloader()


for i in range(1,10):
    print(i)
    tweets_newsoutlet0k1k_w_html = website_dl.download_websites(tweets_newsoutlet[i*1000:(i+1)*1000])
    tweets_newsoutlet0k1k_w_html.to_csv(f'tweets_newsoutlet{i}k{i+1}k_w_html.tsv', sep='\t', index=False)


tweets_newsoutlet0k1k_w_html['htmlsstr'] = tweets_newsoutlet0k1k_w_html['htmls'].apply(lambda x: str(x)[:200])
tweets_newsoutlet0k1k_w_html['htmlsstr'].value_counts().head(10)

#guardian blocks us
#washingtonpost timeout

#cat2h21 = pd.read_csv('weakly_labeled_data/tweets_cat2h21.tsv', sep='\t')
#website_dl = WebsiteDownloader()
#cat2h21_w_htmls2 = website_dl.download_websites(cat2h21)
#cat2h21_w_htmls2.to_csv('weakly_labeled_data/tweets_cat2h21_w_html.tsv', sep='\t', index=False)

#cat2h41 = pd.read_csv('weakly_labeled_data/tweets_cat2h41.tsv', sep='\t')
#website_dl = WebsiteDownloader()
#cat2h41_w_htmls2 = website_dl.download_websites(cat2h41)
#cat2h41_w_htmls2.to_csv('weakly_labeled_data/tweets_cat2h41_w_html.tsv', sep='\t', index=False)
