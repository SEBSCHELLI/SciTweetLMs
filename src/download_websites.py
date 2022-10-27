import os
import ast
import pandas as pd
import numpy as np
from tqdm import tqdm
from newspaper import Article, Config
from playwright.sync_api import sync_playwright

tqdm.pandas()

if 'SciTweetLMs' not in os.getcwd():
    os.chdir('/home/schellsn/python/SciTweetLMs')

class WebsiteDownloader:
    def __init__(self):
        user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.134 Safari/537.36 Edg/103.0.1264.71'
        self.newspaper_config = Config()
        self.newspaper_config.browser_user_agent = user_agent

        self.page = None

        self.playwright_domains = ['nytimes.com']

    def download_websites(self, tweets):
        tweets = tweets.sample(n=len(tweets), replace=False)
        assert 'processed_urls' in tweets.columns, "DataFrame needs column 'processed urls' that contains list of resolved urls"

        if type(tweets['processed_urls'].iloc[0]) == str:
            tweets['processed_urls'] = tweets['processed_urls'].apply(lambda urls: ast.literal_eval(urls))

        with sync_playwright() as p:
            browser = p.firefox.launch()
            self.page = browser.new_page()
            tweets['htmls'] = tweets[['processed_urls']].progress_apply(lambda urls: self.extract_article_htmls(*urls), axis=1)
            browser.close()

        return tweets

    def _extracted_with_playwright(self, url):
        for pdo in self.playwright_domains:
            if pdo in url:
                return True

        return False

    def extract_article_htmls(self, urls):
        htmls = []
        for url in urls:
            use_playwright = self._extracted_with_playwright(self, url)

            if use_playwright:
                article_html, exc = self._extract_htmls_playwright(url)
                if exc is None:
                    htmls.append(article_html)
                else:
                    htmls.append(f'Exception: {exc}')
                    print(f'Could not extract html for {url} because of the following exception:\n', exc)

            else: #use newspaper
                article_html, exc = self._extract_htmls_newspaper(url)
                if exc is None:
                    htmls.append(article_html)
                else:
                    htmls.append(f'Exception: {exc}')
                    print(f'Could not extract html for {url} because of the following exception:\n', exc)

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

#cat2h21 = pd.read_csv('weakly_labeled_data/tweets_cat2h21.tsv', sep='\t')
#website_dl = WebsiteDownloader()
#cat2h21_w_htmls2 = website_dl.download_websites(cat2h21)
#cat2h21_w_htmls2.to_csv('weakly_labeled_data/tweets_cat2h21_w_html.tsv', sep='\t', index=False)

cat2h41 = pd.read_csv('weakly_labeled_data/tweets_cat2h41.tsv', sep='\t')
#website_dl = WebsiteDownloader()
#cat2h41_w_htmls2 = website_dl.download_websites(cat2h41)
#cat2h41_w_htmls2.to_csv('weakly_labeled_data/tweets_cat2h41_w_html.tsv', sep='\t', index=False)

tweets_newsoutlet = pd.read_csv('../../heuristics_classifier/weakly_labeled_data/tweets_newsoutlet.tsv', sep='\t')
tweets_newsoutlet2 = pd.read_csv('../../heuristics_classifier/weakly_labeled_data/tweets_cat2h41.tsv', sep='\t')
website_dl = WebsiteDownloader()

tweets_newsoutlet0k10k_w_html = website_dl.download_websites(tweets_newsoutlet[:10000])
tweets_newsoutlet0k10k_w_html.to_csv('weakly_labeled_data/tweets_newsoutlet0k10k_w_html.tsv', sep='\t', index=False)