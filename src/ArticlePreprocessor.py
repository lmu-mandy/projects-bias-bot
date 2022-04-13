from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import os
import re

class ArticlePreprocessor:
    ARTICLE_PATHS = ['../data/raw_articles/batch_one', '../data/raw_articles/batch_two', '../data//raw_articles/batch_three', 
                    '../data//raw_articles/batch_four', '../data/raw_articles/batch_six', '../data/raw_articles/batch_seven',
                    '../data/raw_articles/batch_eight']
    SAMPLE_PATHS = ['../data/samples/sample_1.csv', '../data/samples/sample_2.csv', '../data/samples/sample_3.csv', '../data/samples/sample_4.csv',
                    '../data/samples/sample_4.csv', '../data/samples/sample_5.csv', '../data/samples/sample_6.csv', '../data/samples/sample_7.csv', 
                    '../data/samples/sample_8.csv', '../data/samples/sample_9.csv', '../data/samples/sample_10.csv'
                    ]
    WRITE_PATH = "../data/preprocessed"
    def __init__(self, reprocess=False, article_paths=ARTICLE_PATHS, sample_paths=SAMPLE_PATHS, write_path=WRITE_PATH):
        self.write_path = write_path
        if reprocess:
            self.article_paths = article_paths
            self.sample_paths = sample_paths
            self.duplicate_dict, self.article_outlets = self.remove_and_store_duplicates()
            self.file_locations = self.write_preprocessed_articles()
            self.preprocess_samples()

        self.article_counts = self.get_article_counts()

    def remove_and_store_duplicates(self):
        duplicate_dict = {}
        article_outlets = {}
        for batch in self.article_paths:
            for article_num in range(0,500):
                full_path = f"{batch}/{article_num}.txt"
                article = open(full_path, "r+")
                article_contents = article.read()
                article_soup = BeautifulSoup(article_contents, features="html.parser")

                outlet_identifier_dict = {"https://www.foxnews.com": "FOX", "https://www.cnn.com": "CNN", "https://www.bbc.co.uk": "BBC"}
                outlet = ""
                article_URL = ""

                try:
                    article_URL = article_soup.find("input", {"id": "url"}).get('value')
                    for key in outlet_identifier_dict.keys():
                        if key in article_URL:
                            outlet = outlet_identifier_dict[key]
                            article_outlets[article_URL] = outlet_identifier_dict[key]
                    if outlet == "":
                        raise Exception("URL did not contain appropriate url identifier")
                except Exception as e:
                    raise Exception(f"There was an error: {e}")
                
                if article_URL in duplicate_dict:
                    duplicate_dict[article_URL].append(full_path)
                else:
                    duplicate_dict[article_URL] = [full_path]
        
        return duplicate_dict, article_outlets
    
    def write_preprocessed_articles(self):
        outlet_path_count = {"FOX": 0, "CNN": 0, "BBC": 0}
        file_locations = {}
        if not os.path.exists(f"{self.write_path}"):
            os.mkdir(f"{self.write_path}")
        for URL in self.duplicate_dict.keys():
            outlet = self.article_outlets[URL]
            new_path = f"{self.write_path}/{outlet}/{outlet_path_count[outlet]}.txt"
            outlet_path_count[outlet] = outlet_path_count[outlet] + 1

            if not os.path.exists(f"{self.write_path}/{outlet}"):
                os.mkdir(f"{self.write_path}/{outlet}")

            if not os.path.isfile(f"{self.write_path}/{outlet}/{outlet_path_count[outlet]}.txt"):
                old_location = self.duplicate_dict[URL][0]
                article = open(old_location, "r+")
                article_contents = article.read()

                new_file = open(new_path, "w")
                new_file.write(article_contents)

            file_locations[URL] = new_path
        return file_locations

    def preprocess_samples(self):
        data = pd.DataFrame()
        for sample_path in self.sample_paths:
            new_samples = pd.read_csv(sample_path)
            new_samples = new_samples.dropna(subset=['Answer.url'])
            article_location = np.empty(0)
            for _, row in new_samples.iterrows():
                article_location = np.append(article_location, self.file_locations[row['Answer.url']])

            new_samples["Article Location"] = article_location
            data = pd.concat([data, new_samples])

        new_sample_data = data.to_csv()
        new_sample_file = open(f"{self.write_path}/samples.csv", "w")
        new_sample_file.write(new_sample_data)

    def get_article_counts(self):
        article_counts = {}
        BBC_articles = os.listdir(f"{self.write_path}/BBC")
        FOX_articles = os.listdir(f"{self.write_path}/FOX")
        CNN_articles = os.listdir(f"{self.write_path}/CNN")

        article_counts["BBC"] = max([int(re.findall("\d+", article)[0]) for article in BBC_articles])
        article_counts["CNN"] = max([int(re.findall("\d+", article)[0]) for article in CNN_articles])
        article_counts["FOX"] = max([int(re.findall("\d+", article)[0]) for article in FOX_articles])

        return article_counts
