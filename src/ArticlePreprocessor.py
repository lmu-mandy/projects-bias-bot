from bs4 import BeautifulSoup
import os

class ArticlePreprocessor:
    ARTICLE_PATHS = ['../data/batch_one', '../data/batch_two', '../data/batch_three', '../data/batch_four', '../data/batch_six', '../data/batch_seven']
    SAMPLE_PATHS = []
    WRITE_PATH = "../data"
    def __init__(self, article_paths=ARTICLE_PATHS, sample_paths=SAMPLE_PATHS, write_path=WRITE_PATH):
        self.article_paths = article_paths
        self.sample_paths = sample_paths
        self.write_path = write_path
        self.duplicate_dict, self.article_outlets = self.remove_and_store_duplicates()
        self.file_locations = self.write_preprocessed_articles()

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

            file_locations[new_path] = URL
        return file_locations

    def get_preprocessed_df(self):
        


if __name__ == "__main__":
    preprocessor = ArticlePreprocessor()
