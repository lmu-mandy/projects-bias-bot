class Sample:
    """class to store data about each collected sample"""
    def __init__(self, article_header, article_text, source, label, party):
        """
        initialize class

        article_header: list
            tokenized article headline
        article_text: list
            tokenized article body text
        source: string
            article news source
        label: string
            bias label assigned to the article
        party: string
            the political affiliation of the person who assigned the bias score
        """
        self.headline = article_header
        self.text = article_text
        self.source = source
        self.label = label
        self.party = party