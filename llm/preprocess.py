import re
from nltk.corpus import stopwords

class TextPreprocessor:
    def __init__(self):
        self.punctuations = r'''!()-[]{};:'"\,<>./?@#$%^&*_“~='ã¶``'''
        self.stop_words = set(stopwords.words("english"))

    def text_preprocessing(self, text):
        '''
        A method to preprocess text
        '''
        for x in text.lower():
            if x in self.punctuations:
                text = text.replace(x, "")

        # removing words that have numbers in them
        text = re.sub(r'\w*\d\w*', '', text)
        # remove digits
        text = re.sub(r'[0-9]+', ' ', text)

        # clean the whitespaces
        text = re.sub(r'\s+', ' ', text).strip()

        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  
                                   u"\U0001F300-\U0001F5FF"  
                                   u"\U0001F680-\U0001F6FF"  
                                   u"\U0001F700-\U0001F77F"  
                                   u"\U0001F780-\U0001F7FF"  
                                   u"\U0001F800-\U0001F8FF"  
                                   u"\U0001F900-\U0001F9FF"  
                                   u"\U0001FA00-\U0001FA6F"  
                                   u"\U0001FA70-\U0001FAFF"  
                                   u"\U00002702-\U000027B0"  
                                   u"\U000024C2-\U0001F251" 
                                   "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)

        # lowercase eth
        text = text.lower()

        # drop the stop words

        # add the tags
        return text
