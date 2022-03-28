import sklearn
from autocorrect import Speller
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import urllib.request
import spacy
import re
import markovify
import matplotlib.pyplot as plt
import collections
from imdb import Cinemagoer
import pandas as pd

nltk.download('punkt', quiet = True)
nltk.download('stopwords', quiet = True)
nltk.download('vader_lexicon', quiet = True)

def main():
    '''Function that contains the entire assignment.'''
    ###Characterizing by Word Frequencies

    print()
    all_movie_reviews = {"data" : {"reviews": []}}
    movie_list = ["The Shawshank Redemption", "The Dark Knight", "Inception", "Fight Club", "Forrest Gump", "Disaster Movie", "House of the Dead", "Date Movie", "Teen Wolf Too", "Gladiator", "The Wolf of Wall Street", "The Lion King", "Smolensk", "Gully", "Turks in Space", "Foodfight!", "Kirk Cameron's Saving Christmas", "The Cost of Deception", "Birdemic: Shock and Terror", "Daniel the Wizard"]

    print("Movie List for IMDB reviews :")
    print()
    #create an instance of the Cinemagoer class
    ia = Cinemagoer()
    #search movie
    for i in range(len(movie_list)):
        current_movie = ia.search_movie(movie_list[i])[0]
        print("Movie Name: " + str(movie_list[i]) + " ID: " + current_movie.movieID)

        movie_reviews = ia.get_movie_reviews(current_movie.movieID)
        for j in range(len(movie_reviews["data"]["reviews"])):
            all_movie_reviews["data"]["reviews"].append(movie_reviews["data"]["reviews"][j])

    def rem_stop(text):
        '''Function that removes all stopwords (common words like a, an, in, etc).'''
        stop_words = set(stopwords.words('english'))
        l = word_tokenize(text)
        l = [word for word in l if word not in stop_words and word.isalnum()]
        l = " ".join(l)
        return(l)


    def clean(document):
        '''Function that cleans and formats text.'''
        document = document.lower()
        spell = Speller(lang='en')
        document = spell(document)  
        document = rem_stop(document)

        return document

    print()
    print("Sentiment analysis using SentimentIntensityAnalyzer()")

    review_data = []
    for i in range(len(all_movie_reviews["data"]["reviews"])):

        review = all_movie_reviews['data']['reviews'][i]['content']

        #clean the review
        current_review = clean(review)
        #define positive 1 and negative 0

        sentiment_score = SentimentIntensityAnalyzer().polarity_scores(current_review)
        positive = 1
        negative = 0

        if sentiment_score["pos"] > sentiment_score["neg"]:
            sentiment = positive
        else:
            sentiment = negative
        print(str(i) + " " + str(sentiment_score))
        #comment the above print statement out
        review_data.append([all_movie_reviews['data']['reviews'][i]['content'], all_movie_reviews['data']['reviews'][i]['rating']])
        review_data.append([all_movie_reviews['data']['reviews'][i]['content'], sentiment])

    imdb_rating_df = pd.DataFrame(review_data, columns = ["review", "sentiment"])

    ###Markov Text Synthesis

    def text_cleaner(text):
        '''Utility function for text cleaning.'''
        text = re.sub(r'--', ' ', text)
        text = re.sub('[\[].*?[\]]', '', text)
        text = re.sub(r'(\b|\s+\-?|^\-?)(\d+|\d*\.\d+)\b','', text)
        text = ' '.join(text.split())
        return text

    print("Getting data: 3 books[Our Mutual Friend, The Life And Adventures Of Nicholas Nickleby, David Copperfield] by Charles Dickens")

    #Our Mutual Friend by Charles Dickens
    our_mutual_friend_url = 'https://www.gutenberg.org/files/883/883-0.txt'
    response = urllib.request.urlopen(our_mutual_friend_url)
    data = response.read()
    our_mutual_friend_text = data.decode('utf-8')
    our_mutual_friend_text = our_mutual_friend_text[our_mutual_friend_text.index("Chapter 1"):]
    our_mutual_friend_text = text_cleaner(our_mutual_friend_text)


    #The Life And Adventures Of Nicholas Nickleby by Charles Dickens
    nicholas_nickleby_url = 'https://www.gutenberg.org/files/967/967-0.txt'
    response = urllib.request.urlopen(nicholas_nickleby_url)
    data = response.read()
    nicholas_nickleby_text = data.decode('utf-8')
    nicholas_nickleby_text = nicholas_nickleby_text[nicholas_nickleby_text.index("CHAPTER 1"):]
    nicholas_nickleby_text = text_cleaner(nicholas_nickleby_text)


    #David Copperfield by Charles Dickens
    david_copperfield_url = 'https://www.gutenberg.org/files/766/766-0.txt'
    response = urllib.request.urlopen(david_copperfield_url)
    data = response.read()
    david_copperfield_text = data.decode('utf-8')
    david_copperfield_text = david_copperfield_text[david_copperfield_text.index("CHAPTER 1"):]
    david_copperfield_text = text_cleaner(david_copperfield_text)

    print("Markov Text Synthesis using the texts from the 3 books: ")

    text_processor = spacy.load('en_core_web_sm')

    our_mutual_friend_doc = text_processor(our_mutual_friend_text[:50000])
    nicholas_nickleby_doc = text_processor(nicholas_nickleby_text[:50000])
    david_copperfield_doc = text_processor(david_copperfield_text[:50000])

    our_mutual_friend_sentences = ' '.join([sentence.text for sentence in our_mutual_friend_doc.sents if len(sentence.text) > 1])
    nicholas_nickleby_sentences = ' '.join([sentence.text for sentence in nicholas_nickleby_doc.sents if len(sentence.text) > 1])
    david_copperfield_sentences = ' '.join([sentence.text for sentence in david_copperfield_doc.sents if len(sentence.text) > 1])
    charles_dickens_sentences = our_mutual_friend_sentences + nicholas_nickleby_sentences + david_copperfield_sentences

    #text generator using Markov Text Synthesis with markovify
    text_generator = markovify.Text(charles_dickens_sentences, state_size = 3)


    #randomly generates three sentences
    print()
    print("Generating 3 new random sentences using Markov Text Synthesis: ")
    print()
    for i in range(3):
        print(text_generator.make_sentence(tries = 100))
    
    ###Computing Summary Statistics

    print()
    print("Summary Statistics")


    #only considering the first 50000 characters
    print()
    print("Only considering first 50000 characters of each of his books: ")

    clean_our_mutual_friend_text = clean(our_mutual_friend_text[:50000])
    clean_nicholas_nickleby_text = clean(nicholas_nickleby_text[:50000])
    clean_david_copperfield_text = clean(david_copperfield_text[:50000])

    def get_word_count(text):
        '''This funtion gets the word count of the text and removes puncuation from that count.'''
        wordcount = {}
        for word in text.lower().split():
            word = word.replace(".","")
            word = word.replace(",","")
            word = word.replace(":","")
            word = word.replace("\"","")
            word = word.replace("!","")
            word = word.replace("*","")
            if word not in wordcount:
                wordcount[word] = 1
            else:
                wordcount[word] += 1
        return wordcount

    most_common_count = 10


    #Our Mutual Friend

    print()
    wordcount_our_mutual_friend = get_word_count(clean_our_mutual_friend_text)
    word_counter_our_mutual_friend = collections.Counter(wordcount_our_mutual_friend)
    print("The 10 most common words in Our Mutual Friend is: ")
    for word, count in word_counter_our_mutual_friend.most_common(most_common_count):
        print(word, ": ", count)

    lst = word_counter_our_mutual_friend.most_common(most_common_count)
    df = pd.DataFrame(lst, columns = ['Word', 'Count'])
    df.plot.bar(x = 'Word', y = 'Count', title = "Our Mutual Friend by Charles Dickens")
    plt.show()

    #The Life And Adventures Of Nicholas Nickleby

    print()
    wordcount_nicholas_nickleby = get_word_count(clean_nicholas_nickleby_text)
    word_counter_nicholas_nickleby = collections.Counter(wordcount_nicholas_nickleby)
    print("The 10 most common words in The Life And Adventures Of Nicholas Nickleby is: ")
    for word, count in word_counter_nicholas_nickleby.most_common(most_common_count):
        print(word, ": ", count)

    lst = word_counter_nicholas_nickleby.most_common(most_common_count)
    df = pd.DataFrame(lst, columns = ['Word', 'Count'])
    df.plot.bar(x = 'Word', y = 'Count', title = "The Life And Adventures Of Nicholas Nickleby by Charles Dickens")
    plt.show()

    #David Copperfield

    print()
    wordcount_david_copperfield = get_word_count(clean_david_copperfield_text)
    word_counter_david_copperfield = collections.Counter(wordcount_david_copperfield)
    print("The 10 most common words in David Copperfield is: ")
    for word, count in word_counter_david_copperfield.most_common(most_common_count):
        print(word, ": ", count)

    lst = word_counter_david_copperfield.most_common(most_common_count)
    df = pd.DataFrame(lst, columns = ['Word', 'Count'])
    df.plot.bar(x = 'Word', y = 'Count', title = "David Copperfield by Charles Dickens")
    plt.show()

if __name__ == "__main__":
    main()