
import pandas as pd
import numpy as np
import re
from nltk import pos_tag, WordNetLemmatizer
from spellchecker import SpellChecker
from multiprocessing import Pool

def pre_process_tweet(string):
    spell = SpellChecker()
    string = string.lower()
    string = re.sub("[/\()?;:,.*&^%$#!1234567890-]", "", string)
    string = while_replace(string)
    lis = string.split()
    final = []
    user_count = 0
    for word in lis:
        if word[0] == "@":
            if user_count < 1:
                final.append("person")
                user_count += 1
            else:
                pass
        elif (word.count("h") >= 2 and word.count("a") >= 2) or "lol" in word:
            final.append("laugh")

        elif len(word) < 2 and word != "i" and word != "a":
                pass
        
        elif "http" in word:
             final.append("website")
        
        else:
            fixed = spell.correction(word)
            final.append(fixed)
    return final

"""Replaces double spaces and tripple characrers for easier split and spellchecking"""
def while_replace(tweet):
    while '  ' in tweet:
        tweet = tweet.replace('  ', ' ')
    new = ""
    for index in range(0,len(tweet)):
        char = tweet[index]
        if index == 0 or index == len(tweet)-1:
            new += char
        else:
            if tweet[index-1] == char and tweet[index+1] == char:
                pass
            else:
                
                new += char
        
    return new

"""Old function/was merged with preprocess"""
def old(lis):
    user_count = 0
    for word in lis:
        if word[0] == "@":
            if user_count < 1:
                lis[lis.index(word)] = "person"
                user_count += 1
            else:
                lis.remove(word)
        elif word.count("h") >= 2 and word.count("a") >= 2:
            lis[lis.index(word)] = "laugh"
        else:
            if len(word) < 2 and word != "i" and word != "a":
                lis.remove(word)
    return lis

"""Conditionals for POS arg in Lemmatizer, returns List"""
def lemmatize(lis):
    lemmatizer = WordNetLemmatizer()
    final = []
    for tup in lis:
        if tup[1] == "JJ" or tup[1] == "JJR" or  tup[1] == "JJS":
            word = lemmatizer.lemmatize(tup[0], pos = "a")
            final.append(word)
        elif tup[1] == "RB" or tup[1] == "RBR" or  tup[1] == "RBS":
            word = lemmatizer.lemmatize(tup[0], pos = "r")
            final.append(word)
        elif tup[1] == "VB" or tup[1] == "VBD" or  tup[1] == "VBG" or tup[1] == "VBN" or tup[1] == "VBP" or tup[1] == "VBZ":
            word = lemmatizer.lemmatize(tup[0], pos = "v")
            final.append(word)
        else:
            word = lemmatizer.lemmatize(tup[0], pos = "n")
            final.append(word)
    return final
"""Function made to deal with data corruption in file transfer to HPC"""
def tagging(tweet):
    try:
        x = pos_tag(tweet)
        return x
    except:
        return None


"""Main Part of program, Loads DF and creates new column for each processing step, saves to excel"""
if __name__ == "__main__":
    df = pd.read_csv("Labeled_Data.csv", encoding='latin-1')
    df.columns = ["Sentiment", "ID", "Date", "Query", "Name", "Tweet"]
    with Pool(12) as p:
        prep_1 = (p.map(pre_process_tweet, df["Tweet"]))
    df["Prep_1"] = prep_1
    with Pool(12) as p:
        prep_2 = (p.map(tagging, df["Prep_1"]))
    df["Prep_2"] = prep_2
    with Pool(12) as p:
        prep_3 = (p.map(lemmatize, df["Prep_2"]))
    df["Prep_3"] = prep_3
    df.to_excel("Final_Data.xlsx")


