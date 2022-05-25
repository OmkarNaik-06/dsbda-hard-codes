#Dividing a block/body of text into words or sentences is known as tokenization

import nltk
nltk.download('punkt')
block = "CSI-DYPIEMR is the Student Chapter of Computer Society of India in Dr. D. Y. Patil Pratishthan's Dr. D. Y. Patil Institute of Engineering, Management, and Research. Computer Society of India is a body of computer professionals in India. It was started on 6 March 1965 by a few computer professionals and has now grown to be the national body representing computer professionals. It has 72 chapters across India, 511 student branches, and 100,000 members."
print("This is word wise tokenization-:",'\n', nltk.word_tokenize(block), '\n')
print("x-o-x-o-x-o-x-o-x-o-x-o-x-o-x-o-x-o-x-o-x-o-x-o-x-o-x-o-x-o-x-o-x-o-x-o-x-o-x-o-x-o-x-o-x-o-x-o-x-o-x-o-x-o-x-o-x", '\n')
print("This is sentence wise tokenization-:",'\n', nltk.sent_tokenize(block))

from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = stopwords.words('english') 
# This function contains the entire list of stop words present inside different languages, for our use case, we'll
#focus on english stopwords
token = nltk.word_tokenize(block)
cleaned_token = []
for word in token:
    if word not in stop_words:
        cleaned_token.append(word)
print("This is the unclean version-:",'\n',  token, '\n')
print("x-o-x-o-x-o-x-o-x-o-x-o-x-o-x-o-x-o-x-o-x-o-x-o-x-o-x-o-x-o-x-o-x-o-x-o-x-o-x-o-x-o-x-o-x-o-x-o-x-o-x-o-x-o-x-o-x", '\n')
print("This is the cleaned version-:",'\n', cleaned_token)

from nltk.stem import PorterStemmer
stemmer = nltk.PorterStemmer()
words = ['rain', 'rained', 'raining', 'rains']
stemmed = [stemmer.stem(word) for word in words]
print(stemmed)

from nltk.stem import WordNetLemmatizer 
nltk.download('wordnet')
nltk.download('omw-1.4')
lemmatizer = nltk.WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(word) for word in cleaned_token]
print(lemmatized)

from nltk import pos_tag 
nltk.download('averaged_perceptron_tagger')
tagged = nltk.pos_tag(cleaned_token)     
print(tagged)

import pandas as pd
import sklearn as sk
import math 
block_1 = "Our aim is to develop a good work culture among students, a culture where students from various technical backgrounds come together to teach, guide and collaborate with each other on various projects and grow together."
block_2 = "Keeping in mind the interest of the IT professionals and computer enthusiasts, CSI works towards making the profession an area of choice amongst all sections of the society. The promotion of Information Technology as a profession is the top priority of CSI today. To fulfill this objective, the CSI regularly organizes conferences, conventions, lectures, projects, and awards. And at the same time, it also ensures that regular training and skill updating are organized for the future IT professionals."
#split so each word have their own string
first_block = block_1.split(" ")
second_block = block_2.split(" ")
#join them to remove common duplicate words
total= set(first_block).union(set(second_block))
print(total)

wordDictA = dict.fromkeys(total, 0) 
wordDictB = dict.fromkeys(total, 0)
for word in first_block:
    wordDictA[word]+=1
for word in second_block:
    wordDictB[word]+=1
pd.DataFrame([wordDictA, wordDictB])

# Now we'll remove stopwords from the list
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
filtered_sentence = [w for w in wordDictA if not w in stop_words]
print(filtered_sentence)

def computeTF(wordDict, doc):
    tfDict = {}
    corpusCount = len(doc)
    for word, count in wordDict.items():tfDict[word] = count/float(corpusCount)
    return(tfDict)
#running our sentences through the tf function:
tfFirst = computeTF(wordDictA, first_block)
tfSecond = computeTF(wordDictB, second_block)
#Converting to dataframe for visualization
tf = pd.DataFrame([tfFirst, tfSecond])
print(tf)

# Now we'll implement the IDF formula
def computeTFIDF(tfBow, idfs):
    tfidf = {}
    for word, val in tfBow.items(): tfidf[word] = val*idfs[word]
    return(tfidf)
#running our two sentences through the IDF:
idfFirst = computeTFIDF(tfFirst, idfs)
idfSecond = computeTFIDF(tfSecond, idfs)
#putting it in a dataframe
idf= pd.DataFrame([idfFirst, idfSecond])
print(idf)

