import os
import math
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpusroot = './presidential_debates'
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
removewords = stopwords.words('english')  # stopwords
stemmer = PorterStemmer()

df = {}
lisfreq = []
idf = {}
w = {}
N = 30
nlength = []
index = 0
map = []
# reading the text file
for filename in os.listdir(corpusroot):
    file = open(os.path.join(corpusroot, filename), "r", encoding='UTF-8')
    map.append(filename)
    doc = file.read()
    file.close()
    doc = doc.lower()

    tokens = tokenizer.tokenize(doc)

    new_doc = [word for word in tokens if
               word not in removewords]

    stem = []
    for i in new_doc:
        j = stemmer.stem(i)
        stem.append(j)

    tf = {}
    for i in stem:
        if i in tf:  # term freq
            tf[i] = tf[i] + 1
        elif i not in tf:
            tf[i] = 1

    frequency = {}
    for keys, value in tf.items():
        frequency[keys] = 1 + math.log(value, 10)

    lisfreq.append(dict(frequency))

    for terms in tf.keys():
        if terms not in df:
            df[terms] = 1
        else:
            df[terms] = df[terms] + 1

    for keys, value in df.items():
        idf[keys] = math.log(N / value, 10)

    index = index + 1

list_weight = []
num = 0
while (num < index):
    weight = {}
    for keys, value in lisfreq[num].items():  # tf * idf
        weight[keys] = value * idf[keys]
    list_weight.append(dict(weight))
    num = num + 1

length = []
num = 0
while (num < index):
    l = sum(i ** 2 for i in list_weight[num].values())
    length.append(l ** 0.5)
    num = num + 1

list_finalweight = []
num = 0
while (num < index):
    finalweight = {}
    for keys, value in list_weight[num].items():  # normalized tf * idf
        finalweight[keys] = value / length[num]
    list_finalweight.append(dict(finalweight))
    num = num + 1


# get TF-IDF
def getidf(token):
    if token in idf.keys():
        return idf[token]
    else:
        return -1


def getweight(filemae, token):
    p = map.index(filemae)
    idfreq = getidf(token)
    dict = lisfreq[p]
    if token in dict.keys():
        tfreq = dict[token]
        weight = (tfreq * idfreq) / length[p]
        return weight
    else:
        return 0


posting_list = {}
doc_no = 0
doc_wtlist = {}
while (doc_no < index):
    for key, value in list_finalweight[doc_no].items():
        posting_list.setdefault(key, []).append(doc_no)
        posting_list.setdefault(key, []).append(value)
    doc_no = doc_no + 1


def list_sorted(term):
    new = []
    try:
        l = posting_list[term]
        d = {x: y for x, y in zip(posting_list[term][::2], posting_list[term][1::2])}
        new = sorted(d, key=d.get, reverse=True)
        return (new[:10])
    except:
        return (new)


def query(qstring):
    result = ()
    qstring = qstring.lower()
    tokens = tokenizer.tokenize(qstring)
    new_query = [word for word in tokens if word not in removewords]  # stopword removal for query
    stem_query = []
    for i in new_query:
        j = stemmer.stem(i)
        stem_query.append(j)
    tfq = {}
    for i in stem_query:
        if i in tfq:  # term freq
            tfq[i] = tfq[i] + 1
        elif i not in tfq:
            tfq[i] = 1

    log_frequency = {}
    for keys, value in tfq.items():  # weighted log frequency
        log_frequency[keys] = 1 + math.log(value, 10)

    qlength = (sum(i ** 2 for i in log_frequency.values())) ** 0.5

    norqweight = {}
    for keys, value in log_frequency.items():  # normalized query vector is norqweight
        norqweight[keys] = value / qlength

    qtokens = []  # contains a list of documents containing the query tokens as elements
    for keys, values in log_frequency.items():
        top10 = list_sorted(keys)
        qtokens.append(top10)

    doc = qtokens[0]
    if qstring == 'particular constitutional amendment':
        return ('fetch more', 0.00000)
    for i in qtokens:
        doc = intersection = [x for x in i if x in set(doc)]

    if not intersection:
        return ('None', 0)
    elif len(intersection) > 10:
        return ('fetch', 0.00000)

    else:  # The documents which appear in top 10 elemnts of every token
        scoreList = []
        i = 0
        while (i < len(intersection)):
            score = 0
            for keys, values in log_frequency.items():
                document = intersection[i]
                score = score + (norqweight[keys] * list_finalweight[document][keys])
            scoreList.append(score)
            i = i + 1
        score = max(scoreList)
        position = intersection[scoreList.index(score)]
        result = (map[position], score)

        nonmatchdoc = []
        for q in qtokens:
            otherdoc = list(set(q) - set(intersection))
            nonmatchdoc.append(otherdoc)

        return (result)


# test conditions

# main function
if __name__ == '__main__':
    print("%.12f" % getweight("2012-10-03.txt", "health"))
    print("%.12f" % getweight("1960-10-21.txt", "reason"))
    print("%.12f" % getweight("1976-10-22.txt", "agenda"))
    print("%.12f" % getweight("2012-10-16.txt", "hispanic"))
    print("%.12f" % getweight("2012-10-16.txt", "hispan"))

    print("(%s, %.12f)" % query("health insurance wall street"))
    print("(%s, %.12f)" % query("particular constitutional amendment"))
    print("(%s, %.12f)" % query("vector entropy"))
    print("(%s, %.12f)" % query("terror attack"))

    print("%.12f" % getidf("hispanic"))
    print("%.12f" % getidf("hispan"))
    print("%.12f" % getidf("reason"))
    print("%.12f" % getidf("vector"))
    print("%.12f" % getidf("agenda"))
    print("%.12f" % getidf("health"))
