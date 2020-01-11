from selenium import webdriver
import time, sys
import re
import csv
import string
from textblob import TextBlob
import nltk
import numpy as np
from nltk.corpus import stopwords
from pandas import DataFrame
from sklearn import naive_bayes
from flask import Flask, redirect, url_for, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
app = Flask(__name__)


global num_of_names
def preprocess_reviews(reviews):
    
    # reviews = [REPLACE_NO_SPACE.sub(NO_SPACE, line.lower()) for line in reviews]
    # reviews = [REPLACE_WITH_SPACE.sub(SPACE, line) for line in reviews]
    
    r = []
    for comment in reviews:
        r.append(re.sub('[^A-Za-z1-9 ]','',comment).lower())
    return r

def Navie_Bias():
    reviews_train = []
    for line in open('train.txt', 'r', encoding="utf-8"):
        
        reviews_train.append(line.strip())


    data = []
    with open('mycsv.csv' , 'r', encoding = "utf-8") as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader: 
            data.append(''.join(row)) 

    print("length of data is ",len(data))

    reviews_train_clean = preprocess_reviews(reviews_train)


    data_clean = preprocess_reviews(data)

    target = [1 if i < 12500 else 0 for i in range(25000)]

    vectorizer = CountVectorizer(binary=True, ngram_range = (1, 2))
    reviews_vector = vectorizer.fit_transform(reviews_train_clean)
    data_vector = vectorizer.transform(data_clean)


    X_train, X_test, y_train, y_test = train_test_split(reviews_vector, target, train_size = 0.80)

    clf = naive_bayes.MultinomialNB()
    clf.fit(X_train, y_train)

    acc = accuracy_score(y_test, clf.predict(X_test))
    print("accuracy ",acc)


    # print(data_vector)
    res = clf.predict(data_vector)
    # print(res)
    ones = 0
    zeros = 0
    for i in range(len(data)):
        if res[i] == 0:
            zeros = zeros+1
        else :
            ones = ones+1
        print(data[i], res[i])
    print("========")
    print("positive comments", ones)
    print("negitive", zeros)

    # return ones, zeros
    if ones > zeros:
        return ("Positive")
    else:
        return ("Negative")

def Logistic_Regression():
    reviews_train = []
    for line in open('train.txt', 'r', encoding="utf-8"):
        
        reviews_train.append(line.strip())
        
    data = []
    with open('mycsv.csv' , 'r', encoding = "utf-8") as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader: 
            data.append(''.join(row)) 

    # data = data[2:len(data)]
    print("length of data is ",len(data))


# REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")
# REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
# NO_SPACE = ""
# SPACE = " "

    reviews_train_clean = preprocess_reviews(reviews_train)


    data_clean = preprocess_reviews(data)



    print("clean data",data_clean)
    cv = CountVectorizer(binary=True, ngram_range = (1, 2))
    cv.fit(reviews_train_clean)
    X = cv.transform(reviews_train_clean)
    data_res = cv.transform(data_clean)



    target = [1 if i < 12500 else 0 for i in range(25000)]

    X_train, X_val, y_train, y_val = train_test_split(
        X, target, train_size = 0.80
    )



    final_model = LogisticRegression()
    final_model.fit(X_train, y_train)
    print("accuracy is", accuracy_score(y_val, final_model.predict(X_val)))



    res = final_model.predict(data_res)

    print("==========")
    ones = 0
    zeros = 0
    for i in range(len(data)):
        if res[i] == 0:
            zeros = zeros+1
        else :
            ones = ones+1
        print(data[i], res[i])
    print("========")
    print("positive comments", ones)
    print("negitive", zeros)
    # return ones, zeros
    if ones > zeros:
        return ("Positive")
    else:
        return ("Negative")


        
    # movie_review_array = np.array(["this is not bad project"])
    # vector = cv.transform(movie_review_array)
    # print(final_model.predict(vector))


def Text_Blob():
    filename = "mycsv.csv"

    li = []
    with open(filename, encoding="utf8") as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            # print(type())
            li.append("".join(row))

        # print(li)

    final = []
    for comment in li:
        final.append(re.sub('[^A-Za-z1-9 ]', '', comment))

    # print(final)
    words = []
    for i in final:
        temp = i.split(" ")
        # print(temp)
        for k in temp:
            words.append(k.lower())
    # print(words)


    s = set(stopwords.words('english'))
    # print("stopwords======")
    # print(s)
    for i in words:
        if i in s:
            words.remove(i)
    # print("=============")
    # print(set(words))
    final_sum = 0
    comment_Data = []
    neg = 0
    neg_count = 0
    pos_count = 0
    neu_count = 0
    pos = 0
    neu = 0
    
    result = []
    count = []
    for comment in final:
        comment_Data.append(TextBlob(comment).sentiment)
        print(comment, TextBlob(comment).sentiment)
        print("--------------------------------")
        p = TextBlob(comment).sentiment
        data = p.polarity
        final_sum += data
        if data > 0:
            pos += data
            pos_count += 1
        elif data < 0:
            neg += data
            neg_count += 1
        else:   
            neu += data
            neu_count += 1

    result.append(pos)
    result.append(neg)
    result.append(neu)
    count.append(pos_count)
    count.append(neg_count)
    count.append(neu_count)

    # with open('dict.csv', 'w', newline='',encoding="utf-8") as f:
    #     thewriter = csv.writer(f)
    #     for i in count:
    #         thewriter.writerow([i])

    result_dict = {"Positive":pos_count, "Negative":neg_count, "Neutral":neu_count}

    print(result, "Result_Data")

    print(count, "Count")


    with open('dict.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in result_dict.items():
           writer.writerow([key, value])

    if pos_count > neg_count:
        print("\N{grinning face}, Positive")
        return ("\N{grinning face}, Positive")
    elif neg_count < pos_count:
        print("Negative")
        return ("Negative")
    else:
        print("Neutral")
        return ("Neutral")
    print(max(count), "Its maxi")
    # print(final_sum)
    # # print(comment_Data)

    print("-----------------------------------------------------------------")
    print(result_dict)


def process(url):
    try:

        start = time.time()
        def isEnglish(s):
            try:
                s.encode('ascii')
            except UnicodeEncodeError:
                return False
            else:
                return True

        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument("--mute-audio")

        driver = webdriver.Chrome(chrome_options=chrome_options)
        # yt_link = sys.argv[1]
        yt_link = url
        # yt_link = input("Link to Youtube video: ")
        print("----------------------------------------------------------")
        # if yt_link == "":
        #     print("Enter URL")
        # else:
        driver.get(yt_link)
        # driver.maximize_window()
        driver.set_window_position(-10000, 0)

        time.sleep(3)

        title = driver.find_element_by_xpath('//*[@id="container"]/h1/yt-formatted-string').text
        # print("Video Title: " + title)
        # print("-----------------------------------------------------------")

        comment_section = driver.find_element_by_xpath('//*[@id="comments"]')
        driver.execute_script("arguments[0].scrollIntoView();", comment_section)
        time.sleep(3)

        last_height = driver.execute_script("return document.documentElement.scrollHeight")
        while True:
            # Scroll down to bottom
            driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")

            # Wait to load page
            time.sleep(4)

            # Calculate new scroll height and compare with last scroll height
            new_height = driver.execute_script("return document.documentElement.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

        driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")

        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  
                                   u"\U0001F300-\U0001F5FF"
                                   u"\U0001F680-\U0001F6FF"
                                   u"\U0001F1E0-\U0001F1FF"
                                   "]+", flags=re.UNICODE)

        name_elems=driver.find_elements_by_xpath('//*[@id="author-text"]')
        comment_elems = driver.find_elements_by_xpath('//*[@id="content-text"]')
        # print("****************************")
        # print(comment_elems)
        # print("****************************")
        num_of_names = len(name_elems) 
        print("length ", num_of_names)           
        if num_of_names == 0:
            return -1 

        comments_Data = []
        for i in range(num_of_names):
            username = name_elems[i].text    # .replace(",", "|")
            # username = emoji_pattern.sub(r'', username)
            # username = str(username).replace("\n", "---")
            comment = comment_elems[i].text
            # comment = comment.replace(",", "|")
            # comment = emoji_pattern.sub(r'', comment)
            # comment = str(comment).replace("\n", "---")
            if isEnglish(comment) == False:
               comment = "NOT ENGLISH"
               continue
            else:
                comments_Data.append(comment)

        with open('mycsv.csv', 'w', newline='',encoding="utf-8") as f:
            thewriter = csv.writer(f)
            for i in comments_Data:
                thewriter.writerow([i])


            
            #if isEnglish(comment) == False:
             #   comment = "NOT ENGLISH"
        print(comments_Data)  
        print(len(comments_Data),"--------------------")      
            # print(username + ": " + comment) # comment.translate({ord(i):None for i in '' if i not in string.printable})
            # print("-------------------------------------------------------------------------------------------------------------------")
        end = time.time()
        print(end - start, "TIMEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
        driver.close()

        print("with textblob")
        tb_result = Text_Blob()
        print("LogisticRegression============")
        # lr_result = Logistic_Regression()
        # print("Navie_Bias---------------------")
        # nb_result = Navie_Bias()
        # return tb_result, lr_result
        return "Hey the title is -->", title, "The sentiment is-->", tb_result
        # =====================================================================================
    except:
        if url == "":
            # print("enter URL////")
            return redirect(url_for('sample'))


@app.route('/', methods=['GET'])
def sample():
    return render_template('sample.html')


@app.route('/flask', methods=['POST'])
def hello_world():
    if request.method == 'POST':
        url = request.form['link']
        # tb_res, lr_ones, lr_zeros, nb_ones, nb_zeros= process(url)
        # tb_res, lr_res = process(url)
        # tb_res= process(url)
        # return 'result is %s '% dict1
        # print, "--------------------------")
        # 
        # res_dic = {"TextBlob Res":tb_res, "Logistic_Regression Res": lr_res}
        # res_dic = {"TextBlob Res":tb_res}
        # return render_template('result.html', name = res_dic)
        if 'youtu.be' in url:
            tb_res = process(url)
            res_dic = {"TextBlob Result" : tb_res}
            return render_template('result.html', name = res_dic)

        if url == "" or 'watch?v' not in url:
            print("1")
            return render_template('sample.html')
            print("2")
        else:
            tb_res = process(url)
            print("tb_res", tb_res)
            if tb_res == -1:
                res_dic = {"Result is" : "No comments"}
                # return "No Comments"
                return render_template('result.html', name = res_dic)
            else:     
                res_dic = {"TextBlob Result" : tb_res}
                return render_template('result.html', name = res_dic)




if __name__ == '__main__':
    app.run(use_reloader = True, debug = True)
