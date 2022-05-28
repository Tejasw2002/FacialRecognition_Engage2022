# importing the required libraries
from flask import Flask, render_template, request, session
from deepface import DeepFace
from openpyxl import load_workbook
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity as cs
from scipy import spatial
import os
import shutil

# Directory where the images of the customers are stored
# The files are saved with the id of the customers
CUSTOMER_PHOTOS_DIR = r'C:\Users\tejaw\Desktop\Eng\Customers'

# File which stores the personal records of the customers
CUSTOMERS_RECORDS = r'C:\Users\tejaw\Desktop\Eng\Customers.txt'

# An excel file to store the orders of the customers
filename = r'C:\Users\tejaw\Desktop\Eng\Book2.xlsx'


wb = load_workbook(filename)

app = Flask(__name__)

# Our temporary jpg file, uploaded by the user will be stored here
app.config['UPLOAD_FOLDER'] = r'C:\Users\\tejaw\Desktop\Eng\\'
app.secret_key = "super secret key"

# This file contains the name of the food items and the tags associated with them
df1 = pd.read_excel('Book1.xlsx')

# Creating a vector for each food item based on the tags associated with them
cv = CountVectorizer()
vectors = cv.fit_transform(df1['Tags']).toarray()

# This function returns the customer_id of a new customer.
# The id in the last record of the file is extracted and incremented by 1, to get the id of the new customer


def customer_id():
    COUNT = 0
    file = open(CUSTOMERS_RECORDS, "r")
    lines = file.readlines()
    if len(lines) != 0:
        COUNT = int(lines[-1].split(",")[0])
    file.close()
    return COUNT+1


def findimage(image):
    df = DeepFace.find(image, CUSTOMER_PHOTOS_DIR)
    # A dataframe of the images matching the given image is returned, with three fields,
    # id, path of the image and the distance metric
    # We use the path of the image to get the id of the customer

    cust_id = int(df['identity'][0].split('/')[1].split(".")
                  [0]) if df.shape[0] != 0 else 0
    to_remove = r"C:\Users\tejaw\Desktop\Eng\Customers/representations_vgg_face.pkl"
    if os.path.exists(to_remove):
        os.remove(to_remove)
    found = 1
    if cust_id == 0:
        # Cust_id = 0 means the customer is visiting the store for the first time
        # as they do not have nay match in the database of images
        found = 0

        # Assign an ID to the new customer usign the customer_id() function
        count = str(customer_id())

        # Storing the image of the new customer in the database
        filename = count+".jpg"
        shutil.copy(image, CUSTOMER_PHOTOS_DIR)
        os.rename(CUSTOMER_PHOTOS_DIR+"\\"+'temp.jpg',
                  CUSTOMER_PHOTOS_DIR+"\\"+filename)
        file = open(CUSTOMERS_RECORDS, "a")

        # Storing the personal details of the customer
        demography = DeepFace.analyze(image)
        sex = demography['gender'][0]
        age = str(demography['age'])
        ethnicity = demography['dominant_race']
        entry = count + ", " + sex + ", " + age + ", " + ethnicity + "\n"
        file.write(entry)
        file.close()
        cust_id = count
    return [found, cust_id]

# This function takes a list of tags and vectorizes it.


def create_vector(list_of_tags):
    x = cv.vocabulary_
    x = {k: v for k, v in sorted(x.items(), key=lambda item: item[1])}
    vector = [0]*len(x)
    for tag in list_of_tags:
        vector[x[tag.lower()]] += 1
    return np.array(vector)


@app.route("/")
def hello_world():
    return render_template('index.html')


@app.route("/app.py")
def upload_page():
    return render_template('upload.html')


@app.route("/view-details", methods=['POST', 'GET'])
def do():
    if request.method == 'POST':
        img = request.form.get('img')
        file = request.files['image']
        if img == 'EMPTY':
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], "temp.jpg"))
        else:
            import base64
            img_data = img[img.index(',')+1:]
            img_data = bytearray(img_data.encode())
            print(img_data)
            with open("temp.jpg", "wb") as fh:
                fh.write(base64.decodebytes(img_data))
        
        # The uploaded images are stored in the folder temporarily as temp.jpg
        # The api matches this file against all the files in the Customers folder

        found, cust_id = findimage("temp.jpg")
        message = recommend(found, cust_id, "temp.jpg")
        session['cust_id'] = cust_id
        session['mood'] = message['mood'].capitalize()
        return render_template('user.html', message=message, data=list(df1['Food-item']))


@app.route("/thankyou.html", methods=['POST', 'GET'])
def thank():
    if request.method == 'POST':
        cust_id = session.get('cust_id')
        mood = session.get('mood')
        order_list = request.form.get("order")
        updateExcel(cust_id, mood, order_list)
        return render_template('thanks.html')


def recommend(found, cust_id, image):
    message = {}
    message['greeting'] = "Welcome Customer  " + str(cust_id)
    emotion = DeepFace.analyze(image, actions=['emotion'])['dominant_emotion']
    temp = ""
    if emotion == 'happy':
        temp = "Glad to see you are having a great day! 🤩"
    elif emotion == 'sad':
        temp = "Been a rough day? Let's get you some comfort food 😇"
    else:
        emotion = 'neutral'
        temp = "Let us make your day even better. 😁"
    message["emotion"] = temp
    message['mood'] = emotion
    message["found"] = "We currently have no suggestions for you!!!" if found == 0 else "Based on your previous purchases, we have some suggestions for you"
    df2 = pd.read_excel('Book2.xlsx')
    x = list(df2.loc[(df2['Customer_ID'] == cust_id) & (
        df2['Emotion'] == emotion.capitalize())]['Order'])
    if len(x) == 0:
        x = list(df2.loc[(df2['Customer_ID'] == cust_id)]['Order'])
    y = []
    for item in x:
        for meal in item.split(","):
            y.append(meal.strip())
    tags = []
    for food in y:
        temp = df1.loc[df1['Food-item'] == food]['Tags']
        for item in temp:
            for i in item.split(","):
                tags.append(i.strip())
    vec = create_vector(tags)
    listed = []
    for v in vectors:
        result = 1 - spatial.distance.cosine(v, vec)
        listed.append(result)
    items = sorted(list(enumerate(listed)),
                   reverse=True, key=lambda x: x[1])[0:5]
    message["recommendations"] = []
    for i in items:
        message["recommendations"].append(df1.iloc[i[0]]['Food-item'])
    return message


def updateExcel(cust_id, mood, order_list):
    wb.worksheets[0].append([cust_id, mood, order_list])
    wb.save(filename)


if __name__ == "__main__":
    app.run(debug=True)
