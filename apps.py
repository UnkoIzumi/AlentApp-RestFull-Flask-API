import json
import pickle
import random

import nltk
import numpy
# import speech_recognition as sr
# from nltk.stem import LancasterStemmer
import matplotlib.pyplot as plt
from nlp_id.lemmatizer import Lemmatizer  # library pendeteksi kata yang berhimbuan awalan kata dan akhiran kata
from nlp_id.tokenizer import Tokenizer  # pemisahan kalimat yang akan diformat bentuk tag dengan acuan tiap spasi (per character)
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.models import Sequential  # library pelatihan menggunakan metode sequential
from tensorflow.python.keras.models import model_from_yaml  # library perubahan format data training ke yaml
from flask import Flask, request, jsonify
from flask_cors import CORS
import re

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

nltk.download('punkt')

# stemmer = LancasterStemmer()
lemmatizer = Lemmatizer()  # pemanggilan fungsi lemmatizer
tokenizer = Tokenizer()  # pemanggilan fungsi tokenizer

with open("intents2.json") as file:
    data = json.load(file)

try:
    with open("chatbot.pickle", "rb") as file:
        words, labels, training, output = pickle.load(file)

except:
    words = []
    labels = []
    docs_x = []
    docs_y = []
    del_words = ['dan', 'saya', 'aku', ',', 'tidak', 'dari', 'suka', 'ingin', 'tau', 'sangat', 'mata', 'lebih', 'mau',
                 'ketika', 'sifat', 'oh', 'seperti', 'itu', 'selesai', 'nggak', 'gampang', 'jatuh', 'gak', 'terlalu',
                 'yang',
                 'lakukan', 'dalam', 'adalah', 'sebagai', 'berbagai', 'takut']

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = tokenizer.tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in del_words]
    words = sorted(list(set(words)))

    labels = sorted(list(set(labels)))

    training = []
    output = []

    output_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [lemmatizer.lemmatize(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = output_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("chatbot.pickle", "wb") as file:
        pickle.dump((words, labels, training, output), file)

try:
    yaml_file = open('chatbotmodel.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    myChatModel = model_from_yaml(loaded_model_yaml)
    myChatModel.load_weights("chatbotmodel.h5")
    print("Loaded model from disk")

except:
    # pembuatan layer neural network
    myChatModel = Sequential()
    myChatModel.add(Dense(128, input_shape=[len(words)], activation='relu'))
    myChatModel.add(Dropout(0.5))
    myChatModel.add(Dense(64, activation='relu'))
    myChatModel.add(Dropout(0.5))
    myChatModel.add(Dense(len(labels), activation='softmax'))

    # optimize model
    myChatModel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    epochs = 100
    # train model
    history = myChatModel.fit(training, output, epochs=epochs, batch_size=8)

    # ploting data
    epochs_range = range(epochs)

    plt.figure(figsize=(15, 15))
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, history.history['accuracy'], color='g', label='Accuracy')
    plt.legend(loc='upper right')
    plt.title('Training Accuracy')

    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, history.history['loss'], color='r', label='Loss')
    plt.legend(loc='upper right')
    plt.title('Training Loss')
    plt.show()

    # serialize model to yaml and save it to disk
    model_yaml = myChatModel.to_yaml()
    with open("chatbotmodel.yaml", "w") as y_file:
        y_file.write(model_yaml)

    # serialize weights to HDF5
    myChatModel.save_weights("chatbotmodel.h5")
    print("Saved model from disk")


def bag_of_words(s, words):
    dele_words = ['dan', 'saya', 'aku', ',', 'tidak', 'dari', 'suka', 'ingin', 'tau', 'sangat', 'mata', 'lebih', 'mau',
                  "buat"
                  'ketika', 'sifat', 'oh', 'seperti', 'itu', 'selesai', 'nggak', 'gampang', 'jatuh', 'gak', 'terlalu',
                  "laku" "pada"
                  'yang', 'lakukan', 'dalam', 'adalah', 'sebagai', 'berbagai', 'takut', "di", "yaitu", "paling",
                  "banyak", "nya", "sebut"]
    list_check = 0

    bags = [0 for _ in range(len(words))]

    # pemisahan kalimat mejadi per kata dari input suara
    s_words = tokenizer.tokenize(s)
    s_words = [lemmatizer.lemmatize(word.lower()) for word in s_words if word not in dele_words]
    print(s_words)

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                list_check += 1
                bags[i] = 1

    list_error = (((len(s_words)) - list_check) / (len(s_words))) * 100
    list_acc = (list_check / (len(s_words))) * 100
    print("Data Sukses : {:0.2f}%".format(list_acc))
    print("Data Loss : {:0.2f}%".format(list_error))
    return numpy.array(bags), list_check


# input = data dari speech recognition yang akan diuabh menjadi text
# proses = data text akan diolah menggunakan nlp
# output = pencocokan presentase job skill

def chat_with_bot(input_text):
    currentText, wordz = bag_of_words(input_text, words)
    currentTextArray = [currentText]
    numpyCurrentText = numpy.array(currentTextArray)

    if numpy.all((numpyCurrentText == 0)):
        return "I didn't get that, try again"

    result = myChatModel.predict(numpyCurrentText[0:1])
    result_index = result[-1]
    dataindex = len(result_index)

    presx = [0 for _ in range(dataindex)]
    respx = [0 for _ in range(dataindex)]

    for datai in range(dataindex):
        tag = labels[datai]
        # tagdata = result[0][datai]
        # if tagdata > 0.2:
        for tg in data["intents"]:
            if tg['tag'] == tag:
                patterns = tg['patterns']
                responses = tg['responses']
                numpola = len(patterns)

                presx[datai] = (wordz / numpola) * 100
                respx[datai] = random.choice(responses)

    return respx, presx, dataindex

    # else:
    # print("I didn't get that, try again")


@app.route("/", methods=['GET', 'POST'])
def post():
    selectInput = request.form['select']
    chatInput = request.form['chatInput']

    if selectInput == "0":
        resp, pres, maks = chat_with_bot(chatInput)
        list_job = []
        highListJob = []
        for ada in range(maks):
            if pres[ada] >= 3:
                if resp[ada] == "Relating":
                    continue
                elif resp[ada] == "Thinking":
                    continue
                elif resp[ada] == "Impacting":
                    continue
                elif resp[ada] == "Striving":
                    continue
                else:
                    list_job.append("Skill: {}, {:0.2f}%".format(resp[ada], pres[ada]))

        sorting = sorted(list_job, key=lambda s: float(re.search(r'(\d+)\.', s).groups()[0]))

        for i in range(1, 6):
            highListJob.append(sorting[-i])

        datax = '\n'.join(map(str, highListJob))
        return jsonify(chatBotReply="Rekomendasi :\n{}".format(datax))

    elif selectInput == "1":
        resp, pres, maks = chat_with_bot(chatInput)
        list_job = []
        highListJob = []

        for ada in range(maks):
            if pres[ada] >= 3:
                if resp[ada] == "Relating":
                    continue
                elif resp[ada] == "Thinking":
                    continue
                elif resp[ada] == "Impacting":
                    continue
                elif resp[ada] == "Striving":
                    continue
                else:
                    list_job.append("Skill: {}, {:0.2f}%".format(resp[ada], pres[ada]))

        sorting = sorted(list_job, key=lambda s: float(re.search(r'(\d+)\.', s).groups()[0]))

        for i in range(1, 6):
            highListJob.append(sorting[-i])

        datac = '\n'.join(map(str, highListJob))
        return jsonify(chatBotReply="Tidak Rekomendasi :\n{}".format(datac))

    else:
        resp, pres, maks = chat_with_bot(chatInput)
        mapping = []

        for ada in range(maks):
            if resp[ada] == "Relating":
                mapping.append("Dominan : {}(sosialisasi), {:0.2f}%".format(resp[ada], pres[ada]))
            if resp[ada] == "Thinking":
                mapping.append("Dominan : {}(Pemikir), {:0.2f}%".format(resp[ada], pres[ada]))
            if resp[ada] == "Impacting":
                mapping.append("Dominan : {}(Pemimpin), {:0.2f}%".format(resp[ada], pres[ada]))
            if resp[ada] == "Striving":
                mapping.append("Dominan : {}(Pekerja Keras), {:0.2f}%".format(resp[ada], pres[ada]))

        dominanSort = sorted(mapping, key=lambda s: float(re.search(r'(\d+)\.', s).groups()[0]))
        return jsonify(chatBotReply=dominanSort)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=3000, debug=True)
