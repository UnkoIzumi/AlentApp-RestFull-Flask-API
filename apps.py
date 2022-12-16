from flask import Flask, request, jsonify
from flask_cors import CORS
from main import chat_with_bot
import re

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})


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

        data = '\n'.join(map(str, highListJob))
        return jsonify(chatBotReply="Rekomendasi :\n{}".format(data))

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

        data = '\n'.join(map(str, highListJob))
        return jsonify(chatBotReply="Tidak Rekomendasi :\n{}".format(data))

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
    app.run(host='127.0.0.1', port=8080, debug=True)
