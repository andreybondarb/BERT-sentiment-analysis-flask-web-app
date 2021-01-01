#!/usr/bin/env python
from pprint import pprint as pp
from flask import Flask, flash, redirect, render_template, request, url_for
from codecs import open
import time
from bert_classifier import return_ids_masks, predict
import pickle
import numpy as np
import transformers
from transformers import BertTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, random_split
import torch



#Flask

app = Flask(__name__)


@app.route("/", methods=["POST", "GET"])
def index_page(text = ''):

    return render_template('start.html', text=text)

################

pics = ['0-1.svg',
'1-4.svg',
'4-8.svg',
'8-13.svg',
'13-17.svg',
'17-20.svg',
'20-25.svg',
'25-35.svg',
'35-55.svg',
'55-76.svg',
'76-80.svg',
'80-84.svg',
'84-88.svg',
'88-92.svg',
'92-96.svg',
'96-98.svg',
'98-100.svg']

@app.route("/result" , methods=['GET', 'POST'])
def result():
    if request.method == "POST":
        text = request.form["text"]
        logfile = open("ydf_demo_logs.txt", "a", "utf-8")
        print(text)

        test_ids, test_masks = return_ids_masks([text])
        batch_size = 1
        test_data = TensorDataset(test_ids, test_masks)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)


        prediction_message = round(predict(test_dataloader, is_probs=True)[0][1]*100, 2)
        prediction_message = float(prediction_message)
        print (prediction_message)
        if 0 <= prediction_message < 1:
            mood = pics[0]
        elif 1 <= prediction_message < 4:
            mood = pics[1]
        elif 4 <= prediction_message < 8:
            mood = pics[2]
        elif 8 <= prediction_message < 13:
            mood = pics[3]
        elif 13 <= prediction_message < 17:
            mood = pics[4]
        elif 17 <= prediction_message < 20:
            mood = pics[5]
        elif 20 <= prediction_message < 25:
            mood = pics[6]
        elif 25 <= prediction_message < 35:
            mood = pics[7]
        elif 35 <= prediction_message < 55:
            mood = pics[8]
        elif 55 <= prediction_message < 76:
            mood = pics[9]
        elif 76 <= prediction_message < 80:
            mood = pics[10]
        elif 80 <= prediction_message < 84:
            mood = pics[11]
        elif 84 <= prediction_message < 88:
            mood = pics[12]
        elif 88 <= prediction_message < 92:
            mood = pics[13]
        elif 92 <= prediction_message < 96:
            mood = pics[14]
        elif 96 <= prediction_message < 98:
            mood = pics[15]
        elif 98 <= prediction_message <= 100:
            mood = pics[16]

        prediction_message = str(prediction_message)

        logfile.close()


    return render_template('result.html', data=prediction_message, mood=mood)



if __name__ == "__main__":
    app.run(host='0.0.0.0' , port=4566, debug=False)