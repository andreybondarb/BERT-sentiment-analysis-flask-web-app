### BERT sentiment analysis flask web app. 

##### The 'rubert-base-cased' model was used for fine-tuning with russian comments scraped from mobile phone reviews site.

You can follow this link to try it - https://4eca02489f28.ngrok.io

##### Instructions to begin:

##### If you want to follow with docker use this part of instruction:

1. Download docker image: https://hub.docker.com/r/andreybondarb/bert-sentiment

2. Run in your console: 

`sudo docker run --name bert_model -p 4566:4566 --rm andreybondarb/bert-sentiment`

3. Follow the link http://0.0.0.0:4566/ in your browser and try the app.

##### If you want to start the app manually you will need to follow this instruction:

1. If you dont have PyTorch (700 MB~) and transformers libraries you will need to install them - This command in your CLI can be used:

`pip install -r '/requirements.txt'`

2. Download model weights from https://yadi.sk/d/EmtQcZKfGxPMdw and put 'bert_model_Russian_01.pt' (680 MB) file into the folder where 'main.py' file locates.

3. In your CLI run the 'main.py' file.

4. Follow the link http://0.0.0.0:4566/ in your browser and try the app.

##### This app was tested on python 3.7.6, 3.6, with libraries versions:

##### transformers==3.0.2

##### torch==1.6.0

##### flask==1.1.1

