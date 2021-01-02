BERT sentiment analysis flask web app. The model was trained on Russian text from mobile phone reviewers sites.

Instructions to begin:

1. For the classificator to work you need to install additional libraries - PyTorch (700Мб~) and transformers. This command in your CLI can be used:

pip install -r '/requirements.txt'

2. Download model weights from https://yadi.sk/d/EmtQcZKfGxPMdw and put 'bert_model_Russian_01.pt' (680 MB) file into the folder where 'main.py' file locates.

3. In your CLI start the 'main.py' file.

4. Follow the link http://0.0.0.0:4566/ in your browser and try the app.

This app was tested on python 3.7.6, 3.6, with libraries versions:

transformers==3.0.2

torch==1.6.0

flask==1.1.1

