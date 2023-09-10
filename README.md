# NLP_IE_Pipelines
This is a general Natural Language Processing (NLP) system that comprised of a Named Entity Recognition (NER) module and a Relation Extraction (RE) module. The **Information Extraction Document (IE)** class is the main data structure used through out the training, evaluation, and prediction for both NER and RE. 

**Main frameworks**: PyTorch, Transformers (Hugging Face)

**Supported annotation tools**: Label-studio, BRAT, MAE

## Development pipeline overview
The  annotations are first converted to IE, then loaded by Dataset (PyTorch) to create training instances. 
![alt text](https://github.com/daviden1013/NLP_IE_Pipelines/blob/main/Development%20pipeline%20overview.png)

## Prediction pipeline overview
The raw text for information extraction is loaded and converted into IE. Then a fine-tuned NER model makes prediction on the IEs and outputs IEs with entities. An RE model then inpupts the IEs after NER and outputs IEs with entities and relations. 
![alt text](https://github.com/daviden1013/NLP_IE_Pipelines/blob/main/Prediction%20pipeline%20overview.png)
