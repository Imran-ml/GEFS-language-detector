# GEFS-language-detector
https://huggingface.co/ImranzamanML/GEFS-language-detector
### German, English, French and Spanish Language Detector

The GEFS-language-detector model outperformed by achieving an impressive F1 score close to 100%. This result significantly exceeds typical benchmarks and underscores the model's accuracy and reliability in identifying languages.
This is a fined tuned model by using the dataset of papluca [Language Identification](https://huggingface.co/datasets/papluca/language-identification#additional-information) and the base model [xlm-roberta-base](https://huggingface.co/xlm-roberta-base) .

## 100K downloads of the LLM model
https://huggingface.co/ImranzamanML/GEFS-language-detector
![image](https://github.com/Imran-ml/GEFS-language-detector/assets/149146155/7318e8c3-a672-4b71-9b17-d9be7f0cfbd4)


## Predicted output:

Model will return the language detection in the language codes like: 
```
  - de as German
  - en as English
  - fr as French
  - es as Spanish
```
  
## Supported languages
Currently this model support 4 languages but in future more languages will be added. 

Following languages supported by the model:
- German (de)
- English (en)
- French (fr)
- Spanish (es)

# Use a pipeline as a high-level helper

```python
from transformers import pipeline

text=["Mir gefällt die Art und Weise, Sprachen zu erkennen",
      "I like the way to detect languages",
      "Me gusta la forma de detectar idiomas",
      "J'aime la façon de détecter les langues"]
pipe = pipeline("text-classification", model="ImranzamanML/GEFS-language-detector")
lang_detect=pipe(text, top_k=1)
print("The detected language is", lang_detect)
```

# Load model directly

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("ImranzamanML/GEFS-language-detector")
model = AutoModelForSequenceClassification.from_pretrained("ImranzamanML/GEFS-language-detector")

```

## Model Training
  
    Epoch	  Training Loss	    Validation Loss
    1	      0.002600	        0.000148  
    2	      0.001000	        0.000015
    3	      0.000000	        0.000011
    4	      0.001800	        0.000009
    5	      0.002700	        0.000016
    6	      0.001600	        0.000012
    7	      0.001300	        0.000009
    8	      0.001200	        0.000008
    9	      0.000900	        0.000007
    10	      0.000900	        0.000007


## Testing Results
```
    Language   Precision   Recall	F1 	     Accuracy
    de	       0.9997	   0.9998	0.9998   0.9999
    en	       1.0000	   1.0000	1.0000	 1.0000
    fr	       0.9995	   0.9996	0.9996	 0.9996
    es	       0.9994	   0.9996	0.9995	 0.9996
```



## About Author

  **Name**: Muhammad Imran Zaman 
  
  **Company**: [Theum AG](https://theum.com/en/index.htm?t=) 
  
  **Role**: Lead Machine Learning Engineer 

  **Professional Links**:
  - Kaggle: [Profile](https://www.kaggle.com/muhammadimran112233)
  - LinkedIn: [Profile](linkedin.com/in/muhammad-imran-zaman)
  - Google Scholar: [Profile](https://scholar.google.com/citations?user=ulVFpy8AAAAJ&hl=en)
  - YouTube: [Channel](https://www.youtube.com/@consolioo)
  - GitHub: [Channel](https://github.com/Imran-ml)
