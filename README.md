# Model Compression Course
Assignments for the Model Compression Course (ITMO University, AI Talent Hub)

Model: ```bert-base-cased```  
Dataset: ```mteb/tweet_sentiment_extraction``` (30K samples)

| Experiment | Num params | Model size (MB) | Inference time (s) | Macro F1 |
|------------|------------|-----------------|--------------------|----------|
|exp_original_model|1.083e+08|413.188|75.256|0.784|
|exp_dynamic_quantization|2.270e+07|86.609|54.674|0.779|
|exp_unstructured_pruning_random|1.083e+08|413.188|81.144|0.429|
|exp_unstructured_pruning_l1|1.083e+08|413.188|80.742|0.665|
|exp_sparse_training|1.083e+08|413.188|83.362|0.714|



## Weight clustering 
Text classification on IMDB dataset
| Model | Num params | Model size (MB) | Accuracy |
|------------|------------|-----------------|----------|
|baseline | 160033 | 0.625  | 0.904 |
|baseline_clustered | 320097 | 1.83  | 0.834 |
|baseline_clustered_finetuned | 320097 | 1.83  | 0.902 |




## Knowledge distillation 
Model (teacher): ```bert-base-uncased```  
Model (student 1): ```bert-tiny``` 
Model (student 2): ```distilbert-base-uncased``` 
Dataset: ```mteb/tweet_sentiment_extraction``` (30K samples)
| Model | Num params | Model size (MB) |Time for 100 samples (s) | Accuracy | Raw aсcuracy |
|------------|------------|-----------------|-----------------|----------|----------|
|bert-base-uncased (1 exp) | 109484547 | 439 | 10.37 | 0.831 | - |
|bert-tiny | 4386307 | 19  | 1.01 | 0.521 | 0.282 |
|bert-base-uncased (2 exp) | 109484547 | 439 | 11.87 | 0.831 | - |
|distilbert-base-uncased | 66955779 | 269 | 5.34 | 0.789 | 0.282 |
