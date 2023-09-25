# Model Compression Course
Assignments for the Model Compression Course (ITMO University, AI Talent Hub)

Model: ```bert-base-cased```  
Dataset: ```mteb/tweet_sentiment_extraction``` (30K samples)

| Experiment | Num params | Model size (MB) | Inference time (s) | Macro F1 |
|------------|------------|-----------------|--------------------|----------|
| original_model   | 1.08e+08   | 413.188         | 75.124              | 0.783   | 
| dynamic_quantization   | 2.27e+07  | 86.609         | 50.985              | 0.779   | 