## CS-Embed-SemEval2020
Code and specs for CS-Embed's entry for SemEval-2020 Task-9

# Code-Switch BiLSTM Model Summary
_________________________________________________________________
|Layer (type)|Output Shape|Param No.|   
|-----------------------|-------------------------|--------------|
|embedding (Embedding)|(None, 12, 100)|21592000|
|bidirectional (Bidirectional)|(None, 12, 256)|234496|
|bidirectional_1 (Bidirectional)|(None, 256)|394240|
|dropout (Dropout)|(None, 256)|0|
|dense (Dense)|(None, 100)|25700|
|dropout_1 (Dropout)|(None, 100)|0|
|dense_1 (Dense)|(None, 100)|10100|
|dropout_2 (Dropout)|(None, 100)|0|
|dense_2 (Dense)|(None, 3)|303|
_________________________________________________________________
Total params: 22,256,839
Trainable params: 22,256,839
Non-trainable params: 0
_________________________________________________________________
