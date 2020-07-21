# CS-Embed-SemEval2020
Code and specs for CS-Embed's entry for SemEval-2020 Task-9

# Code-Switch BiLSTM Model Summary
_________________________________________________________________
|Layer (type)                   | Output Shape        | Param No.   
|===============================|=====================|===========
|embedding (Embedding)          |(None, 12, 100)      |21592000  
_________________________________________________________________
|bidirectional (Bidirectional)  | (None, 12, 256)     |234496    
_________________________________________________________________
|bidirectional_1 (Bidirectional)| (None, 256)         |394240    
_________________________________________________________________
|dropout (Dropout)              |(None, 256)          |0         
_________________________________________________________________
|dense (Dense)                  |(None, 100)          |25700     
_________________________________________________________________
|dropout_1 (Dropout)            |(None, 100)          |0         
_________________________________________________________________
|dense_1 (Dense)                |(None, 100)          |10100     
_________________________________________________________________
|dropout_2 (Dropout)            |(None, 100)          |0         
_________________________________________________________________
|dense_2 (Dense)                |(None, 3)            |303       
|===============================|=====================|===========
Total params: 22,256,839
Trainable params: 22,256,839
Non-trainable params: 0
_________________________________________________________________
