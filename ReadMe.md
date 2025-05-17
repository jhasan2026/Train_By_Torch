1. **Fundamental of Pytorch**
   1. Tensor in Pytorch [==>](1_Fundemental_of_Pytorch/1_Tensor_in_Pytorch/1_Tensor_in_Pytorch.ipynb)
   2. Autograd [==>](1_Fundemental_of_Pytorch/2_Autograd/Autograd.ipynb)
   
2. **Data Gathering** [==>](2_Data_Gathering/1_DataSet_and_Dataloader.ipynb)

3. **Training Pipeline**
   1. Perceptron
      1. Without NN Module [==>](3_Trainning_Pipelines/1_Perceptron/1_Without_Using_NN_Module/1_Without_Using_NN_Module.ipynb)
      2. Using NN Module [==>](3_Trainning_Pipelines/1_Perceptron/2_Using_NN_Module/1_Using_NN_Module.ipynb)
   2. Multi Layer Perceptron 
      1. MLP Using NN Module [==>](3_Trainning_Pipelines/2_Multi_layer_Perceptron/1_Using_NN_Module/1_MLP_Using_NN_Module.ipynb)
      2. Using Sequential Module [==>](3_Trainning_Pipelines/2_Multi_layer_Perceptron/2_Sequenctial_Module_using/1_Sequenctial_Module_using.ipynb)
      3. Using Dataloader [==>](3_Trainning_Pipelines/2_Multi_layer_Perceptron/3_Using_Dataload/1_MLP_using_dataloader.ipynb)

4. **ANN**
   1. ANN Fasion MNIST [==>](4_ANN/1_ANN_Fasion_MNIST/1_ANN_Fasion_MNIST.ipynb)
   
5. **Computer Vision**
   1. CNN Architecture [==>](5_Computer_Vision/1_CNN_Architecture/1_Fasion_MNIST/1_CNN_Architecture.ipynb)
 
6. **Natural Language Processing**
   1. RNN
      1. Question Answer System   [==>](6_Natural_Language_Processing/1_RNN/1_QA_System/1_QA_System.ipynb)
   2. LSTM
      1. Next Word Prediction
         1. Without Custom Encoding   [==>](6_Natural_Language_Processing/2_LSTM/1_Next_Word_Prediction/1_Next_Word_Prediction.ipynb)
         2. Using Custom Encoding   [==>](6_Natural_Language_Processing/2_LSTM/1_Next_Word_Prediction/2_Next_word_prediction_Using_Custom_Encoding.ipynb)
   3. Bidirectional LSTM
      1. Next Word Prediction   [==>](6_Natural_Language_Processing/3_Bidirectional_LSTM/1_Next_Word_Prediction_Bidirectional_LSTM.ipynb)
   4. Build LLM
      1. Preprocessing
         1. Using Custom Encoding   [==>](6_Natural_Language_Processing/4_Build_LLM/1_Preprocessing/1_LLM_From_Scratch_Using_Custom_Encoding.ipynb)
         2. Byte-Pair Encoding Using   [==>](6_Natural_Language_Processing/4_Build_LLM/1_Preprocessing/2_LLM_From_Scratch_BPE.ipynb)
      2. Attention Mechanism
         1. Luong Attention
            1. Attention Wight Calculation   [==>](6_Natural_Language_Processing/4_Build_LLM/2_Attention_Machanism/1_Luong_Attention/1_Attention_Weight_Calculate.ipynb)
            2. Luong Attention Implementation   [==>](6_Natural_Language_Processing/4_Build_LLM/2_Attention_Machanism/1_Luong_Attention/2_Luong_Attenstion.ipynb)
         2. Self Attention
            1. Self Attention Calculation   [==>](6_Natural_Language_Processing/4_Build_LLM/2_Attention_Machanism/2_Self_Attention/1_Self_Attention_Calculation.ipynb)
            2. Self Attention Implementation   [==>](6_Natural_Language_Processing/4_Build_LLM/2_Attention_Machanism/2_Self_Attention/2_Self_Attention_Using_Linear.ipynb)
         3. Masked Attention
            1. Masked Attention Implementation   [==>](6_Natural_Language_Processing/4_Build_LLM/2_Attention_Machanism/3_Masked_Self_Attetion/1_Masked_Self_Attetion.ipynb)
            2. Masked Attention Implementation for batchs   [==>](6_Natural_Language_Processing/4_Build_LLM/2_Attention_Machanism/3_Masked_Self_Attetion/2_Masked_Self_Attention_Batch.ipynb)
         4. Multi-head Masked Attention
            5. Wrapper Class of Multi-head Masked Attention   [==>](6_Natural_Language_Processing/4_Build_LLM/2_Attention_Machanism/4_Multi-head_Masked_Attention/1_Wrapper_Multi-head_Masked_Attention.ipynb)
            5. Optimized Multi-head Masked Attention   [==>](6_Natural_Language_Processing/4_Build_LLM/2_Attention_Machanism/4_Multi-head_Masked_Attention/2_Multi_Head_Masked_Attention.ipynb)
      3. LLM Architecture
         1. GPT Model Architecture   [==>](6_Natural_Language_Processing/4_Build_LLM/3_LLM_Architecture/1_GPT_Model_Architecture/1_GPT_Model_Architecture.ipynb)
         2. Layer Normalization   [==>](6_Natural_Language_Processing/4_Build_LLM/3_LLM_Architecture/2_Layer_Normalization/1_Layer_Normalization.ipynb)
         3. Feed Forward With GELU   [==>](6_Natural_Language_Processing/4_Build_LLM/3_LLM_Architecture/3_Feed_Forward_With_GELU/1_Feed_Forward_With_GELU.ipynb)
         4. Shortcut Connection   [==>](6_Natural_Language_Processing/4_Build_LLM/3_LLM_Architecture/4_Shortcut_Connection/1_Shortcut_Connection.ipynb)
         5. Transformer Block   [==>](6_Natural_Language_Processing/4_Build_LLM/3_LLM_Architecture/5_Transformer_Block/1_Transformer_Block.ipynb)
         6. Complete GPT Model
            1. Small GPT-2 Model (124 M)   [==>](6_Natural_Language_Processing/4_Build_LLM/3_LLM_Architecture/6_Complete_GPT_Model/1_GPT_2_small_Model.ipynb)
            2. Medium GPT-2 Model (350 M)   [==>](6_Natural_Language_Processing/4_Build_LLM/3_LLM_Architecture/6_Complete_GPT_Model/2_GPT-2-Medium_Model.ipynb)
            3. Large GPT-2 Model (750 M)   [==>](6_Natural_Language_Processing/4_Build_LLM/3_LLM_Architecture/6_Complete_GPT_Model/3_GPT-2-Large_Model.ipynb)
            4. XL GPT-2 Model (1.63 B)   [==>](6_Natural_Language_Processing/4_Build_LLM/3_LLM_Architecture/6_Complete_GPT_Model/4_GPT-2-XL_Model.ipynb)
         7. Text Generation     [==>](6_Natural_Language_Processing/4_Build_LLM/3_LLM_Architecture/7_Text_Genearation/1_Text_Generation.ipynb)    
      4. Pretraining on Unlabeled Data
         1. Text-Token Convert     [==>](6_Natural_Language_Processing/4_Build_LLM/4_Pretraining_on_unlabeled_data/1_Text_Token_Convert/1_Text_Token_Convert.ipynb)
         2. Loss Function Generate     [==>](6_Natural_Language_Processing/4_Build_LLM/4_Pretraining_on_unlabeled_data/2_Loss_Function_Generate/1_Loss_Function_Generate.ipynb)
         3. Training LLM     [==>](6_Natural_Language_Processing/4_Build_LLM/4_Pretraining_on_unlabeled_data/3_Training_LLM/1_Training_LLM.ipynb)
         4. Improvement by some randomness     [==>](6_Natural_Language_Processing/4_Build_LLM/4_Pretraining_on_unlabeled_data/4_Improvements/1_Improve_Training.ipynb)
         5. Load Parameter     [==>](6_Natural_Language_Processing/4_Build_LLM/4_Pretraining_on_unlabeled_data/5_Load_Parameter/1_Load_Trained_Weight.ipynb)


# convert jupyter notebook to python
"""jupyter nbconvert --to script file.ipynb"""


