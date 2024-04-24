import pandas as pd
from preprocess import TextPreprocessor  # Import your TextPreprocessor class
text_processor = TextPreprocessor()

class data_selection():
    def __init__(self,name,nlabel,attention):
        self.name=name
        self.nlabel=nlabel
        self.attention=attention
    
    def HateOffensive(self,nlabel,attention):
        dataframe=pd.read_csv("/home/naseem_fordham/Pytorch/data/Hate_Offensive.csv")
        if nlabel<3:
            dataframe["class"] = dataframe["class"].apply(lambda x: 1.0 if x in [0., 1.] else 0.0)
            dataframe = dataframe.sample(frac = 1,random_state=42).reset_index(drop = True)
        dataframe.dropna(inplace=True)
        return dataframe

    def Imdb(self,nlabel,attention):
        dataframe=pd.read_csv('/home/naseem_fordham/Pytorch/data/IMDB_Dataset.csv')
        dataframe = dataframe.sample(frac = 1,random_state=42).reset_index(drop = True)

        return dataframe
    
    def Peace_violence(self,nlabel,attention):
        dataframe=pd.read_csv('/home/naseem_fordham/Pytorch/data/PV_2class.csv')
        dataframe["tweet"]=dataframe["tweet"].apply(lambda x : text_processor.text_preprocessing(x))
        dataframe = dataframe.sample(frac = 1,random_state=42).reset_index(drop = True)

        return dataframe

 
 
    def HateXplain(self,nlabel,attention):
        if attention=='YES':
            dataframe=pd.read_csv("/home/naseem_fordham/Pytorch/data/hatexplain.csv")
            
            if nlabel<3:
            #loading hate ofefsive data and converting it into 2 class data as toxic and non-toxic
                dataframe["class"] = dataframe["class"].apply(lambda x: 1.0 if x in [1.0,0.0] else 0.0)
                # dataframe=dataframe[['tweet','class']]
            
        elif attention=='NO': 
            dataframe=pd.read_csv("/home/naseem_fordham/Pytorch/data/hatexplain.csv")
            dataframe=dataframe[['tweet','class']]
            if nlabel<3:
                dataframe["class"] = dataframe["class"].apply(lambda x: 1.0 if x in [1.0,0.0] else 0.0)
                dataframe=dataframe[['tweet','class']]
            
            
            
            
            
        # dataframe=pd.read_csv("/home/naseem_fordham/Pytorch/data/Hate_Xplain.csv")
        # # dataframe=dataframe[~dataframe['class'].isin([0])]
        # if nlabel<3:
        #     #loading hate ofefsive data and converting it into 2 class data as toxic and non-toxic
        #     dataframe["class"] = dataframe["class"].apply(lambda x: 1.0 if x in [1.0,2.0] else 0.0)
            # dataframe=dataframe[~dataframe['class'].isin([1])]
            # dataframe["class"] = dataframe["class"].apply(lambda x: 1.0 if x in [2.0] else 0.0)
            
            
            
        # dataframe=dataframe[['class','tweet']]
        return dataframe
    
        
    def Xhate999_Gao_train(self,nlabel,attention):
        dataframe=pd.read_csv('/home/naseem_fordham/Pytorch/data/XHate999-EN-Gao-train.csv')
        dataframe["tweet"]=dataframe["tweet"].apply(lambda x : text_processor.text_preprocessing(x))
        dataframe = dataframe.sample(frac = 1,random_state=42).reset_index(drop = True)

        return dataframe
    
        
    def Xhate999_Gao_test(self,nlabel,attention):
        dataframe=pd.read_csv('/home/naseem_fordham/Pytorch/data/XHate999-EN-Gao-test.csv')
        dataframe["tweet"]=dataframe["tweet"].apply(lambda x : text_processor.text_preprocessing(x))
        dataframe = dataframe.sample(frac = 1,random_state=42).reset_index(drop = True)

        return dataframe
    
    def Xhate999_Wul_train(self,nlabel,attention):
        dataframe=pd.read_csv('/home/naseem_fordham/Pytorch/data/XHate999-EN-Wul-train.csv')
        dataframe["tweet"]=dataframe["tweet"].apply(lambda x : text_processor.text_preprocessing(x))
        dataframe = dataframe.sample(frac = 1,random_state=42).reset_index(drop = True)

        return dataframe
    
        
    def Xhate999_Wul_test(self,nlabel,attention):
        dataframe=pd.read_csv('/home/naseem_fordham/Pytorch/data/XHate999-EN-Wul-test.csv')
        dataframe["tweet"]=dataframe["tweet"].apply(lambda x : text_processor.text_preprocessing(x))
        dataframe = dataframe.sample(frac = 1,random_state=42).reset_index(drop = True)

        return dataframe
    
    def Xhate999_Trac_train(self,nlabel,attention):
        dataframe=pd.read_csv('/home/naseem_fordham/Pytorch/data/XHate999-EN-Trac-train.csv')
        dataframe["tweet"]=dataframe["tweet"].apply(lambda x : text_processor.text_preprocessing(x))
        dataframe = dataframe.sample(frac = 1,random_state=42).reset_index(drop = True)

        return dataframe
    
    def Xhate999_Trac_test(self,nlabel,attention):
        dataframe=pd.read_csv('/home/naseem_fordham/Pytorch/data/XHate999-EN-Trac-test.csv')
        dataframe["tweet"]=dataframe["tweet"].apply(lambda x : text_processor.text_preprocessing(x))
        dataframe = dataframe.sample(frac = 1,random_state=42).reset_index(drop = True)

        return dataframe
    

    
    def Xhate999_combine_train(self,nlabel,attention):
        dataframe=pd.read_csv('/home/naseem_fordham/Pytorch/data/XHate999-combine-train.csv')
        dataframe["tweet"]=dataframe["tweet"].apply(lambda x : text_processor.text_preprocessing(x))
        dataframe = dataframe.sample(frac = 1,random_state=42).reset_index(drop = True)

        return dataframe
    
    def Xhate999_combine_test(self,nlabel,attention):
        dataframe=pd.read_csv('/home/naseem_fordham/Pytorch/data/XHate999-combine-test.csv')
        dataframe["tweet"]=dataframe["tweet"].apply(lambda x : text_processor.text_preprocessing(x))
        dataframe = dataframe.sample(frac = 1,random_state=42).reset_index(drop = True)

        return dataframe


    