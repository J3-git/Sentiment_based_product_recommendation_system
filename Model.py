import pandas as pd
import numpy as np
import pickle

class Recommendation_system:


    def __init__(self):
        self.main_df = pd.read_pickle('Data/sample30.pkl')
        self.processed_df = pd.read_pickle('Data/preprocessed_df.pkl')

        with open('Model/recommendation_UBCF.pkl', 'rb') as file:
            self.recomendation_matrix = pickle.load(file)

        with open('Model/tfidf_vectorizer.pkl', 'rb') as file:
            self.word_vectorizer = pickle.load(file)

        with open('Model/SentimentClassification_logit.pkl', 'rb') as file:
            self.SentimentClassification_logit = pickle.load(file)

        self.product_sentiment_dict = {}

        
    def top_4_products(self, username):
        if (username in self.main_df['reviews_username'].values) and (len(username) !=0):
            invalid_username = False
            recomended_20 = self.recomendation_matrix.loc[username].sort_values(ascending=False)[0:20]
            recomended_20_names = recomended_20.index
            
            review_filtered = self.processed_df

            recomended_20_df = review_filtered[self.main_df.name.isin(recomended_20_names)]
            
            X_reviews = self.word_vectorizer.transform(recomended_20_df['combined_txt'].tolist())
            Y_sentiment_prediction = self.SentimentClassification_logit.predict(X_reviews)
            
            recomended_20_df['sentiment_predicted'] = pd.Series(Y_sentiment_prediction)
            
            for product in recomended_20_names:
                total_positive_reviews = recomended_20_df[recomended_20_df.name == product]['sentiment_predicted'].sum()
                total_product_reviews = recomended_20_df[recomended_20_df.name == product]['sentiment_predicted'].index.nunique()
                positive_sentiment_percent = round((total_positive_reviews / total_product_reviews) * 100,2)
                self.product_sentiment_dict[product] = positive_sentiment_percent
                
            top_4_recomended = sorted(self.product_sentiment_dict.items(), key=lambda item: item[1],reverse=True)[:4]
            output = pd.DataFrame(top_4_recomended, columns=['Product','Sentimen(%)'],index=np.arange(1, len(top_4_recomended)+1))

            return output,invalid_username
        else:
            invalid_username = True
            output = 'Invalid username entered. PLease enter a valid username.'
            return output,invalid_username