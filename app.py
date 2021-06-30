import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit.proto.Markdown_pb2 import Markdown

from model_methods import download_drive
from model_methods import final_function_1

class_label_names = {0:"None",1:"Staring",2:"Groping",3:"Commenting",4:"Commenting & Staring",5:"Staring & Groping",6:"Commenting & Groping",7:"Commenting, Staring and Groping"}

st.set_option('deprecation.showPyplotGlobalUse', False)

class_labels = list(class_label_names.values())

texts = np.load("texts.npy")

def get_random():
    n = np.random.randint(0,len(texts))
    return texts[n]


def predict():
    print("PREDICT")
    if input_text == "":
        return
    result, prob = final_function_1([input_text])
    st.markdown("Predicted Categories are : **"+result[0]+"**")

    prob = [np.round(x,2) for x in prob[0]]

    ax = sns.barplot(prob ,class_labels, palette="winter", orient='h')
    ax.set_yticklabels(class_labels,rotation=0)
    plt.title("Probabilities of each class")
    for index, value in enumerate(prob):
        plt.text(value, index,str(value))
    st.pyplot()

st.title("Classification of Sexual Harrasment Stories")

st.markdown('**Objective** : Given a short description/story regarding a Sexual Harrasment incident, our Model tries to predict which category the story belongs to.')
st.markdown('The model can predict if it belongs to the following three Categories : **Staring, Groping, Commenting** ')


st.title("Test out the Model")
input_text = st.text_input('Enter your short description/story below', '')
if st.button("Predict"):
    predict()

if st.button("Generate some random stories and Predict"):
    input_text = get_random()
    # st.text_input('Enter your short description/story below', '')
    predict()




