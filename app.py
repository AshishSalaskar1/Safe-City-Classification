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

texts = [
"I was going to my college in the evening and some guys standing in a group were constantly staring and smiling at me",
"sometime ago, I was travelling in an autorickshaw in Dehli. I was wearing a dress, so  naturally my calfs were uncovered. Men on their bikes and even from their cars kept peeping inside the autorickshaw to see. Althrough wich was humilating and creepy. Reaction  I tried to ignore and look at my phone",
"while walking towards the hostel two guys made weird facial expression which were inappropriate",
"Everyday while I stand in my balcony,people stare in such a way that it causes discomfort",
"While travelling at the bus, the man standing behind me was trying to touch me",
"A man, regular in my bus and my route and after we both got off at the same stop, he touched me in a wrong place and i felt very bad",
"there is a girl who was being touched aimlessly by boys in our school but she never reported to the teacher",
"t was around 8 pm, i got off at the vishwavidyalaya metro station and was getting back to my hostel. i saw a guy following me. after a few minutes, he pulled me back to ask if we can be friends. He touched me inappropriately. i shouted 'no' and ran without looking back to my hostel",
"Certain groups of boys were giving rude comments and giving vulgar expressions . It happened during afternoon ", "I live on JP Road very near to Chai Coffee, previously known as Barista and whenever I have to walk past that neighborhood, I always see a gang of people standing outside with their cars parked, blaring music and making comments. Once my niece was standing outside waiting for someone and one of the boys came up to her too. She got very scared. Also, there is a pan dabba next to Chai Coffee and I can see guys sitting on the wall drinking in public. The police know everything and don't do nothing. We are scared of complaining because we live in that area.",
"A person was continuously staring amp making comments on my looks.",
"I was all alone in the bus and a group of boys were whistling and commenting me. They even tried to touch me",
"Staring and by making indecent expressions and touching and all.All this happens generally in Metro and Buses.Even if you just pass by the bus stands in the evening,you might be the victim of inappropriate comments and all",
" This incident took place in Dtc bus no. 883 in morning.People in the bus pass comments and also try to find a reason to touch.This happens very often.",
"Boy was continuously following me till I reached the metro station. He had passed comments and also touched me by his hand"
"whistling and demeaning watching me on the dress that I was wearing while returning from a party."
]

# download_drive("1Z6bjXmyCaoEzXYo_tRDwLTsfeA2F3K3j","glove_vectors")

def get_random():
    n = np.random.randint(0,len(texts))
    return texts[n]


def predict():
    print("PREDICT")
    if input_text == "":
        return
    result, prob = final_function_1([input_text])
    st.markdown("**Entered story/description: **"+input_text)
    st.markdown("**Predicted Categories are : "+result[0]+"**")

    prob = [np.round(x,2) for x in prob[0]]

    ax = sns.barplot(prob ,class_labels, palette="winter", orient='h')
    ax.set_yticklabels(class_labels,rotation=0)
    plt.title("Probabilities of the Story belonging to each class")
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




