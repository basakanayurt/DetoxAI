import streamlit as st
from detoxai.content_detector import *

st.title('Detect Toxicity in posts')
user_input = st.text_area("please put the text to be scanned" , 'type here')
st.subheader('Results')

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True, add_special_tokens=True,
                                                max_length=max_len, pad_to_max_length=True)
tokenizer.convert_ids_to_tokens(tokenizer(user_input)['input_ids'])


if st.checkbox("Show Tokens"):
    st.json(tokenizer.convert_ids_to_tokens(tokenizer(user_input)['input_ids']))

if st.button('Analyze'):
    # 'Starting the prediction...'
    print(user_input)
    model = AllToxicity()
    data = model.predict([user_input])

    print (data[["selfharm_pred","selfharm_prob"]])
    print(data[["hatespeech_pred","hatespeech_prob"]])
    print(data[["spam_pred","spam_prob"]])

    if data['prediction'][0] == 0:
        st.write("The post does not have any toxic content")
    else:
        st.write(data['prediction'][0], " detected with ", np.round(data['probability'][0]*100, 2) , ' % probability' )





