import streamlit as st
from detoxai.classifiers import *

st.title('Detect Toxicity in posts')
user_input = st.text_area("please put the text to be scanned" , 'type here')
st.subheader('Results')


tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True, add_special_tokens=True,
                                                max_length=256, pad_to_max_length=True)
tokenizer.convert_ids_to_tokens(tokenizer(user_input)['input_ids'])

if st.checkbox("Show Tokens"):
    st.json(tokenizer.convert_ids_to_tokens(tokenizer(user_input)['input_ids']))

if st.button('Analyze'):
    # 'Starting the prediction...'
    print(user_input)
    model = Models(task='all')
    data = pd.DataFrame({'data': [user_input], 'prediction': [None], 'probability': [None]})
    preprocess_rows(data)
    data = model.predict_from_data(data)

    if data['prediction'][0]==0:
        st.write("The post does not have any toxic content")
    else:
        st.write(data['prediction'][0], " detected with ", np.round(data['probability'][0]*100,2) , ' % probability' )





