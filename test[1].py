import streamlit as st
import tensorflow.keras
import pandas as pd
from PIL import Image, ImageOps
import numpy as np
import math
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import urllib.request
import json
import time

if 'model1_computed' not in st.session_state:
    st.session_state['model1_computed'] = False
    

st.markdown(
    """
    <style>
    .checkbox-toggle .stCheckbox input:checked + .stCheckmark:before {
        content: "ON";
    }
    .checkbox-toggle .stCheckbox input:checked + .stCheckmark {
        background-color: #2196F3;
    }
    .checkbox-toggle .stCheckbox .stCheckmark:before {
        content: "OFF";
    }
    .checkbox-toggle .stCheckbox .stCheckmark {
        border-radius: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

API_KEY = 'AIzaSyBS1qwh3dLES65V6QSVXrcqpiyQ9fdzg5E'
MAX_RESULTS = 10
l=['Thickening or swelling','Lump in the breast or underarm', 'Pain at affected Area', 'Weight Gain']
st.set_option('deprecation.showfileUploaderEncoding', False)
np.set_printoptions(suppress=True)

model = tensorflow.keras.models.load_model('keras_model.h5')

st.title("Breast Cancer Detection using Convolutional Neural Network")

# Define the search query
QUERY = 'Cancer Survival Stories Womens'

def get_state():
    if st.session_state['login'] == True:
        st.session_state['login'] = False
    else:
        st.session_state['login'] = True
    
    return st.session_state.login 

# Call the YouTube Data API to search for videos
def youtube_search(query, max_results):
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    search_response = youtube.search().list(q=query, type='video', part='id,snippet', maxResults=max_results).execute()

    videos = []
    for search_result in search_response.get('items', []):
        if search_result['id']['kind'] == 'youtube#video':
            videos.append(search_result)
    return videos

# Get the video details from the YouTube Data API
def get_video_details(video_id):
    url = f'https://www.googleapis.com/youtube/v3/videos?id={video_id}&key={API_KEY}&part=snippet,statistics'
    response = urllib.request.urlopen(url)
    data = json.loads(response.read())
    return data

# Display the video gallery
def show_video_gallery(videos):
    for video in videos:
        video_id = video['id']['videoId']
        video_details = get_video_details(video_id)

        st.subheader(video_details['items'][0]['snippet']['title'])
        # st.write(video_details['items'][0]['snippet']['description'])
        st.video(f'https://www.youtube.com/watch?v={video_id}')
        
# Security
#passlib,hashlib,bcrypt,scrypt
import hashlib
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False
# DB Management
import sqlite3 
conn = sqlite3.connect('data.db')
c = conn.cursor()
# DB  Functions
def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')


def add_userdata(username,password):
    c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
    conn.commit()

def login_user(username,password):
    c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
    data = c.fetchall()
    return data



def view_all_users():
    c.execute('SELECT username FROM userstable')
    data = c.fetchall()
    return data


def main():
    st.sidebar.subheader("Breast Cancer Detection using Machine Learning")
    menu = ["Login","SignUp", "Inspiration"]
    choice = st.sidebar.selectbox("Menu",menu)
    
    if choice == "SignUp":
        st.subheader("Create New Account")
        new_user = st.text_input("Username")
        new_password = st.text_input("Password",type='password')
        
        if st.button("Signup"):
            create_usertable()
            add_userdata(new_user,make_hashes(new_password))
            st.success("You have successfully created a valid Account")
            st.info("Go to Login Menu to login")

    elif choice=="Inspiration":
        videos = youtube_search(QUERY, MAX_RESULTS)
        show_video_gallery(videos)
    
    elif choice == "Login":
        st.subheader("Please Enter Valid Credentials")

        username = st.sidebar.text_input("User Name")
        password = st.sidebar.text_input("Password",type='password')
        col1, col2 = st.sidebar.columns(2)
        a = col1.button("Login Now")
        b = col2.button("Logout")
            
        if a:
            create_usertable()
            hashed_pswd = make_hashes(password)
            result = login_user(username,check_hashes(password,hashed_pswd))
            if result:
                st.session_state.model1_computed = True
        if b:
            st.session_state.model1_computed = False
        if st.session_state.model1_computed:
            st.success("Logged In as {}".format(username))
            st.sidebar.success("login Success.")
            form = st.form("my_form")
            col1, col2 = form.columns(2)
            with col1:
                first_name = col1.text_input("First Name") 
                Age = st.number_input('Enter Age', value=1)
                bg = st.selectbox('Blood Group', ('A+', 'A-', 'O+', 'O-', 'AB+', 'AB-'))
            with col2:
                last_name = st.text_input("Last Name")
                options = st.multiselect("Symptoms",l)
                city = st.text_input("Enter City")
            uploaded_file = form.file_uploader("Choose an image...", type=["JPG","PNG"])
            submitted = form.form_submit_button("Submit")
            if submitted:
                col1, col2 = form.columns(2)
                data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
                image = Image.open(uploaded_file)
                col1.image(image, caption='Original Image')
                size = (224, 224)
                image = ImageOps.fit(image, size, Image.ANTIALIAS)

                image_array = np.asarray(image)
                #image.show()
                col2.image(255 - image_array, caption='Preprocessed Image')
                
                normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
                data[0] = normalized_image_array
                prediction = model.predict(data)
                data1 = prediction[0][0]
                data2 = prediction[0][1]
                print(data1, data2)
                data = np.rint(prediction)
                print(data)
                with st.spinner('Wait for it...'):
                    time.sleep(5)
                if(data1>0.3):
                    form.error("Hello "+first_name+", We're sorry to inform you that our screening has detected the presence of Malignant breast cancer with " + str(round(data1*100,2)) + "% Matching with Malignant Cell." )
                else:
                    form.success("Hello "+first_name+", Your breast cancer screening test results came back negative with "+ str(round(data1*100,2)) + "% Matching with Malignant Cell. This means that no signs of cancer were detected.")
        else:
            st.warning("Please Login with correct Crdentails")

            
if __name__ == '__main__':
	main()
