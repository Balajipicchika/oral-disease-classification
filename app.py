from flask import Flask, render_template, request,jsonify
from flask import Flask, request, jsonify
# from tensorflow.keras.models import load_model
from models import model_loader
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
# from collections import Counter
import io
import pandas as pd

app = Flask(__name__)
df = model_loader.load_csv()

# tensorflow should be 2.10 or 2.15
# keras should be 2.10 or 2.15 or 2.9

related_symptoms = {
    "gum disease": ["a cracked tooth", "worn-down fillings or crowns"],
    "a cracked tooth": ["gum disease", "worn-down fillings or crowns"],
    "worn-down fillings or crowns": ["gum disease", "a cracked tooth"],
    "Black, white, or brown tooth stains":["Pain when you bite down","Holes or pits in your teeth"],
    "Pain when you bite down":["Black, white, or brown tooth stains","Holes or pits in your teeth"],
    "Holes or pits in your teeth":["Black, white, or brown tooth stains","Pain when you bite down"],
    "Yellowish discoloration":["Cracked or chipped teeth","Grooves on your teeth's surface"],
    "Cracked or chipped teeth":["Grooves on your teeth's surface","Yellowish discoloration"],
    "Grooves on your teeth's surface":["Yellowish discoloration","Cracked or chipped teeth"],
    "bleeding":["pain","sore throat"],
    "pain":["bleeding","sore throat"],
    "sore throat":["pain","bleeding"],
    "Ear Pain":["Dramatic weight loss","Difficulty chewing or swallowing"],
    "Dramatic weight loss":["Ear Pain","Difficulty chewing or swallowing"],
    "Difficulty chewing or swallowing":["Ear Pain","Dramatic weight loss"],
    "Bad breath":["Painful chewing","Red and swollen gums","Tender or bleeding gums"],
    "Painful chewing":["Bad breath","Red and swollen gums","Tender or bleeding gums"],
    "Red and swollen gums":["Painful chewing","Bad breath","Tender or bleeding gums"],
    "Tender or bleeding gums":["Red and swollen gums","Painful chewing","Bad breath"]
}


model_mango = model_loader.load_my_model()
learning_rate = 0.001
model_mango.compile(loss='sparse_categorical_crossentropy',
                    optimizer=tf.keras.optimizers.Adamax(learning_rate=learning_rate),
                    metrics=['accuracy'])

class_names = ['Data caries', 'Gingivitis', 'Healthy', 'Mouth Ulcer', 'Tooth Discoloration']

def get_precautions(disease):
    precautions_dict = {
        'Data caries': ["1.Brushing Technique :Brush your teeth at least twice a day using a fluoride toothpaste and a soft-bristled toothbrush.",
                         "2.Flossing : Floss daily to clean between your teeth and below the gumline, where a toothbrush cannot reach. ",
                         "3.Limit Sugary Foods and Beverages : Reduce your consumption of sugary and acidic foods and beverages, such as candy, soda, fruit juices, and sports drinks.",
                         "4.Healthy Diet :Eat a balanced diet rich in fruits, vegetables, whole grains, and lean proteins.",
                         "5.Drink Water :Drink plenty of water throughout the day, especially fluoridated water, which helps strengthen tooth enamel and prevent tooth decay."],
        'Gingivitis': ["1.Brushing Technique :Use a soft-bristled toothbrush and brush your teeth at least twice a day, preferably after meals and before bedtime. ", 
                       "2.Flossing : Floss daily to remove plaque and food particles from between your teeth and below the gumline, where a toothbrush cannot reach.",
                       "3.Regular Dental Check-ups :Schedule regular dental check-ups and professional cleanings with your dentist or dental hygienist. ",
                       "4.Healthy Diet :Eat a balanced diet rich in fruits, vegetables, whole grains, and lean proteins.",
                       "5.Avoid Tobacco Products :Avoid smoking and using tobacco products, as they increase the risk of gum disease and hinder the healing process."],
        'Mouth Ulcer': ["1.Maintain Good Oral Hygiene : Brush your teeth at least twice a day and floss daily to remove plaque, bacteria, and food particles that can irritate the mouth and contribute to the development of mouth ulcers.",
                        "2.Avoid Trigger Foods :Identify and avoid foods that may trigger or exacerbate mouth ulcers, such as acidic or spicy foods, citrus fruits, tomatoes, nuts, chocolate, and salty snacks.",
                        "3.Manage Stress :Stress can weaken the immune system and increase the likelihood of developing mouth ulcers.",
                        "4.Avoid Trauma to the Mouth :Be cautious when brushing and flossing to avoid injuring the delicate tissues of the mouth.",
                        "5.Rinse with Saltwater : Rinse your mouth with a saltwater solution (1 teaspoon of salt dissolved in 8 ounces of warm water) several times a day to help reduce inflammation and promote healing of mouth ulcers."],
        'Tooth Discoloration': ["1.Maintain Good Oral Hygiene :Brush your teeth at least twice a day and floss daily to remove plaque, bacteria, and food particles that can contribute to tooth discoloration. ",
                                "2.Limit Staining Foods and Beverages : Reduce your consumption of staining foods and beverages such as coffee, tea, red wine, cola, berries, and tomato sauce. I",
                                "3.Drink Water : Drink plenty of water throughout the day, especially after consuming staining foods and beverages.",
                                "4.Use a Straw :When drinking staining beverages such as coffee, tea, or cola, use a straw to minimize contact with your teeth and reduce the risk of discoloration.",
                                "5.Quit Smoking and Tobacco Use : Avoid smoking and using tobacco products, as they can cause severe tooth discoloration, gum disease, and oral cancer. "],
        'Healthy':["no precautions requries"]
    }
    return precautions_dict.get(disease, [])

# df = pd.read_csv('D:/Web-Development/Projects/oral-disease-classification-mini-project/mini-project/project-file/Data-of-teeth.csv')

related_symptoms = {
    "gum disease": ["a cracked tooth", "worn-down fillings or crowns"],
    "a cracked tooth": ["gum disease", "worn-down fillings or crowns"],
    "worn-down fillings or crowns": ["gum disease", "a cracked tooth"],
    "Black, white, or brown tooth stains":["Pain when you bite down","Holes or pits in your teeth"],
    "Pain when you bite down":["Black, white, or brown tooth stains","Holes or pits in your teeth"],
    "Holes or pits in your teeth":["Black, white, or brown tooth stains","Pain when you bite down"],
    "Yellowish discoloration":["Cracked or chipped teeth","Grooves on your teeth's surface"],
    "Cracked or chipped teeth":["Grooves on your teeth's surface","Yellowish discoloration"],
    "Grooves on your teeth's surface":["Yellowish discoloration","Cracked or chipped teeth"],
    "bleeding":["pain","sore throat"],
    "pain":["bleeding","sore throat"],
    "sore throat":["pain","bleeding"],
    "Ear Pain":["Dramatic weight loss","Difficulty chewing or swallowing"],
    "Dramatic weight loss":["Ear Pain","Difficulty chewing or swallowing"],
    "Difficulty chewing or swallowing":["Ear Pain","Dramatic weight loss"],
    "Bad breath":["Painful chewing","Red and swollen gums","Tender or bleeding gums"],
    "Painful chewing":["Bad breath","Red and swollen gums","Tender or bleeding gums"],
    "Red and swollen gums":["Painful chewing","Bad breath","Tender or bleeding gums"],
    "Tender or bleeding gums":["Red and swollen gums","Painful chewing","Bad breath"]
}


@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/')
def home():
    return render_template('index.html')

# @app.route('/about')
# def about():
#     return render_template('about.html')

@app.route('/prediction')
def predictions():
    return render_template('Prediction.html')

@app.route('/symptoms')
def symptoms():
    return render_template('symptoms.html')

@app.route('/prediction', methods=['GET', 'POST'])
def fetalbwprediction():
    precautions = []
    
    if request.method == 'POST':
        img = request.files['image']
        img_bytes = img.read()
        img = image.load_img(io.BytesIO(img_bytes), target_size=(224, 224))  
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        images = np.vstack([x])
        classes = model_mango.predict(images)
        predicted_class_idx = np.argmax(classes)
        predicted_class = class_names[predicted_class_idx]
        confidence = np.max(classes)

        precautions = get_precautions(predicted_class)

        res = f"This Looks like a {predicted_class}, with {confidence * 100:.2f} % confidence."
    
    return render_template('result.html', result=res, precautions=precautions)

@app.route('/get_related_symptoms', methods=['POST'])
def get_related_symptoms():
    selected_symptom = request.json['selected_symptom']
    if selected_symptom in related_symptoms:
        return jsonify({'related_symptoms': related_symptoms[selected_symptom]})
    else:
        return jsonify({'related_symptoms': []})
@app.route('/get_disease_and_treatment', methods=['POST'])


def get_disease_and_treatment():
    data = request.json
    symptom1 = data['symptom1']
    symptom2 = data['symptom2']
    symptom3 = data['symptom3']

    filtered_df = df[(df['Symptom 1'] == symptom1) & (df['Symptom 2'] == symptom2) & (df['Symptom 3'] == symptom3)]

    if not filtered_df.empty:
        disease = filtered_df.iloc[0]['Disease']
        treatment = filtered_df.iloc[0]['Treatment']
        return jsonify({'disease': disease, 'treatment': treatment})
    else:
        return jsonify({'disease': 'No match found', 'treatment': ''})



if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=10000)