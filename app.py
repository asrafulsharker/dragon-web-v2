from flask import Flask, render_template, request, url_for
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input

import numpy as np
import os  


app = Flask(__name__)


dic = {
    0: {'name': 'After Bollom Dry Flower', 'description': 'This is a normal stage in the flowering and fruiting cycle of the dragon fruit tree. Although moths and bats can pollinate dragon fruit flowers at night, hand pollination serves as an alternative when these pollinators are not present or to increase fruit production. Most of the farmers in our country cultivate by hand pollination. A little brush or cotton swab is needed to perform this pollination. The dragon fruit flower must be pollinated at midnight because it blooms at the odd time of 9:00 PM and fades at 9:00 AM. Some dragon fruit cultivars are self-pollinating, pollinated by wind or insects. If it doesnt bear fruit in one season, we know what kind of tree it is. After pollination, dragon fruit flowers wither, which is a normal process that signals the transition from flower to fruit formation. The plants reproductive cycle and the generation of ripe, healthy fruits depend on this process. This is an indication that the tree will soon start bearing dragon fruit.'},
    1: {'name': 'Before Bloom Mature Bud', 'description': 'The mature bud stage is the final phase before the bud blossoms into a flower. At this point, the bud has reached its full size and is about to open. The color becomes vibrant and intense, reflecting the dragon fruit’s characteristic hues. The outer surface of the bud may appear smoother and more polished compared to the developing bud stage. During the mature bud stage, the bud prepares for flowering, which is an essential step in the dragon fruit’s reproductive cycle. The bud is packed with energy and nutrients, ready to support the growth of flowers and eventual fruit formation. The timing of bud maturation may vary depending on the dragon fruit variety and environmental factors such as temperature and sunlight. Once the mature bud blooms, it reveals a stunning flower that lasts only for an abbreviated period, usually for a single night. It happens between days 15-17.'},
    2: {'name': 'Bloom Flower', 'description': 'The dragon fruit flower is a large, vibrant bloom that opens in the evening and lasts only for one night. Its trumpet-like shape and colorful petals, ranging from white to pink, emit a sweet tropical fragrance. The flower relies on nocturnal pollinators like moths and bats attracted by its scent and color. As they visit for nectar, they unknowingly transfer pollen, aiding in reproduction. This fleeting spectacle, measuring 25 to 30 centimeters (about 11.81 in) in diameter, adds to its allure and rarity. But there are some rare varieties that cannot pollinate on their own. They must be pollinated by hand pollination technique. After successful pollination, the dragon fruit flower begins its transformation into some fruit. The base of the flower swells and enlarges, forming the initial structure of the fruit.'},
    4: {'name': 'Fresh Mature Fruit', 'description': 'A mature dragon fruit has a visually striking appearance. It typically has a vibrant outer skin with shades of pink, red, or yellow, depending on the variety. The skin is usually covered in scales or spikes, giving it a unique and exotic look. The flesh of a mature dragon fruit is soft and juicy. Harvesting time for dragon fruits depends on the variety and local growing conditions. The fruit is ready for harvest around 30-50 days (about 1 and a half months) after flowering. Signs of maturity include a vibrant color, slight softness to the touch, and a pleasant aroma. It is important to harvest the fruit at its optimal stage of maturity to ensure the best flavor and texture.'},
    3: {'name': 'Freash Premature Fruit', 'description': 'After successful pollination, the fruit formation stage can last 1 to 2 weeks. The base of the flower swells and develops into the fruit. The fruit growth stage varies depending on the dragon fruit variety, growing conditions, and climate. On average, it can take anywhere from 30 to 50 days (about 1 and a half months) for the fruit to reach its mature size. Premature dragon fruits are smaller than mature fruits. They have a round or oval shape with firm skin. The color of the premature fruit is not as vibrant or intense as that of a mature fruit, and the outer skin still displays shades of green.'},
    5: {'name': 'Premature Bud', 'description': 'The premature bud of a dragon fruit is the early stage of growth, emerging from the stem or branch of a mature plant. It is small and tightly closed, displaying a primarily green color with hints of pink or reddish tones. Serving as a protective covering, the bud shields the developing flower and fruit components from external factors. It accumulates essential nutrients, transported from the plants root system, to support further growth. As the bud matures, it eventually bursts open, revealing the inner structures of the flower. The premature bud is the starting point, embodying the plants growth potential and setting the stage for the development of a mature fruit. It Takes time about 14-16 days (about 2 and a half weeks).'}
}




model = load_model('fine_tuned_inceptionv3.h5')

model.make_predict_function()

def predict_label_with_description(img_path):
    i = load_img(img_path, target_size=(240, 240))  # Adjust to the size the model was trained with
    i = img_to_array(i)
    i = np.expand_dims(i, axis=0)
    i = preprocess_input(i)
    p = model.predict(i)
    predicted_class_index = np.argmax(p)
    predicted_class_info = dic[predicted_class_index]
    confidence = p[0][predicted_class_index] * 100
    return predicted_class_info, confidence




@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")

@app.route("/about")
def about_page():
    return "Dragon FruitHarvest Master Your Ultimate Guide to Perfectly Timed Harvesting and Maturity Detection"

@app.route("/submit", methods=["GET", "POST"])
def get_output():
    if request.method == "POST":
        img = request.files["my_image"]
        img_path = os.path.join(app.root_path, "static", img.filename)
        img.save(img_path)
        predicted_class_info, confidence = predict_label_with_description(img_path)
        return render_template("index.html", predicted_class_info=predicted_class_info, confidence=confidence, img_path=img.filename)

    return render_template("index.html")

if __name__ =='__main__':
    app.run(debug=False, host='0.0.0.0')
