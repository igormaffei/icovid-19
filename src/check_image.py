from keras.models import load_model
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras_vggface.utils import preprocess_input
from keras import backend as np
from keras.preprocessing.image import ImageDataGenerator

image_target = load_img('../img_old/trainning/covid19/7C69C012-7479-493F-8722-ABC29C60A2DD.jpeg', target_size=(224, 224))
#image_target = load_img('../img_old/trainning/notcovid19/X-ray_of_cyst_in_pneumocystis_pneumonia_1.jpg', target_size=(224, 224))
#image_target = load_img('../img_old/trainning/notcovid19/aspiration-pneumonia-5-day0.jpg', target_size=(224, 224))
#image_target = load_img('../img_old/trainning/notcovid19/pneumococcal-pneumonia-day0.jpg', target_size=(224, 224))
#image_target = load_img('../img_old/trainning/notcovid19/legionella-pneumonia-2.jpg', target_size=(224, 224))
#image_target = load_img('../img_new2/trainning/not_covid19/NORMAL (54).png', target_size=(224, 224))
#image_target = load_img('../img_new2/trainning/covid19/COVID-19 (5).png', target_size=(224, 224))

#Process de Analises
img_test = img_to_array(image_target)
img_test = np.expand_dims(img_test, axis=0)
img_test = preprocess_input(img_test)

#Load Model
model = load_model('../model/covid_vgg16.h5')

#Execute prediction from model
predictions = model.predict(img_test,verbose=0)
predicted_class= np.argmax(predictions,axis=-1)

#Load Classfication
train_path = '../img_new2/trainning/'
train_datagen = ImageDataGenerator()
train_generator = train_datagen.flow_from_directory(directory=train_path)

class_indices = (train_generator.class_indices)

# Show Results
print(class_indices)
print(predicted_class)
print(predictions)
print("> COVID-19     : %.2f " % predictions[0][0] )
print("> NOT COVID-19 : %.2f " % predictions[0][1] )