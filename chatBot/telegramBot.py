import nltk
import pickle
import numpy as np
import json
import random
import yandexTranslate

from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters
from keras.models import load_model
from nltk.stem import WordNetLemmatizer


model = load_model('model.h5')
lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

    return sentence_words


def get_bag_of_sentence(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)

    for lemmatized_word in sentence_words:
        for index, word in enumerate(words):
            if word == lemmatized_word:
                bag[index] = 1

    return np.array(bag)


def predict_class(sentence, model):
    bag_of_sentence = get_bag_of_sentence(sentence, words)
    predict_result = model.predict(np.array([bag_of_sentence]))[0]

    error_threshold = 0.25
    results = [[index, result] for index, result in enumerate(predict_result) if result > error_threshold]
    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []
    for result in results:
        return_list.append({"intent": classes[result[0]], "probability": str(result[1])})

    return return_list


def get_response(predicted_intents, intents_json):
    tag = predicted_intents[0]['intent']
    list_of_intents = intents_json['intents']

    for intent in list_of_intents:
        if intent['tag'] == tag:
            result = random.choice(intent['responses'])
            break

    return result


def chatbot_response(message):
    predicted_intents = predict_class(message, model)
    return get_response(predicted_intents, intents)


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message_en = yandexTranslate.translate(update.message.text, 'en')
    answer_en = chatbot_response(message_en)
    answer_ru = yandexTranslate.translate(answer_en, 'ru')
    await update.message.reply_text(answer_ru)


def main() -> None:
    application = Application.builder().token("<telegram_bot_token>").build()

    application.add_handler(MessageHandler(filters.TEXT, echo))

    application.run_polling(allowed_updates=Update.ALL_TYPES)


main()
