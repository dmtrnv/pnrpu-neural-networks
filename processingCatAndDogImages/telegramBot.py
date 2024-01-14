import numpy as np
import keras
import requests

from PIL import Image
from keras.preprocessing import image
from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters


prepared_model = keras.models.load_model('model.h5')


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    file = await update.get_bot().getFile(update.message.photo[0].file_id)
    img_url = file.file_path
    img = Image.open(requests.get(img_url, stream=True).raw)
    img = img.resize((100, 100))
    img_array = image.img_to_array(img)
    img = np.expand_dims(img_array, axis=0)
    img = img / 255

    prediction = prepared_model.predict(img)
    chance_it_is_cat = prediction[0][0] * 100.0
    chance_it_is_dog = prediction[0][1] * 100.0
    await update.message.reply_text(f'Это кот с вероятностью  {chance_it_is_cat:.4g} %\n'
                                    f'Это собака с вероятностью {chance_it_is_dog:.4g} %')


def main() -> None:
    application = Application.builder().token("<telegram_bot_token>").build()

    application.add_handler(MessageHandler(filters.PHOTO, echo))

    application.run_polling(allowed_updates=Update.ALL_TYPES)


main()
