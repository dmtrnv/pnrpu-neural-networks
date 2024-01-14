import requests


IAM_TOKEN = '<iam_token>'
FOLDER_ID = '<folder_id>'


def translate(text, target_language):
    body = {
        "targetLanguageCode": target_language,
        "texts": [text],
        "folderId": FOLDER_ID,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer {0}".format(IAM_TOKEN)
    }

    response = requests.post('https://translate.api.cloud.yandex.net/translate/v2/translate',
                             json=body,
                             headers=headers
                             )

    return response.json()['translations'][0]['text']
