from google.cloud import translate
from retry import retry
from tqdm import tqdm

def get_list_of_chunks(texts, chunk_size):
    list_chunks = []

    for i in range(0, len(texts), chunk_size):
        list_chunks.append(texts[i:i + chunk_size])

    return list_chunks


class Translator:

    def __init__(self, project_id="opportune-chess-406818"):
        self.client = translate.TranslationServiceClient()
        location = "global"
        self.parent = f"projects/{project_id}/locations/{location}"

    @retry(tries=5, jitter=1, backoff=2)
    def get_translations(self, texts, source_lang, target_lang):
        print("entered get_translations")
        all_translations = []

        list_of_chunks = get_list_of_chunks(texts, 200)

        for texts_chunk in tqdm(list_of_chunks):
            response = self.client.translate_text(
                request={
                    "parent": self.parent,
                    "contents": texts_chunk,
                    "mime_type": "text/plain",
                    "source_language_code": source_lang,
                    "target_language_code": target_lang,
                }
            )

            all_translations.extend([translation.translated_text for translation in response.translations])

        return all_translations