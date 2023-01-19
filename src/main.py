import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi.security import HTTPBearer
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn


security = HTTPBearer()
origins = ["*"]

# app = FastAPI(dependencies=[Depends(security)])
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/fetchAnswer')
def fetchAnswer(question: str):
    prediction = email_classifier(question)
    if prediction == "LABEL_1":
        print("This is a Question")
        return "This is a Question"
    else:
        print("This is not a Question")
        return "This is not a Question"



# # load huggingface transformer question-vs-statement classification model
# tokenizer = AutoTokenizer.from_pretrained("shahrukhx01/question-vs-statement-classifier")
# model = AutoModelForSequenceClassification.from_pretrained("shahrukhx01/question-vs-statement-classifier")

# # write the tokenizer, model to file
# file = open("./models/model.shn", "wb")
# pickle.dump((tokenizer, model), file)
# file.close()


def email_classifier(text):
    """
    Tokenizes a given sentence and returns the predicted class. 
    
    Returns:
    LABEL_0 --> sentence is predicted as a statement
    LABEL_1 --> sentence is predicted as a question
    """
    tokenizer = AutoTokenizer.from_pretrained(
        "shahrukhx01/question-vs-statement-classifier")
    model = AutoModelForSequenceClassification.from_pretrained(
        "shahrukhx01/question-vs-statement-classifier")

    inputs = tokenizer(f"{text}", return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_class_id = logits.argmax().item()
    return model.config.id2label[predicted_class_id]

# def transcribeText(audioFile):
#     """
#     Transcribes the given audio file and returns the text
#     """


#     # load model and processor
#     processor = WhisperProcessor.from_pretrained("openai/whisper-base")
#     model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")

#     # load dummy dataset and read soundfiles
#     ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
#     input_features = processor(ds[0]["audio"]["array"], return_tensors="pt").input_features 

#     # Generate logits
#     logits = model(input_features, decoder_input_ids = torch.tensor([[50258]]).logits 
#     # take argmax and decode
#     predicted_ids = torch.argmax(logits, dim=-1)
#     transcription = processor.batch_decode(predicted_ids)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0")


# if __name__ == "__main__":
#     text = str(input("Enter sentence: "))  # get sentence
#     prediction = email_classifier(text)
#     if prediction == "LABEL_1":
#         print("This is a Question")
#     else:
#         print("This is not a Question")

