#Predicted_Answer_function
# Import lib

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertForQuestionAnswering
import torch

def calculate_cosine_similarity(text1, text2):
    # Create a CountVectorizer to convert text into a bag-of-words representation
    vectorizer = CountVectorizer()
    # Fit and transform the texts into vectors
    vectors = vectorizer.fit_transform([text1, text2])
    # Calculate cosine similarity
    similarity = cosine_similarity(vectors)
    return similarity[0, 1]

def Predicted_Answer( Ques) :
    
    # Prepare model, tokenizer
    model_name = "asafaya/bert-base-arabic"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForQuestionAnswering.from_pretrained(model_name)
    
    # Read file
    with open("الجرائم الالكترونية.txt", encoding="utf8") as f:
        File = f.read()
    
    # Split the file to sub files
    Files = File.split("………………………..")
    
    # Tokenize and pad each batch
    tokenized_batches = []
    for file in Files:
        batch_documents = file
        # Tokenize and pad
        tokenized_inputs = tokenizer(Ques, batch_documents, return_tensors="pt", padding=True, truncation=True,max_length = 512)
        tokenized_batches.append(tokenized_inputs)
    
    Answers=[]

    # Process logits 
    for tokenized_inputs in tokenized_batches:
        outputs = model(**tokenized_inputs)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        print(start_logits.shape)
        # Assuming we have start_logits and end_logits from the model
        start_index = torch.argmax(start_logits, dim=1).item()
        end_index = torch.argmax(end_logits, dim=1).item()
        
        # Extract the answer from the input tokens
        input_ids = tokenized_inputs["input_ids"][0].tolist()
        answer_tokens = input_ids[start_index:end_index+1]
        
        # Convert answer tokens back to string
        answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
        
        Answers.append(answer)


    similarity_score = calculate_cosine_similarity(text1, text2)
    print("Cosine Similarity Score:", similarity_score)

    listOfSim = []
    for i in range(len(Answers) ):
        similarity_score = calculate_cosine_similarity(Ques,Answers[i])
        listOfSim.append(similarity_score)

    return  answer