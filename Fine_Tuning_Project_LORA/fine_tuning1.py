from peft import LoraConfig, get_peft_model, TaskType # its is a library which contains the lora adapters
from transformers import AutoTokenizer, AutoModelForSequenceClassification #from huggingface we will get our model BERT
from transformers import TrainingArguments, Trainer
import accelerate # To get to use cpu for our task efficiently and for multiple task
from datasets import Dataset#for coverting python dic to reable datasets
import torch  
import torch.nn.functional as F 


# Raw text data extraction from the data sets
data = {
    "text":[
        "I loved this movie!",
        "This was a fantastic experience",
        "Terrible acting and boring story.",
        "Worst movie I've ever seen."
    ],
    "label":[1,1,0,0] #1 means positve sentiment and 0 means negative sentiment
}


# convert to hugging face datasets
dataset = Dataset.from_dict(data)
# Display the datasets
print(dataset)

"""Load pretrained BERT model and tokenizer"""
#I will use bert-base-uncased

#Define model name
model_name = "bert-base-uncased"
#Load tokenizer for the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
#Load the model with a classification head
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)


print(model.config)
# confirm laoded
print("Tokenizer and model loaded successfully!")



"""Performing preprocess function and tokenize dataset"""

def preprocess(example):
    return tokenizer(
        example["text"], #this is the text which we are having
        padding="max_length",#that every sentence is in this size
        truncation=True,# is sentence is longer make it of this size
        max_length = 64 #this is the size
    )

#apply tokenizer to all examples
tokenized_dataset = dataset.map(preprocess,batched=True) #apply this for all the datasets

#Convert to pytorch format for the training
tokenized_dataset.set_format(type="torch",columns=["input_ids","attention_mask","label"]) # convert the string into numerical normal(torch)

print(tokenized_dataset[0])


#Define Lora Configuration
config = LoraConfig(
    r=8, #rank of low-rank matrices (smaller = lighter)
    lora_alpha=16, #sacling factor
    target_modules=["query","value"], #inject lora in these layer
    lora_dropout=0.1, #dropout for regularization
    bias="none", #dont update biases
    task_type=TaskType.SEQ_CLS #for sequence classification
)


#apply LoRA to the model
model = get_peft_model(model,config)

model.print_trainable_parameters() #showing the para targetd and are present


"""Using the training module where we will start training the model"""

#define training configuration
training_args =TrainingArguments(
    output_dir="./results", #store the trained data in this directory
    per_device_train_batch_size=2, #take 2 examples at a time
    num_train_epochs = 5, #saying to go thorugh whole data set 5 times, everytime you pass is epochs=1
    #more epochs = more accuracy, but too many = overfitting
    no_cuda = True, # force to not use a GPU
    logging_dir="./logs", #save training log info into a folder called logs
    logging_steps=1, #print log training progress every single training step
    save_strategy="no" # dont save model after every step
)


#Intialize Trainer
trainer = Trainer(
    model = model, # model to be trained
    args= training_args, #the configuration we just set
    train_dataset=tokenized_dataset, # the sentences and labels
    tokenizer = tokenizer #the tokenizer is used so trainer know how to convert text back & forth
)



trainer.train() #command to start the pipeline

#using the new model for getting our output
#model = AutoModelForSequenceClassification.from_pretrained("./results")


def predict_sentiment(text):
    #tokenize the input text
    inputs = tokenizer([text],return_tensors='pt',truncation=True,padding="max_length",max_length = 64)
    
    #convert the model into evaluation mode
    model.eval()

    #run the model with no gradient
    with torch.no_grad(): # saves memory as we are not doing backpropogation
        outputs = model(**inputs)#pass input_ids and attention_masks into BERT


    #get logits and apply softmax
    logits = outputs.logits #raw scores
    probs = F.softmax(logits, dim=1) # turns raw score into probabilities upto 1


    predicted_class = torch.argmax(probs).item() # pick the class with highest probability
    confidence = torch.max(probs).item() #gives the highest probability



    # Print result
    sentiment = "Positive" if predicted_class == 1 else "Negative"
    print(f"Text: {text}")
    print(f"Predicted Label: {predicted_class} ({sentiment}) — Confidence: {confidence:.2f}")

# 1. predict_sentiment("This is the best film of the year!")
"""
Output: Text: This is the best film of the year!
        Predicted Label: 1 (Positive) — Confidence: 0.51
"""
# 2. predict_sentiment("That was a beautiful film.")
"""
Output: Text: That was a beautiful film.
        Predicted Label: 1 (Positive) — Confidence: 0.56
"""
# 3. predict_sentiment("I hated every second of it.")
"""
Output: Text: I hated every second of it.
        Predicted Label: 0 (Negative) — Confidence: 0.56
"""
predict_sentiment("It was okay, not bad.")
"""
Output: Text: It was okay, not bad.
        Predicted Label: 1 (Positive) — Confidence: 0.51
"""
