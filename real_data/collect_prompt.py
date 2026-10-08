#### Collect prompt data from CAIS/MMLU datasets

# !pip install datasets
import os
from datasets import load_dataset
import pandas as pd
import tqdm
from openai import OpenAI

# cais/mmlu
sub = "college_biology"
# "clinical_knowledge"
dataset = load_dataset("cais/mmlu", sub)
print(dataset)

questions_df = pd.DataFrame(columns=['subject','question'])
# dataset = load_dataset("cais/mmlu", sub)
for i in range(100):
    #     question = data1[i]['prompt']
    #     questions_df = pd.concat([questions_df, pd.DataFrame([{'question': question}])], ignore_index=True)
    #     # questions_df = questions_df.append({'question': question}, ignore_index=True)
    # questions_df.to_csv(f'data/prompt/questions_{sub}.csv', index=False)
    example = dataset['test'][i]
    question = example['question']
    choices = example['choices']
    # answer = example['answer']
    # subject = example['subject']
    qt0 = f"{question}{choices}"
    questions_df = pd.concat([questions_df, pd.DataFrame([{'subject':sub, 'question': qt0}])], ignore_index=True)

print(questions_df.shape)
questions_df.to_csv(f'data/prompt/questions_{sub}.csv', index=False)

#### Decode the prompt data

# do embadding
def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

import pandas as pd
from tqdm import tqdm
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Define the list of subjects
subs = ['college_biology']
# 'anatomy','clinical_knowledge', 'medical_genetics', 
# Define the function to get embeddings
def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

for sub in subs:
    # Read the CSV file
    csv_path = f'data/prompt/questions_{sub}.csv'
    df = pd.read_csv(csv_path)
    
    # Ensure the 'combined' column exists or adjust as needed
    if 'question' not in df.columns:
        raise ValueError(f"The 'combined' column does not exist in the DataFrame for {sub}")
    
    # Apply the embedding function to the 'combined' column
    tqdm.pandas()  # Add a progress bar to the lambda function
    df['embedding'] = df['question'].progress_apply(lambda x: get_embedding(x, model='text-embedding-3-small'))
    
    # Save the DataFrame with embeddings to a new CSV file
    df.to_csv(f'data/prompt/embedded_questions_{sub}.csv', index=False)

print("Embedding completed and saved for all subjects.")

subs = ['anatomy', 'clinical_knowledge', 'medical_genetics']
df0 = df[df['subject']==subs[0]]
df0.to_csv(f'data/prompt/embedded_questions_{subs[0]}.csv', index=False)

df1 = df[df['subject']==subs[1]]
df1.to_csv(f'data/prompt/embedded_questions_{subs[1]}.csv', index=False)

df2 = df[df['subject']==subs[2]]
df2.to_csv(f'data/prompt/embedded_questions_{subs[2]}.csv', index=False)

