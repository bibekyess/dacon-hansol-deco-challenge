import json
from sentence_transformers import SentenceTransformer # SentenceTransformer Version 2.2.2
import pandas as pd

def save_output(file_name, responses):
    json_data = json.dumps(responses)
    with open(file_name, 'w') as json_file:
        json_file.write(json_data)


def save_submission_format(responses, submission_file_name="./deliverable/fun_submission.csv", sample_submission="../data/sample_responses/sample_submission.csv", test_embed_model_id="distiluse-base-multilingual-cased-v1"):
    submission = pd.read_csv(sample_submission)
    test_embed_model = SentenceTransformer(test_embed_model_id)

    id = submission.id
    submission.drop(['id'],axis = 1, inplace = True)

    for i in range(len(submission)):
        submission.loc[i] = test_embed_model.encode(responses[i]).tolist()

    total_submission = pd.DataFrame({'id': id})
    sub = pd.concat([total_submission, submission] , axis = 1)

    sub.to_csv(submission_file_name, index = False) #Rename
