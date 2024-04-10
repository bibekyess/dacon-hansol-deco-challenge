from hansolrag.pipeline.pipeline import Pipeline
from sentence_transformers import SentenceTransformer
import json
import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResponseGenerator:

    def answer_question(self, question: str, **kwargs) -> str:
        answer_pipeline = Pipeline()
        answer = answer_pipeline.answer(question, **kwargs)
        return answer

    def answer_questions_from_csv(self, csv_file: str, output_file: str, submission_file: str, **kwargs) -> None:
        answer_pipeline = Pipeline()
        test = pd.read_csv(csv_file)
        questions = list(test.질문)

        answer = answer_pipeline.answer(questions, **kwargs)

        if output_file:
            json_data = json.dumps(answer, ensure_ascii=False) # Displays Korean texts in json file
            with open(output_file, 'w') as json_file:
                json_file.write(json_data)

        if submission_file:
            submission = pd.read_csv(os.environ.get('SAMPLE_SUBMISSION_FILE'))
            embed_model_id = os.environ.get('TEST_EMBED_MODEL')

            logger.info("Loading embed model: %s ...", embed_model_id)
            test_embed_model = SentenceTransformer(embed_model_id)
            logger.info("Succesfully loaded embed model: %s.", embed_model_id)

            id = submission.id
            submission.drop(['id'],axis = 1, inplace = True)

            logger.info("Creating embedding for Challenge Compeition submission format ...")
            for i in range(len(submission)):
                submission.loc[i] = test_embed_model.encode(answer[i]).tolist()
            
            total_submission = pd.DataFrame({'id': id})
            sub = pd.concat([total_submission, submission] , axis = 1)
            sub.to_csv(submission_file, index = False)
            logger.info("Succesfully saved the vector responses to %s.", output_file)
