import argparse
import os
import logging
from hansolrag.generator import ResponseGenerator


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Pass single question as string or list of questions as a csv file')
    parser.add_argument('--text', type=str, help='Input text for the question [Displays results on terminal]')
    parser.add_argument('--file', type=str, help='CSV file containing questions')
    parser.add_argument('--output-file', type=str, help='JSON path, Output file path for writing answers')
    parser.add_argument('--submission-file', type=str, help='CSV path, Makes the output in submission format for dacon contest')
    parser.add_argument('--config-file', type=str, default='hansolrag/config/config.yaml', help='Path to the configuration file')
 

    args = parser.parse_args()
    
    os.environ['SAMPLE_SUBMISSION_FILE'] = 'hansolrag/data/sample_responses/sample_submission.csv'
    os.environ['TEST_EMBED_MODEL'] = 'distiluse-base-multilingual-cased-v1'
    
    response_generator = ResponseGenerator()

    if args.text:
        answer = response_generator.answer_question(args.text, config_path=args.config_file)
        print("Generated answer: ", answer)

    elif args.file:
        response_generator.answer_questions_from_csv(args.file, args.output_file, args.submission_file, config_path=args.config_file)
    
    else:
        raise ValueError("Please provide a question using '--text' argument or specify a CSV file using '--file' argument")


if __name__ == "__main__":
    main()