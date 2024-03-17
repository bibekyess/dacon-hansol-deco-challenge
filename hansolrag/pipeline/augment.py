from pydantic import BaseModel

class Augment(BaseModel):
    prompt: str

    @classmethod
    def get_prompt(cls, question, raw_query_engine, prev_q=""):
        # prev_q is a must needed for some questions like this: What is the biggest cause of plaster revision? And please tell me how to solve this.”
        INSTRUCTION_PROMPT_TEMPLATE = """\
        ### System:
        벽지에 대한 고객 문의에 정확하고 유용한 답변을 작성한다. <질문>의 의도를 파악하여 정확하게 <보고서>만을 기반으로 답변하세요.

        ### User:
        <보고서>
        {CONTEXT}
        </보고서>
        지침사항을 반드시 지키고, <보고서>를 기반으로 <질문>에 답변하세요.
        <질문>
        {QUESTION}
        </질문>

        ### Assistant:
        """
        RESPONSE_TEMPLATE = """\
        {ANSWER}

        """

        response_1 = raw_query_engine.query(question)

        context_list = []
        for r in response_1.source_nodes:
            # print(r.score)
            if r.score > 0:
                if r.score <= 4 and len(context_list) >= 1:
                    pass
                else:
                    context_list.append(r.text)

        # Special case when the follow up question is junk
        if len(context_list) == 0:
            response_2 = raw_query_engine.query(prev_q + " " + question)
            for r in response_2.source_nodes:
                if r.score > 0:
                    context_list.append(r.text)

        context = prev_q + "\n\n".join(context_list + [question])

        prompt = INSTRUCTION_PROMPT_TEMPLATE.format(CONTEXT=context, QUESTION=question)   

        return cls(prompt=prompt)     