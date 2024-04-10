from pydantic import BaseModel

class Augment(BaseModel):
    prompt: str

    @classmethod
    def get_prompt(cls, mode, retriever, question, prev_q=""):
        # prev_q is a must needed for some questions like this: What is the biggest cause of plaster revision? And please tell me how to solve this.”
        if mode=="gpu-solar":
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
        elif mode=="cpu-gpt2" or mode=="cpu-gemini":
            INSTRUCTION_PROMPT_TEMPLATE = """\
            <start_of_turn>user
            벽지에 대한 고객 문의에 정확하고 유용한 답변을 작성한다. <질문>의 의도를 파악하여 정확하게 <보고서>만을 기반으로 답변하세요.
            보고서: {CONTEXT}
            질문: {QUESTION}
            <start_of_turn>model
            """
        else:
            raise ValueError("Currently, only three mode names are supported: 'gpu-solar;, 'cpu-gpt2' and 'cpu-gemini'")

        response_1 = retriever.query_engine.retrieve(question)

        context_list = []
        for r in response_1:
            if r.score > 0:
                if r.score <= 4 and len(context_list) >= 1:
                    pass
                else:
                    context_list.append(r.text)

        # Special case when the follow up question is junk
        if len(context_list) == 0:
            response_2 = retriever.query_engine.retrieve(prev_q + " " + question)
            for r in response_2:
                if r.score > 0:
                    context_list.append(r.text)

        context = prev_q + "\n\n".join(context_list)

        prompt = INSTRUCTION_PROMPT_TEMPLATE.format(CONTEXT=context, QUESTION=question)   

        return cls(prompt=prompt)
    