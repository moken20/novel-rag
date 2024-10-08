from src.utils.call_api.util_openai import (
    update_params_with_dict,
    prepare_input_messages,
    call_chatgpt_api,
)
from configulator.config import OpenAISettings

class Generator:
    def __init__(self, generation_params: list):
        base_model = OpenAISettings.openai_params
        self.step1_params = update_params_with_dict(base_model, generation_params[0])
        self.step2_params = update_params_with_dict(base_model, generation_params[1])
        self.step3_params = update_params_with_dict(base_model, generation_params[2])
        self.step4_params = update_params_with_dict(base_model, generation_params[3])
        self.step5_params = update_params_with_dict(base_model, generation_params[4])
        self.reasoning_params = update_params_with_dict(base_model, generation_params[5])

    def generate_answer(self, user_input: str, novel_info: str):
        # step1 understanding an auestion and extracting key points
        step1_messages = prepare_input_messages(prompt=self.step1_params.messages, user_input=user_input)
        step1_output = call_chatgpt_api(messages=step1_messages, model=self.step1_params.model, request_params=self.step1_params)

        # Step 2: Considerations for problem-solving
        step2_messages = prepare_input_messages(prompt=self.step2_params.messages, step1_response=step1_output)
        step2_output = call_chatgpt_api(messages=step2_messages, model=self.step2_params.model, request_params=self.step2_params)

        # Step 3: Creating the initial answer
        step3_messages = prepare_input_messages(prompt=self.step3_params.messages, step2_response=step2_output, novel_info=novel_info)
        step3_output = call_chatgpt_api(messages=step3_messages, model=self.step3_params.model, request_params=self.step3_params)

        # Step4: Reflection (Self-verification)
        step4_messages = prepare_input_messages(prompt=self.step4_params.messages,
                                                user_input=user_input,
                                                novel_info=novel_info,
                                                step3_response=step3_output
                                                )
        step4_output = call_chatgpt_api(messages=step4_messages, model=self.step4_params.model, request_params=self.step4_params)

        # Step 5: Creating the final answer
        step5_messages = prepare_input_messages(prompt=self.step5_params.messages,
                                                step3_response=step3_output,
                                                reflection_response=step4_output,
                                                novel_info=novel_info,
                                                user_input=user_input,
                                                )
        ans = call_chatgpt_api(messages=step5_messages, model=self.step5_params.model, request_params=self.step5_params)

        return ans

    def generate_reason(self, ans: str, user_input: str, novel_info: str):
        reasoning_messages = prepare_input_messages(prompt=self.reasoning_params.messages,
                                                    user_input=user_input,
                                                    novel_info=novel_info,
                                                    final_response=ans)
        reason = call_chatgpt_api(messages=reasoning_messages, model=self.reasoning_params.model, request_params=self.reasoning_params)
        return reason
