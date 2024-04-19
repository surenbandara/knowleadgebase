from LLMInference import LLAMA2Wrapper
import os

wrapper = LLAMA2Wrapper("NousResearch/Llama-2-7b-chat-hf",os.path.expanduser("~/FYP-Prasad/FYP_IntelliScript-Sever/Extracted-text-data-v3"))

wrapper.llama2_response()
