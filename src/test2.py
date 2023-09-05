import gradio as gr
from transformers import pipeline

translator = pipeline("translation_en_to_fr")


def translate(text):
    return translator(text)[0]['translation_text']


input_text = gr.inputs.Textbox(lines=10, label="输入文本")
output_text = gr.outputs.Textbox(label="翻译结果")

gr.Interface(fn=translate, inputs=input_text, outputs=output_text, title="文档翻译器",
             description="使用transformers模型进行英文到法文的翻译").launch()
