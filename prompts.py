report_prompt = """
Based on the image create a radiology report describing the image.
Don't use colors overlayid on the image they are just for your reference.
Describe the information like if you were looking at a raw image without any overlays.
Make sure you referece all the finding visible on the image.
The report should be narrative not structurize.
Don't use more that 200 words
Don't include recomendations.
"""

vqa_prompt = """
Greate a set examples for Visual Questions Answering dataset based on this image.
Generate a set of questions and answers.
The answers should be complete and detailed
### FORMAT ###
Return the response as a list of jsons
[
    {{"question": "", "answer": ""}},
    {{"question": "", "answer": ""}},
    ...
]
"""