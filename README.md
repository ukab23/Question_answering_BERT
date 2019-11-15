# Question_answering_BERT
Question Answering using BERT language model
qa.py file is for reading all text files from given folder and ask question on any one of them. You will get answer from BERT model.
api.py and qa2.py are files for question answering API.
Steps to make Question answering API.
1) execute api.py file
2) open postman api manager and put following things in it.
    url = http://127.0.0.1:5000/qa
    method = POST
    raw json body = {
    "paragraph" : "Your Paragraph",
    "question" : "Your Question"
    }
3) press send button to get answer.
you will get answer along with paragraph and question asked.
