from flask import Flask, render_template, request, jsonify
from ai21 import AI21Client

app = Flask(__name__)


client = AI21Client(api_key="PUT YOUR AI21 KEY HERE")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        content = request.form['content']
        question = request.form['question']
        prompt = f"Respond to my Question, based on the given content. Content : {content} Question : {question}."
        genaires = answerr_genai(prompt)
        return str(genaires)
    return render_template('index.html')

def answerr_genai(prompt):
    # Call AI21 API to summarize text
    response= client.completion.create(
    model="j2-ultra",
    prompt=prompt,
    num_results=1,
    max_tokens=200,
    temperature=0.1,)
    
    # st = response['completions'][0]['data']['text']
    # st = response['completions'][0]['data']['text']
    st = response.completions[0].data.text
    return st

if __name__ == '__main__':
    app.run(debug=True)
