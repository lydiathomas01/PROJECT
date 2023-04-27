from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/result', methods=['POST'])
def result():
    input_text = request.form['input_text']
    # Process input_text using your Python program to generate the output
    output = main(input_text)
    return render_template('result.html', input_text=input_text, output=output)

if __name__ == '__main__':
    app.run(debug=True)
