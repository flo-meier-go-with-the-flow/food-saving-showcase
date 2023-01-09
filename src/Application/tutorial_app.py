from flask import Flask
from flask import request, make_response, render_template

app = Flask(__name__)
@app.route('/')
def index():
   return "Hello, world!"

@app.route('/profile')
def profile():
    return 'This is profile page'


@app.route('/login')
def log_in():
    return 'This is login page'

if __name__ == '__main__':
    if len(sys.argv) > 1:
        arg_host, arg_port = sys.argv[1].split(':')
        app.run(host=arg_host, port=arg_port)
    else:
        app.run()