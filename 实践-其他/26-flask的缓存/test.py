import functools
from flask import Flask, request, render_template

app = Flask(__name__)

@functools.lru_cache(maxsize=2)
def test_cache(a):
    print('In test_catch: %s', a)
    return 'hello %s' % a

@app.route('/test', methods=['GET'])
def test():
    name = request.args.get('a')
    print(name)
    return test_cache(name)

if __name__ == '__main__':
    app.run('127.0.0.1', 5555)