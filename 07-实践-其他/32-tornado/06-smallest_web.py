import tornado.ioloop
import tornado.web

class MainHandler(tornado.web.RequestHandler):

    def write_error(self, status_code, **kwargs):
        if status_code == 500:
            self.write("500 error custom!")

    def initialize(self):
        print('initialize()')

    def prepare(self):
        print('prepare')

    def on_finish(self):
        print('on_finish')

    # def finish(self, chunk=None):
    #     print('finish')
    #

    def get(self):
        1/0
        self.write("Hello, world")

def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(8888)
    tornado.ioloop.IOLoop.current().start()