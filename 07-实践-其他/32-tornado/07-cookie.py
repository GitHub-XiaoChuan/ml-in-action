import tornado.httpserver
import tornado.ioloop
import tornado.web
import tornado.options

from tornado.options import define, options

define("port", default=8000, help="run on the given port", type=int)


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        cookie = self.get_secure_cookie("count")
        count = int(cookie) + 1 if cookie else 1
        countString = "1 time" if count == 1 else "%d times" % count


        # "2|1:0|10:1548209801|5:count|4:MjQ=|86211176f8596663cca23beb35b624e2dfc27a90bc0c490ebbd290dc2b2ab2fe"
        self.set_secure_cookie("count", str(count))

        self.write('<html><head><title>cookie</title></head><body>'
                   '<h1>You are viewed this page %s </h1>' % countString +
                   '</body></html>')


if __name__ == '__main__':
    tornado.options.parse_command_line()

    settings = {
        'cookie_secret': 'hello'
    }

    application = tornado.web.Application(handlers=[(r'/', MainHandler)], **settings)

    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()
